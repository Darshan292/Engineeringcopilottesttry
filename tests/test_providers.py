"""Providers agree on the wire format and on nothing around it.

Every endpoint here speaks OpenAI-compatible chat completions, which is why
swapping one for another looks trivial. The failures live in the margins: how a
provider says "come back later", what it rations, what it calls a context
window. Those are the things this file pins, because each one has a failure mode
that is silent or absurd rather than obvious.

The worst of them is the reset header. Groq states a duration; OpenRouter states
a Unix millisecond timestamp. Reading the second as the first is a sleep of
roughly fifty thousand years, and nothing about the code would look wrong.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from backend import groq_client
from backend.billing import price_check
from backend.groq_client import GroqError, complete
from backend.providers import (
    GENERIC,
    GROQ,
    OPENROUTER,
    detect_provider,
    seconds_until_reset,
)

GOOD_BODY = {
    "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
    "model": "deepseek/deepseek-chat:free",
    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
}

# OpenRouter's free-tier per-minute 429: no retry-after, no body arithmetic,
# just a reset instant as a Unix millisecond timestamp.
OPENROUTER_429 = {"error": {"message": "Rate limit exceeded: free-models-per-min", "code": 429}}


# --- reading the reset header ---------------------------------------------


def test_a_millisecond_timestamp_is_not_read_as_a_duration():
    """The bug that would have hidden longest and hurt most."""
    now = time.time()
    raw = str(int((now + 42) * 1000))

    seconds = seconds_until_reset(raw, absolute=True, now=now)

    assert seconds == pytest.approx(42, abs=1)
    assert seconds < 120, "a reset instant was read as a duration"


def test_the_magnitude_overrules_a_stale_profile_flag():
    """A wrong flag in a table must not be able to cause an absurd sleep.

    Profiles go stale. This one going stale would mean sleeping for millennia,
    so the value's own magnitude is checked as well as the flag.
    """
    now = time.time()
    raw = str(int((now + 42) * 1000))

    seconds = seconds_until_reset(raw, absolute=False, now=now)

    assert seconds == pytest.approx(42, abs=1)


@pytest.mark.parametrize(
    "raw, expected",
    [("250ms", 0.25), ("4.3", 4.3), ("4.3s", 4.3), ("1m", 60.0), ("2h", 7200.0)],
)
def test_durations_are_read_as_durations(raw, expected):
    assert seconds_until_reset(raw, absolute=False) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", None, "soon", "later-today"])
def test_an_unparseable_reset_is_refused_rather_than_guessed(raw):
    assert seconds_until_reset(raw, absolute=True) is None


def test_a_reset_already_in_the_past_is_zero_not_negative():
    now = time.time()
    assert seconds_until_reset(str(int((now - 30) * 1000)), absolute=True, now=now) == 0.0


# --- picking a provider ----------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://openrouter.ai/api/v1", OPENROUTER),
        ("https://api.groq.com/openai/v1", GROQ),
        ("http://127.0.0.1:11434/v1", GENERIC),
        ("https://some-vendor.example.com/v1", GENERIC),
    ],
)
def test_the_provider_is_inferred_from_the_endpoint(url, expected, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert detect_provider(url) is expected


def test_an_explicit_provider_wins_over_the_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert detect_provider("https://api.groq.com/openai/v1") is OPENROUTER


def test_an_unknown_endpoint_is_not_assumed_to_be_groq(monkeypatch):
    """Guessing Groq's rules for someone else's endpoint is how the margins bite."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert detect_provider("https://llm.internal.example/v1").name == "generic"


# --- what each provider rations -------------------------------------------


def test_openrouter_rations_requests_and_groq_rations_tokens():
    """Budgeting the wrong one produces refusals unrelated to the real limit."""
    assert OPENROUTER.default_tokens_per_minute == 0, "OpenRouter states no token ceiling"
    assert OPENROUTER.default_requests_per_day < GROQ.default_requests_per_day
    assert GROQ.default_tokens_per_minute > 0


def test_a_provider_without_a_token_ceiling_uses_the_whole_context_window():
    """Zero must mean 'not rationed', not 'rationed to nothing'."""
    from backend.core.tokens import build_budget

    unrationed = build_budget("openai/gpt-oss-120b", "sys", 2_000, token_allowance_per_minute=0)
    rationed = build_budget("openai/gpt-oss-120b", "sys", 2_000, token_allowance_per_minute=8_000)

    assert unrationed.available_for_input > rationed.available_for_input
    assert unrationed.effective_window == unrationed.context_window


# --- reading a model list --------------------------------------------------


@pytest.mark.parametrize(
    "entry, expected",
    [
        ({"context_window": 131_072}, 131_072),
        ({"context_length": 163_840}, 163_840),
        ({"top_provider": {"context_length": 65_536}}, 65_536),
        ({"max_context_length": 8_192}, 8_192),
        ({}, None),
    ],
)
def test_a_context_window_is_found_under_any_provider_s_name(entry, expected):
    assert groq_client._context_length_of(entry) == expected


@pytest.mark.parametrize(
    "entry, expected, note",
    [
        ({"pricing": {"prompt": "0", "completion": "0"}}, True, "every quoted rate is zero"),
        (
            {"pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            False,
            "it charges for tokens",
        ),
        (
            {"pricing": {"prompt": "0", "completion": "0", "request": "0.001"}},
            False,
            "free tokens, but a fee per call",
        ),
        ({}, False, "no prices quoted, so nothing can be verified"),
        (
            {"pricing": {"prompt": None, "completion": "0"}},
            True,
            "an omitted rate is one that does not apply",
        ),
        (
            {"pricing": {"prompt": "contact sales", "completion": "0"}},
            False,
            "a rate that cannot be parsed cannot be confirmed zero",
        ),
    ],
)
def test_free_is_decided_by_the_price_list_alone(entry, expected, note):
    """A ':free' suffix is a naming convention, not a guarantee.

    The old rule accepted either the suffix or the prices, which is fine for a
    label and wrong for a billing decision: the suffix is a string the operator
    can typo and the provider can reuse, while the price list is the thing that
    determines what gets charged. Only the price list counts now, and a model
    with no prices quoted is refused rather than assumed free.
    """
    assert price_check(entry)[0] is expected, note


def test_the_free_suffix_alone_does_not_make_a_model_free():
    """Named separately because it is the specific regression that matters."""
    assert price_check({"id": "vendor/model:free"})[0] is False
    assert price_check({"id": "vendor/model:free", "pricing": {"prompt": "0.5"}})[0] is False


# --- the client, end to end ------------------------------------------------


@pytest.fixture
def openrouter(monkeypatch, patch_settings):
    """Point the client at OpenRouter with a scripted transport."""

    def install(responses):
        queue = list(responses)
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            status, body, headers = queue.pop(0) if len(queue) > 1 else queue[0]
            return httpx.Response(status, json=body, headers=headers or {})

        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setattr(
            groq_client, "_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        patch_settings(
            groq_api_key="test-key",
            groq_base_url="https://openrouter.ai/api/v1",
            groq_model="deepseek/deepseek-chat:free",
        )
        return calls

    return install


@pytest.fixture
def no_real_sleep(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(groq_client, "_sleep", fake_sleep)
    return slept


def test_openrouter_attribution_headers_are_sent(openrouter, no_real_sleep):
    calls = openrouter([(200, GOOD_BODY, None)])

    asyncio.run(complete("sys", "user", max_tokens=100))

    assert calls[0].headers.get("HTTP-Referer")
    assert calls[0].headers.get("X-Title")


def test_a_429_with_no_retry_after_is_still_retried(openrouter, no_real_sleep):
    """OpenRouter's free-tier 429 states nothing. Silence is not a refusal.

    Treating "no stated delay" as "give up" would reproduce, on a different
    provider, exactly the bug the retry loop was written to fix.
    """
    calls = openrouter([(429, OPENROUTER_429, None), (200, GOOD_BODY, None)])

    result = asyncio.run(complete("sys", "user", max_tokens=100))

    assert result["text"] == '{"ok": true}'
    assert len(calls) == 2, "a 429 with no stated delay was not retried"
    assert no_real_sleep, "retried without waiting at all"
    assert 1 <= no_real_sleep[0] <= 65


def test_a_429_reset_header_is_honoured_as_an_instant(openrouter, no_real_sleep):
    reset_at = int((time.time() + 18) * 1000)
    calls = openrouter([
        (429, OPENROUTER_429, {"x-ratelimit-reset": str(reset_at)}),
        (200, GOOD_BODY, None),
    ])

    asyncio.run(complete("sys", "user", max_tokens=100))

    assert len(calls) == 2
    # ~18s, not ~1.8 billion.
    assert no_real_sleep[0] == pytest.approx(18, abs=2)


def test_blind_retries_back_off_rather_than_hammering(openrouter, no_real_sleep):
    openrouter([(429, OPENROUTER_429, None)])

    with pytest.raises(GroqError):
        asyncio.run(complete("sys", "user", max_tokens=100, max_wait_seconds=200))

    assert len(no_real_sleep) > 1
    assert no_real_sleep == sorted(no_real_sleep), f"backoff did not increase: {no_real_sleep}"


def test_a_missing_model_is_refused_before_the_request(openrouter, no_real_sleep, patch_settings):
    """An empty model on the wire is someone else's confusing error."""
    calls = openrouter([(200, GOOD_BODY, None)])
    patch_settings(
        groq_api_key="test-key",
        groq_base_url="https://openrouter.ai/api/v1",
        groq_model="",
    )

    with pytest.raises(GroqError) as exc:
        asyncio.run(complete("sys", "user", max_tokens=100))

    assert calls == [], "a request went out with no model set"
    assert exc.value.status == 503
    assert "LLM_MODEL" in exc.value.hint
    assert "/api/models" in exc.value.hint
