"""A rate limit is a wait, not a failure.

The provider tells us exactly when to come back -- "Please try again in
4.303999999s" -- and the application used to throw that away and surface the
error. On a free tier, where hitting the ceiling is the normal case rather than
the exceptional one, that turned a four-second pause into a failed request and
made the whole tool look unusable.

The bargain this application offers is that a small allowance costs time, not
output. These tests hold it to that.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend import llm_client
from backend.llm_client import LLMError, complete

RATE_LIMIT_BODY = {
    "error": {
        "message": (
            "Rate limit reached for model `meta-llama/llama-4-scout-17b-16e-instruct` in "
            "organization `org_01khkf68pfefpvcb9mn0fccezs` service tier `on_demand` on tokens "
            "per minute (TPM): Limit 30000, Used 18391, Requested 13761. Please try again in "
            "4.303999999s."
        ),
        "type": "rate_limit_exceeded",
    }
}

GOOD_BODY = {
    "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
    "model": "openrouter/free",
    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
}


@pytest.fixture
def no_real_sleep(monkeypatch):
    """Record what would have been waited instead of actually waiting."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_client, "_sleep", fake_sleep)
    return slept


def _transport(responses):
    """An httpx transport that replays a scripted list of responses."""
    calls = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        status, body, headers = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(status, json=body, headers=headers or {})

    return handler, calls


@pytest.fixture
def scripted(monkeypatch, patch_settings):
    def install(responses):
        handler, calls = _transport(responses)

        def client_factory():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(llm_client, "_client", client_factory)
        # Settings is a frozen dataclass, so the whole object is swapped.
        patch_settings(api_key="test-key")
        return calls

    return install


def test_a_rate_limit_is_waited_out_not_surfaced(scripted, no_real_sleep):
    calls = scripted([
        (429, RATE_LIMIT_BODY, None),
        (200, GOOD_BODY, None),
    ])

    result = asyncio.run(complete("sys", "user", model="openrouter/free", max_tokens=100))

    assert result["text"] == '{"ok": true}'
    assert len(calls) == 2, "the request was not retried"
    # Waited the time the provider asked for, not an invented backoff.
    assert no_real_sleep and 4.3 <= no_real_sleep[0] <= 5.0, no_real_sleep
    assert result["rate_limit_retries"] == 1
    assert result["rate_limited_seconds"] >= 4.3


def test_repeated_rate_limits_are_retried_until_the_budget_runs_out(scripted, no_real_sleep):
    """Bounded: waiting forever is its own kind of failure."""
    scripted([(429, RATE_LIMIT_BODY, None)])

    with pytest.raises(LLMError) as exc:
        asyncio.run(complete("sys", "user", model="openrouter/free", max_tokens=100, max_wait_seconds=10))

    # 10s of budget against ~4.55s waits allows two, then gives up.
    assert len(no_real_sleep) == 2, no_real_sleep
    assert exc.value.status == 429
    assert "did not clear after waiting" in exc.value.message


def test_the_wait_comes_from_the_retry_after_header_when_present(scripted, no_real_sleep):
    scripted([
        (429, RATE_LIMIT_BODY, {"retry-after": "12"}),
        (200, GOOD_BODY, None),
    ])
    asyncio.run(complete("sys", "user", model="openrouter/free", max_tokens=100))
    assert 12 <= no_real_sleep[0] <= 13


def test_a_routing_model_reports_which_model_actually_hit_the_limit(scripted, no_real_sleep):
    """'I chose openrouter/free, why does it name llama-4-scout?' has an answer."""
    scripted([(429, RATE_LIMIT_BODY, None)])

    with pytest.raises(LLMError) as exc:
        asyncio.run(complete("sys", "user", model="openrouter/free", max_tokens=100, max_wait_seconds=0))

    assert "openrouter/free" in exc.value.hint
    assert "meta-llama/llama-4-scout-17b-16e-instruct" in exc.value.hint
    assert "routed to" in exc.value.hint


def test_the_real_cost_is_learned_from_the_rate_limit_body(scripted, no_real_sleep):
    """An agentic model's wire cost cannot be learned from successful calls.

    While the estimate is too low, the calls fail; a failed call reports no
    usage. The 429 states what the provider counted, which breaks the deadlock.
    """
    from backend.core.tokens import calibrator, estimate_messages_tokens

    scripted([(429, RATE_LIMIT_BODY, None)])
    before = calibrator.for_model("openrouter/free").samples

    # A realistic prompt, so the correction is a plausible 2x rather than a
    # 100x that the outlier guard would (rightly) discard. This mirrors the
    # real observation: a 6,044-token estimate the provider counted as 13,761,
    # because an agentic model prepends tool schemas we never see.
    system_prompt = "You are a careful engineering assistant. " * 700
    ours = estimate_messages_tokens(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": "analyse this"}]
    ) + 4096
    assert 4_000 < ours < 13_761, f"test prompt is not a realistic size: {ours}"

    with pytest.raises(LLMError):
        asyncio.run(
            complete(system_prompt, "analyse this", model="openrouter/free",
                     max_tokens=4096, max_wait_seconds=0)
        )

    cal = calibrator.for_model("openrouter/free")
    assert cal.samples > before, "nothing was learned from the provider's own arithmetic"
    assert cal.ratio > 1.0, "the estimate was under and should have been corrected upward"
    # And the correction is roughly the ratio the provider's own numbers imply.
    assert cal.ratio == pytest.approx(13_761 / ours, rel=0.05)


def test_an_implausible_correction_is_discarded(scripted, no_real_sleep):
    """The guard that made the test above need a realistic prompt.

    A ratio of 100x does not mean the estimator is 100x wrong; it means the
    payload measured was not the payload sent. Believing it would poison every
    later budget for that model.
    """
    from backend.core.tokens import calibrator

    scripted([(429, RATE_LIMIT_BODY, None)])
    before = calibrator.for_model("openrouter/free").samples

    with pytest.raises(LLMError):
        asyncio.run(complete("s", "u", model="openrouter/free", max_tokens=10, max_wait_seconds=0))

    assert calibrator.for_model("openrouter/free").samples == before


# --- the configured ceiling is a decision, not a suggestion ----------------


@pytest.fixture
def limits(monkeypatch):
    """A fresh limiter plus control over what was 'configured'."""
    from backend import governance
    from backend.governance import TokenBudgetLimiter

    def install(configured_tpm):
        limiter = TokenBudgetLimiter(per_minute=configured_tpm or 8_000, per_day=200_000)
        monkeypatch.setattr(governance, "token_limiter", limiter)
        monkeypatch.setattr(governance, "CONFIGURED_TPM", configured_tpm)
        monkeypatch.setattr(governance, "CONFIGURED_TPD", None)
        return governance, limiter

    return install


def test_a_configured_token_limit_is_never_raised_by_the_provider(limits):
    """The bug: TOKEN_LIMIT_PER_MINUTE=8000 and the UI showed 70,000.

    Adoption overwrote the configured value outright, so an operator who
    deliberately paced below their real ceiling had no way to make it stick --
    and for a routing model the advertised number is not even the one that
    binds, so the app budgeted against a limit that did not apply.
    """
    governance, limiter = limits(8_000)

    result = governance.adopt_provider_limits({"x-ratelimit-limit-tokens": "70000"})

    assert limiter.per_minute == 8_000, "the provider overruled an explicit configuration"
    assert result["provider_reported"] == 70_000
    assert result["capped_by_config"] is True


def test_the_provider_may_still_lower_the_limit(limits):
    """Discovery informs the budget; it just cannot lift it above the decision."""
    governance, limiter = limits(8_000)

    governance.adopt_provider_limits({"x-ratelimit-limit-tokens": "3000"})

    assert limiter.per_minute == 3_000


def test_with_nothing_configured_the_provider_is_believed(limits):
    """No decision to overrule, so the real account limit is better than a guess."""
    governance, limiter = limits(None)

    governance.adopt_provider_limits({"x-ratelimit-limit-tokens": "70000"})

    assert limiter.per_minute == 70_000


def test_the_limit_named_in_a_429_is_adopted_and_still_capped(limits):
    """The 429 names the model that actually ran and the limit that actually bound."""
    governance, limiter = limits(8_000)

    learned = governance.adopt_limit_from_rate_limit_error(
        "openrouter/free", RATE_LIMIT_BODY["error"]["message"]
    )

    assert learned["limit"] == 30_000
    assert learned["requested"] == 13_761
    assert learned["routed_to"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert limiter.per_minute == 8_000, "a configured ceiling still wins"


def test_a_429_without_arithmetic_changes_nothing(limits):
    governance, limiter = limits(8_000)
    assert governance.adopt_limit_from_rate_limit_error("m", "slow down") is None
    assert limiter.per_minute == 8_000


def test_a_plain_model_is_not_calibrated_from_another_model_s_limit(scripted, no_real_sleep):
    """Attributing someone else's cost poisons every later budget.

    A 429 naming a different model is the truth about a router, which really
    did dispatch there and really did pay that. For a plain chat model it is
    something else -- a shared organisation bucket, a proxy -- and believing it
    inflated the system reserve until there was no input budget left at all, for
    every subsequent request in the process.
    """
    from backend.core.tokens import calibrator

    scripted([(429, RATE_LIMIT_BODY, None)])
    system_prompt = "You are a careful engineering assistant. " * 700
    before = calibrator.for_model("openai/gpt-oss-120b").samples

    with pytest.raises(LLMError):
        asyncio.run(
            complete(system_prompt, "analyse this", model="openai/gpt-oss-120b",
                     max_tokens=4096, max_wait_seconds=0)
        )

    cal = calibrator.for_model("openai/gpt-oss-120b")
    assert cal.samples == before, "a plain model was calibrated from a routed model's cost"
    assert cal.ratio == 1.0


def test_the_limit_itself_is_still_adopted_for_a_plain_model(scripted, no_real_sleep, limits):
    """Declining to calibrate must not mean ignoring the limit that bound."""
    governance, limiter = limits(None)
    scripted([(429, RATE_LIMIT_BODY, None)])

    with pytest.raises(LLMError):
        asyncio.run(complete("s", "u", model="openai/gpt-oss-120b", max_tokens=10, max_wait_seconds=0))

    assert limiter.per_minute == 30_000
