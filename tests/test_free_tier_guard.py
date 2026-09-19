"""The deployment must never incur a charge, and that has to be provable.

"Point it at a free model and be careful" is not a guarantee, because every way
it fails is silent: a dropped `:free` suffix, a withdrawn free variant whose
paid twin answers instead, a router that picks whatever it likes, an account
with credits that spends them without ever returning an error.

So these tests do not check that free models are allowed. They check that
everything else is refused, and that the refusal happens before the socket is
opened.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend import billing, groq_client
from backend.billing import BillingRefused, FreeTierGuard, price_check

FREE = {"prompt": "0", "completion": "0", "request": "0", "image": "0"}
PAID = {"prompt": "0.0000008", "completion": "0.0000024", "request": "0", "image": "0"}

CATALOGUE = [
    {"id": "deepseek/deepseek-chat:free", "pricing": FREE, "context_length": 64_000},
    {"id": "openrouter/free", "pricing": FREE, "context_length": 131_072},
    {"id": "anthropic/claude-sonnet-4", "pricing": PAID, "context_length": 200_000},
    {"id": "openrouter/auto", "pricing": FREE, "context_length": 131_072},
    {"id": "weird/no-prices", "context_length": 8_000},
]


@pytest.fixture
def loaded_guard(monkeypatch):
    fresh = FreeTierGuard()
    fresh.load_catalogue(CATALOGUE)
    monkeypatch.setattr(billing, "guard", fresh)
    monkeypatch.setattr(groq_client, "guard", fresh)
    monkeypatch.delenv("FREE_TIER_ONLY", raising=False)
    return fresh


# --- layer 1: the price list decides --------------------------------------


def test_a_free_model_is_allowed(loaded_guard):
    loaded_guard.assert_free("deepseek/deepseek-chat:free")
    loaded_guard.assert_free("openrouter/free")


def test_a_paid_model_is_refused_with_its_own_price(loaded_guard):
    with pytest.raises(BillingRefused) as exc:
        loaded_guard.assert_free("anthropic/claude-sonnet-4")
    assert "0.0000008" in exc.value.message
    assert "FREE_TIER_ONLY" in exc.value.hint


def test_a_model_not_in_the_catalogue_is_refused(loaded_guard):
    """Unknown means refused. Assuming free is how a typo becomes a bill."""
    with pytest.raises(BillingRefused) as exc:
        loaded_guard.assert_free("deepseek/deepseek-chat")  # the :free suffix dropped
    assert "not in the provider's catalogue" in exc.value.message


def test_a_model_with_no_prices_quoted_is_refused(loaded_guard):
    with pytest.raises(BillingRefused) as exc:
        loaded_guard.assert_free("weird/no-prices")
    assert "cannot be shown to be free" in exc.value.message


def test_a_router_over_the_whole_catalogue_is_refused(loaded_guard):
    """openrouter/auto is priced at zero and selects paid models anyway.

    The pricing check cannot see that, which is exactly why naming it is not
    redundant with the check.
    """
    with pytest.raises(BillingRefused) as exc:
        loaded_guard.assert_free("openrouter/auto")
    assert "paid models" in exc.value.message
    # But the free router is not caught by the same net.
    loaded_guard.assert_free("openrouter/free")


def test_a_per_request_fee_is_caught_even_when_tokens_are_free():
    """Checking prompt and completion alone misses a flat fee per call."""
    free, why = price_check({"pricing": {"prompt": "0", "completion": "0", "request": "0.001"}})
    assert not free and "request=0.001" in why


def test_an_unreadable_price_is_treated_as_paid():
    free, why = price_check({"pricing": {"prompt": "contact us", "completion": "0"}})
    assert not free and "could not be read" in why


def test_nothing_is_callable_before_the_price_list_has_been_read(monkeypatch):
    """A catalogue that failed to load must stop the app, not be assumed free."""
    empty = FreeTierGuard()
    monkeypatch.delenv("FREE_TIER_ONLY", raising=False)
    with pytest.raises(BillingRefused) as exc:
        empty.assert_free("deepseek/deepseek-chat:free")
    assert exc.value.status == 503
    assert "could not be read" in exc.value.message


def test_the_guard_can_be_switched_off_deliberately(loaded_guard, monkeypatch):
    """Someone who wants to spend money may, but must say so."""
    monkeypatch.setenv("FREE_TIER_ONLY", "false")
    loaded_guard.assert_free("anthropic/claude-sonnet-4")


# --- layer 2: the provider enforces it too --------------------------------


def test_openrouter_is_told_to_refuse_anything_that_costs(loaded_guard):
    fields = loaded_guard.request_guard_fields("openrouter")
    assert fields == {"provider": {"max_price": {"prompt": 0, "completion": 0}}}


def test_the_ceiling_is_dropped_when_enforcement_is_off(loaded_guard, monkeypatch):
    monkeypatch.setenv("FREE_TIER_ONLY", "false")
    assert loaded_guard.request_guard_fields("openrouter") == {}


# --- layer 3: believe the invoice -----------------------------------------


def test_a_reported_cost_latches_the_guard_shut(loaded_guard):
    """The only layer that can catch a mistake the other two did not foresee."""
    loaded_guard.assert_free("deepseek/deepseek-chat:free")  # fine before

    charged = loaded_guard.observe_usage(
        "deepseek/deepseek-chat:free", {"total_tokens": 100, "cost": 0.00031}
    )
    assert charged == pytest.approx(0.00031)

    # Every later call is refused, including the model that was fine a moment ago.
    with pytest.raises(BillingRefused) as exc:
        loaded_guard.assert_free("deepseek/deepseek-chat:free")
    assert "already recorded a charge" in exc.value.message
    assert loaded_guard.public()["blocked"] is True


def test_a_zero_cost_changes_nothing(loaded_guard):
    assert loaded_guard.observe_usage("deepseek/deepseek-chat:free", {"cost": 0}) == 0.0
    assert loaded_guard.public()["blocked"] is False
    loaded_guard.assert_free("deepseek/deepseek-chat:free")


def test_usage_without_a_cost_field_is_not_an_alarm(loaded_guard):
    """Providers omit what does not apply; absence is not a charge."""
    assert loaded_guard.observe_usage("m", {"total_tokens": 10}) == 0.0
    assert loaded_guard.public()["blocked"] is False


# --- the refusal happens before any network call --------------------------


@pytest.fixture
def never_called(monkeypatch, patch_settings):
    """A transport that fails the test if it is ever reached."""
    reached = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(str(request.url))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "model": "x",
            "usage": {"total_tokens": 1},
        })

    monkeypatch.setattr(
        groq_client, "_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    patch_settings(groq_api_key="test-key")
    return reached


def test_a_paid_model_never_reaches_the_network(loaded_guard, never_called):
    with pytest.raises(BillingRefused):
        asyncio.run(
            groq_client.complete("sys", "user", model="anthropic/claude-sonnet-4", max_tokens=10)
        )
    assert never_called == [], "a paid model was actually sent to the provider"


def test_a_free_model_does_reach_the_network_and_carries_the_ceiling(loaded_guard, monkeypatch, patch_settings):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
            "model": "deepseek/deepseek-chat:free",
            "usage": {"total_tokens": 42, "cost": 0},
        })

    monkeypatch.setattr(
        groq_client, "_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    patch_settings(groq_api_key="test-key", groq_base_url="https://openrouter.ai/api/v1")

    result = asyncio.run(
        groq_client.complete("sys", "user", model="deepseek/deepseek-chat:free", max_tokens=10)
    )

    assert result["text"] == '{"ok": true}'
    assert result["cost"] == 0.0
    assert sent[0]["provider"] == {"max_price": {"prompt": 0, "completion": 0}}


# --- the daily call allowance is a budget, not a detail -------------------


def _varied_log(lines: int) -> str:
    shapes = [
        "ERROR svc-a [db] pool exhausted waited {i}ms id={i}",
        "WARN svc-b [cache] miss key=user:{i} region=eu-{i}",
        "INFO svc-c [api] GET /orders/{i} 200 in {i}ms",
        "ERROR svc-d [queue] nack job={i} reason=deadline-{i}",
    ]
    return "\n".join(
        f"2026-09-14T02:{i // 60 % 60:02d}:{i % 60:02d}Z " + shapes[i % 4].format(i=i)
        for i in range(lines)
    )


def _split_plan(calls_remaining, monkeypatch):
    from backend.core import tokens as tokens_module
    from backend.core.chunking import plan_context
    from backend.core.tokens import build_budget
    from backend.parsers.logs import parse_logs

    monkeypatch.setitem(tokens_module.CONTEXT_WINDOWS, "free/model:free", 8_000)
    monkeypatch.setattr(tokens_module, "registry", tokens_module.ModelRegistry())
    ir = parse_logs(_varied_log(600))
    budget = build_budget("free/model:free", "sys" * 400, 2_200, token_allowance_per_minute=0)
    return plan_context(ir, budget, calls_remaining_today=calls_remaining)


def test_a_split_plan_is_allowed_when_the_day_can_afford_it(monkeypatch):
    plan = _split_plan(50, monkeypatch)
    assert plan.strategy == "map_reduce"
    assert plan.estimated_calls > 1


def test_a_split_plan_is_refused_when_it_would_exceed_the_day(monkeypatch):
    """The combine tree costs as many calls again as the map phase.

    Capping the number of parts alone let a fourteen-call plan through with
    twelve calls left, because seven parts is under any part-count cap and the
    seven combine calls were simply not counted. On a fifty-call day that is the
    difference between a working afternoon and a dead one.
    """
    unlimited = _split_plan(None, monkeypatch)
    assert unlimited.estimated_calls == 14, "fixture no longer produces the case under test"

    plan = _split_plan(12, monkeypatch)

    assert plan.strategy != "map_reduce"
    assert plan.estimated_calls == 1
    assert "14 calls exceeds the 12 left in today's allowance" in plan.reason


def test_the_request_limiter_charges_every_upstream_call(client, for_tool):
    """One click is one HTTP request and many provider calls.

    The provider counts the calls. Counting clicks meant a split input recorded
    one against a fifty-a-day allowance while the provider recorded fourteen.
    """
    from backend.governance import limiter

    stub = for_tool("log-rca")
    before = limiter.snapshot("testclient")["used_today"]

    res = client.post("/api/log-rca", json={"input": "2026-01-01T00:00:00Z ERROR s [x] boom"})
    assert res.status_code == 200, res.text

    after = limiter.snapshot("testclient")["used_today"]
    assert after - before == stub.call_count, (
        f"{stub.call_count} upstream call(s) were made but {after - before} recorded"
    )
