"""Shared fixtures.

The model is stubbed at `pipeline.base.complete` -- the single point where the
pipeline talks to the network. Everything upstream of it (redaction, parsing,
budgeting, context planning) and everything downstream (grounding, confidence,
execution, rendering) runs for real in every test.

`VALID_RESPONSES` holds a schema-valid response per tool. Tests that need a
specific failure override it, which is how fabricated citations, overconfidence
and repair loops are exercised without a network call.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from backend import groq_client
from backend import main as main_module
from backend.main import app
from backend.pipeline import base as pipeline_base
from backend.routes import tools as tools_route

TOOL_IDS = ["unit-tests", "api-docs", "log-rca", "postmortem"]


VALID_RESPONSES: dict[str, dict] = {
    "log-rca": {
        "summary": "Checkout failed for roughly twelve minutes after a reindex job saturated the shared connection pool.",
        "severity": "SEV2",
        "severity_rationale": "Total checkout failure, fully recovered within 15 minutes.",
        "affected_components": ["checkout-api", "payments-gw"],
        "user_impact": "Checkout returned 500 for about 12 minutes.",
        "timeline": [
            {"evidence_id": "L3", "time": "02:15:02", "event": "First slow query against stock_levels"},
            {"evidence_id": "L7", "time": "02:16:45", "event": "First checkout 500"},
        ],
        "root_cause": {
            "statement": "The nightly reindex ran a full scan on the primary and exhausted the pool.",
            "evidence_ids": ["L2", "L3"],
        },
        "model_confidence": "medium",
        "confidence_rationale": "Timing is consistent but no database metrics are present.",
        "supporting_evidence": [
            {"statement": "Reindex began four minutes before the first error.", "evidence_ids": ["L2"]},
            {"statement": "Pool utilisation rose monotonically.", "evidence_ids": ["L4"]},
        ],
        "contradicting_evidence": [
            {"statement": "Adding replicas worsened saturation.", "evidence_ids": ["L20"]}
        ],
        "alternative_hypotheses": [
            {
                "statement": "An independent proxy connection ceiling.",
                "supporting_evidence_ids": ["L21"],
                "contradicting_evidence_ids": [],
                "how_to_confirm": "Check proxy connection counts before 02:14.",
            }
        ],
        "immediate_actions": [{"action": "Cancel the reindex job", "owner_role": "[ONCALL]", "rationale": "Frees pool capacity"}],
        "followup_actions": [{"action": "Repoint reindex at the replica", "owner_role": "[DB_OWNER]", "rationale": "Removes contention"}],
        "evidence_gaps": ["No proxy connection metrics before 02:14."],
    },
    "postmortem": {
        "title": "Checkout unavailable due to reindex on primary",
        "date": "2026-09-14",
        "duration": "12 minutes",
        "severity": "SEV2",
        "status": "Mitigated",
        "incident_commander_role": "[INCIDENT_COMMANDER]",
        "services_affected": ["checkout-api"],
        "users_affected": "About 340 failed checkout attempts.",
        "impact_duration": "02:16 to 02:28",
        "business_impact": "Failed checkouts and a consumer backlog.",
        "data_integrity": "[NOT DISCUSSED]",
        "timeline": [
            {"evidence_id": "U1", "time": "02:18", "actor_role": "[INCIDENT_COMMANDER]", "event": "Paged for 500s"},
            {"evidence_id": "U17", "time": "02:26", "actor_role": "[SERVICE_OWNER]", "event": "Reindex cancelled"},
        ],
        "trigger": {"statement": "The reindex ran against the primary.", "evidence_ids": ["U15"]},
        "root_cause": {"statement": "A config change repointed the job at the primary and no control caught it.", "evidence_ids": ["U25"]},
        "contributing_factors": [{"statement": "No alert on pool saturation.", "evidence_ids": ["U24"]}],
        "detection_gap": {"statement": "Detection came from customer 500s, not an internal signal.", "evidence_ids": ["U24"]},
        "resolution": "Service recovered when the reindex was cancelled.",
        "resolution_is_permanent_fix": False,
        "what_went_well": ["IC assigned within a minute."],
        "what_went_poorly": ["Scaling made saturation worse."],
        "where_we_got_lucky": ["It happened off-peak."],
        "action_items": [
            {"action": "Alert on pool utilisation above 80% for 2 minutes", "type": "Detect", "owner_role": "[DB_OWNER]", "priority": "P0", "rationale": "Closes the detection gap"}
        ],
        "open_questions": ["How many failed checkouts were retried?"],
    },
    "unit-tests": {
        "language": "python",
        "framework": "pytest",
        "run_command": "pytest -q",
        "cases": [
            {"name": "test_empty_order_raises", "category": "error_handling", "behaviour": "Empty list raises ValueError", "target_function_id": "F1", "covers_boundary": "customer_tier not in tier_rates"},
            {"name": "test_promo_boundary", "category": "edge_case", "behaviour": "Promo excluded at exactly 10000", "target_function_id": "F1", "covers_boundary": "subtotal > 10000"},
        ],
        "test_code": (
            "import pytest\n"
            "from pricing import apply_tiered_discount, LineItem\n\n"
            "def test_empty_order_raises():\n"
            "    with pytest.raises(ValueError):\n"
            "        apply_tiered_discount([], 'gold')\n\n"
            "def test_promo_boundary():\n"
            "    r = apply_tiered_discount([LineItem('a', 10000, 1)], 'gold', 'SAVE20')\n"
            "    assert r['applied_rate'] == 0.10\n"
        ),
        "untestable": ["date.today() default makes results time-dependent."],
        "assumptions": ["Module is importable as 'pricing'."],
    },
    "api-docs": {
        "overview": "Deployment management API.",
        "openapi_yaml": (
            "openapi: 3.1.0\n"
            "info:\n  title: Deployments API\n  version: '1.0.0'\n"
            "servers:\n  - url: https://api.example.com\n"
            "paths:\n"
            "  /v1/deployments:\n"
            "    get:\n      summary: List\n      responses:\n        '200':\n          description: OK\n"
            "    post:\n      summary: Create\n      responses:\n        '201':\n          description: Created\n"
            "  /v1/deployments/{deployment_id}/rollback:\n"
            "    post:\n      summary: Rollback\n      parameters:\n        - name: deployment_id\n          in: path\n          required: true\n          schema:\n            type: string\n      responses:\n        '200':\n          description: OK\n"
        ),
        "endpoints": [
            {"route_id": "R1", "method": "GET", "path": "/v1/deployments", "purpose": "List deployments", "auth": "Bearer", "parameters": [], "success_example": "{}", "errors": [], "curl_example": "curl ..."},
            {"route_id": "R2", "method": "POST", "path": "/v1/deployments", "purpose": "Create", "auth": "Bearer", "parameters": [], "success_example": "{}", "errors": [{"status": 409, "condition": "In progress", "body": "{}"}], "curl_example": "curl ..."},
            {"route_id": "R3", "method": "POST", "path": "/v1/deployments/{deployment_id}/rollback", "purpose": "Roll back", "auth": "Bearer", "parameters": [], "success_example": "{}", "errors": [], "curl_example": "curl ..."},
        ],
        "notes": ["Types came from pydantic models."],
    },
}


class StubModel:
    """Records what was sent and replays a scripted response."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: list[dict] = []
        self.tool: str | None = None

    def script(self, *payloads: dict) -> None:
        """Queue responses; the last one repeats once exhausted."""
        self.responses = list(payloads)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_user_content(self) -> str:
        return self.calls[-1]["user_content"] if self.calls else ""

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1]["system_prompt"] if self.calls else ""

    async def __call__(self, system_prompt, user_content, **kwargs):
        self.calls.append({"system_prompt": system_prompt, "user_content": user_content, **kwargs})

        if self.responses:
            payload = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        else:
            payload = VALID_RESPONSES[self.tool]

        text = payload if isinstance(payload, str) else json.dumps(payload)

        # Report usage proportional to what was actually sent. A fixed small
        # number made every cost and budget assertion meaningless -- the token
        # limiter would never fill, so a test could not tell a working budget
        # from a broken one.
        from backend.core.tokens import estimate_tokens

        prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_content)
        completion_tokens = estimate_tokens(text)
        return {
            "text": text,
            "model": "stub-model",
            "finish_reason": "stop",
            "elapsed_ms": 42,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


@pytest.fixture
def stub(monkeypatch) -> StubModel:
    model = StubModel()
    monkeypatch.setattr(pipeline_base, "complete", model)
    return model


@pytest.fixture
def for_tool(stub):
    """Point the stub at a tool's canned valid response."""

    def _set(tool: str) -> StubModel:
        stub.tool = tool
        return stub

    return _set


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def patch_settings(monkeypatch):
    """Settings is a frozen dataclass, so swap the whole object, not a field."""

    def _apply(**overrides):
        replaced = dataclasses.replace(groq_client.settings, **overrides)
        monkeypatch.setattr(groq_client, "settings", replaced)
        monkeypatch.setattr(tools_route, "settings", replaced)
        monkeypatch.setattr(main_module, "settings", replaced)
        return replaced

    return _apply


@pytest.fixture(autouse=True)
def reset_process_state():
    """Reset every piece of process-global state between tests.

    The pipeline deliberately carries state across requests -- the token
    calibrator learns from observed usage and the model registry caches what
    the provider reported -- which is correct at runtime and poisonous in a
    test suite. A calibrated estimator narrows the safety margin, which changes
    the computed budget, which changes whether an input chunks or compresses.
    Without this, a test's result depended on which tests ran before it.
    """
    from backend.core.tokens import calibrator, registry
    from backend.governance import limiter, token_limiter

    # The token budget defaults to the Groq free tier's 8,000/minute, which a
    # multi-call test legitimately exceeds. Tests get an effectively unlimited
    # budget; `test_api.py` exercises the real ceiling explicitly.
    token_limiter.per_minute = 10_000_000
    token_limiter.per_day = 10_000_000

    def clear():
        limiter._minute.clear()
        limiter._day.clear()
        token_limiter._minute.clear()
        token_limiter._day.clear()
        with calibrator._lock:
            calibrator._models.clear()
        with registry._lock:
            registry._windows.clear()
            registry._fetched_at = 0.0
            registry._attempted_at = 0.0

    clear()
    yield
    clear()


def response_for(tool: str, **overrides) -> dict:
    """A valid response with fields replaced, for failure-mode tests."""
    payload = copy.deepcopy(VALID_RESPONSES[tool])
    payload.update(overrides)
    return payload
