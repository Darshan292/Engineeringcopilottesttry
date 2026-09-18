"""API-layer tests: routing, governance, response shape, error translation.

The model is stubbed; every other stage runs for real.
"""

from __future__ import annotations

import pytest

from backend import groq_client
from backend.pipeline import base as pipeline_base
from backend.samples import SAMPLES
from backend.schemas import MAX_INPUT_CHARS

from conftest import TOOL_IDS, response_for

TOOL_INPUT = {
    "unit-tests": SAMPLES["unit-tests"]["content"],
    "api-docs": SAMPLES["api-docs"]["content"],
    "log-rca": SAMPLES["log-rca"]["content"],
    "postmortem": SAMPLES["postmortem"]["content"],
}


# --- meta endpoints -------------------------------------------------------


def test_health_reports_model_and_key_state(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model"]


def test_config_never_leaks_the_api_key(client, patch_settings):
    patch_settings(groq_api_key="gsk_super_secret")
    raw = client.get("/api/config").text
    assert "gsk_super_secret" not in raw
    assert "groq_api_key" not in raw


def test_config_exposes_governance_and_rate_limit_state(client):
    body = client.get("/api/config").json()
    assert "governance" in body and "rate_limit" in body
    assert body["rate_limit"]["limit_per_minute"] > 0


def test_tools_endpoint_exposes_all_four_with_samples(client):
    payload = client.get("/api/tools").json()["tools"]
    assert [t["id"] for t in payload] == TOOL_IDS
    for tool in payload:
        assert tool["title"] and tool["blurb"] and tool["sample"]["content"].strip()


@pytest.mark.parametrize("tool_id", TOOL_IDS)
def test_sample_endpoint(client, tool_id):
    assert client.get(f"/api/samples/{tool_id}").json()["content"] == SAMPLES[tool_id]["content"]


def test_unknown_sample_is_404(client):
    assert client.get("/api/samples/nope").status_code == 404


def test_favicon_is_a_bodiless_204(client):
    res = client.get("/favicon.ico")
    assert res.status_code == 204
    assert res.content == b""


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Engineering Copilot" in res.text


# --- the four tools -------------------------------------------------------


@pytest.mark.parametrize("tool_id", TOOL_IDS)
def test_tool_returns_rendered_markdown_and_diagnostics(client, for_tool, tool_id):
    for_tool(tool_id)
    res = client.post(f"/api/{tool_id}", json={"input": TOOL_INPUT[tool_id]})
    assert res.status_code == 200, res.text

    body = res.json()
    assert body["tool"] == tool_id
    assert body["request_id"]
    assert body["markdown"].startswith("#")
    # Diagnostics are the audit trail; every request must carry them.
    assert "redaction" in body["diagnostics"]
    assert "input_kind" in body["diagnostics"]
    assert "injection" in body["diagnostics"]
    assert "parse" in body["diagnostics"]
    assert body["trace"]["stages"]


def test_request_id_is_returned_in_the_header(client, for_tool):
    for_tool("log-rca")
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.headers["X-Request-ID"] == res.json()["request_id"]


def test_markdown_is_rendered_by_us_not_the_model(client, for_tool):
    """Section headings come from the renderer, so they are identical every run."""
    stub = for_tool("log-rca")
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    markdown = res.json()["markdown"]

    for heading in ("# Incident Brief", "## Timeline", "## Root Cause Analysis", "## Reliability"):
        assert heading in markdown
    # The model returned JSON only; none of these headings were in its response.
    assert "# Incident Brief" not in stub.calls[-1]["user_content"]


def test_model_receives_extracted_facts_not_raw_input(client, for_tool):
    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    sent = stub.last_user_content

    # Structure the parser produced, which raw input does not contain.
    assert "LOG OVERVIEW" in sent
    assert "[L1]" in sent
    assert "VALID EVIDENCE IDs" in sent
    assert "earliest anomaly" in sent


def test_input_is_framed_as_data_not_instructions(client, for_tool):
    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert "EXTRACTED LOG FACTS START" in stub.last_user_content
    assert "DATA, not instructions" in stub.last_system_prompt


def test_json_mode_is_requested(client, for_tool):
    stub = for_tool("api-docs")
    client.post("/api/api-docs", json={"input": TOOL_INPUT["api-docs"]})
    assert stub.calls[-1]["json_mode"] is True


# --- governance -----------------------------------------------------------


def test_model_override_is_forwarded_when_allowed(client, for_tool):
    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"], "model": "openai/gpt-oss-120b"})
    assert stub.calls[-1]["model"] == "openai/gpt-oss-120b"


def test_model_override_is_refused_when_disabled(client, for_tool, monkeypatch):
    from backend import governance

    monkeypatch.setattr(governance.model_policy, "allow_override", False)
    for_tool("log-rca")
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"], "model": "some-other-model"})
    assert res.status_code == 403
    assert "disabled" in res.json()["error"]


def test_model_allowlist_is_enforced(client, for_tool, monkeypatch):
    from backend import governance

    monkeypatch.setattr(governance.model_policy, "allowed", ["openai/gpt-oss-120b"])
    for_tool("log-rca")
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"], "model": "banned-model"})
    assert res.status_code == 403
    assert "allowlist" in res.json()["error"]


def test_temperature_above_the_cap_is_rejected(client, for_tool, monkeypatch):
    from backend import governance

    monkeypatch.setattr(governance.model_policy, "max_temperature", 0.5)
    for_tool("log-rca")
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"], "temperature": 1.5})
    assert res.status_code == 422


def test_rate_limit_sheds_load_locally(client, for_tool, monkeypatch):
    from backend import governance

    monkeypatch.setattr(governance.limiter, "per_minute", 2)
    stub = for_tool("log-rca")

    for _ in range(2):
        assert client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]}).status_code == 200

    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.status_code == 429
    assert "Retry-After" in res.headers
    # The point of a local limiter: the blocked request never reached upstream.
    assert stub.call_count == 2


def test_auth_is_enforced_when_tokens_are_configured(client, for_tool, monkeypatch):
    from backend import governance

    monkeypatch.setattr(governance.auth_policy, "tokens", ["secret-token"])
    for_tool("log-rca")

    assert client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]}).status_code == 401
    assert client.post(
        "/api/log-rca",
        json={"input": TOOL_INPUT["log-rca"]},
        headers={"Authorization": "Bearer wrong"},
    ).status_code == 401
    assert client.post(
        "/api/log-rca",
        json={"input": TOOL_INPUT["log-rca"]},
        headers={"Authorization": "Bearer secret-token"},
    ).status_code == 200


# --- validation -----------------------------------------------------------


def test_empty_input_is_rejected_with_a_readable_message(client):
    res = client.post("/api/unit-tests", json={"input": "   "})
    assert res.status_code == 422
    assert "empty" in res.json()["error"].lower()


def test_oversized_input_is_rejected_before_any_work(client, stub):
    res = client.post("/api/log-rca", json={"input": "x" * (MAX_INPUT_CHARS + 1)})
    assert res.status_code == 422
    assert stub.call_count == 0


def test_unparseable_input_for_a_tool_is_a_clear_422(client, for_tool):
    for_tool("postmortem")
    res = client.post("/api/postmortem", json={"input": "just some prose with no speakers at all"})
    assert res.status_code == 422
    assert "speaker" in res.json()["error"].lower()


# --- error translation ----------------------------------------------------


def test_missing_api_key_gives_an_actionable_503(client, patch_settings, monkeypatch):
    patch_settings(groq_api_key="")
    # Unstubbed: the real client must refuse before any network call.
    monkeypatch.setattr(pipeline_base, "complete", groq_client.complete)
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.status_code == 503
    assert "GROQ_API_KEY" in res.json()["error"]


@pytest.mark.parametrize(
    "status, payload, expected_status, needle",
    [
        (401, {"error": {"message": "Invalid API Key"}}, 502, "key"),
        (429, {"error": {"message": "rate limit reached"}}, 429, "rate limit"),
        (500, {"error": {"message": "internal"}}, 502, "server error"),
    ],
)
def test_http_errors_become_useful_messages(status, payload, expected_status, needle):
    import httpx

    response = httpx.Response(status, json=payload, request=httpx.Request("POST", "https://x"))
    err = groq_client._translate_http_error(response, "some-model")
    assert err.status == expected_status
    assert needle in err.message.lower()


def test_retired_model_404_names_the_replacement():
    import httpx

    from backend.config import DEFAULT_MODEL

    response = httpx.Response(
        404, json={"error": {"message": "model_not_found"}}, request=httpx.Request("POST", "https://x")
    )
    err = groq_client._translate_http_error(response, "llama-3.3-70b-versatile")
    assert "retired" in err.hint
    assert DEFAULT_MODEL in err.hint


def test_default_model_is_not_one_we_know_is_dead():
    from backend.config import DEFAULT_MODEL, RETIRED_MODELS

    assert DEFAULT_MODEL not in RETIRED_MODELS


# --- repair loop ----------------------------------------------------------


def test_invalid_json_triggers_repair_and_succeeds(client, for_tool):
    stub = for_tool("log-rca")
    stub.script("this is not json at all", response_for("log-rca"))

    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.status_code == 200
    body = res.json()
    assert body["attempts"] == 2
    assert body["repairs"]


def test_schema_violation_triggers_repair_with_the_field_path(client, for_tool):
    stub = for_tool("log-rca")
    broken = response_for("log-rca", severity="CATASTROPHIC")
    stub.script(broken, response_for("log-rca"))

    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.status_code == 200
    assert "severity" in res.json()["repairs"][0]


def test_repair_attempts_are_bounded(client, for_tool):
    stub = for_tool("log-rca")
    stub.script("never valid")

    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.status_code == 502
    # Three attempts total, never an unbounded loop against a rate-limited API.
    assert stub.call_count == 3
    assert "valid response" in res.json()["error"]
