"""API-layer tests: routing, governance, response shape, error translation.

The model is stubbed; every other stage runs for real.
"""

from __future__ import annotations

import pytest

from backend import llm_client
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
    assert "model" in body and "api_key_configured" in body


def test_an_unset_model_is_reported_with_the_reason_rather_than_a_guess(client, patch_settings):
    """Empty is a valid state here, and it has to explain itself.

    No model id is compiled in, because free ids get withdrawn and a baked-in
    default eventually becomes a 404 about a model the user never chose. The
    cost of that decision is that a fresh install has no model, so the empty
    value must arrive with the reason attached rather than looking like a bug.
    """
    patch_settings(model="")
    body = client.get("/api/health").json()

    assert body["model"] == ""
    assert body["model_warning"]
    assert "LLM_MODEL" in body["model_warning"]
    assert "/api/models" in body["model_warning"]


def test_config_never_leaks_the_api_key(client, patch_settings):
    """The secret value, and any field that could carry it.

    A substring scan for "api_key" was the old check, and it started failing on
    the honest `api_key_configured` boolean -- a false positive that would have
    been "fixed" by deleting the assertion. So look for the value itself, and
    walk the structure for a field that actually holds a key rather than one
    that merely reports whether there is one.
    """
    secret = "sk-or-v1-super-secret-value"
    patch_settings(api_key=secret)
    body = client.get("/api/config").json()
    raw = client.get("/api/config").text

    assert secret not in raw
    assert secret[:12] not in raw, "a prefix of the key leaked, which is enough to identify it"

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                assert key not in {"api_key", "apiKey", "key", "authorization", "token"}, (
                    f"{here} is a field that would carry the credential itself"
                )
                walk(value, here)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(body)
    # The boolean that replaced it is still there and still honest.
    assert body["api_key_configured"] is True


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
    patch_settings(api_key="")
    # Unstubbed: the real client must refuse before any network call.
    monkeypatch.setattr(pipeline_base, "complete", llm_client.complete)
    res = client.post("/api/log-rca", json={"input": TOOL_INPUT["log-rca"]})
    assert res.status_code == 503
    assert "OPENROUTER_API_KEY" in res.json()["error"]


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
    err = llm_client._translate_http_error(response, "some-model")
    assert err.status == expected_status
    assert needle in err.message.lower()


def test_an_unknown_model_404_points_at_the_live_catalogue():
    """The catalogue is the only source of truth, so the hint sends you there.

    Free model ids churn: a `:free` variant that resolves today can be withdrawn
    next week. Naming a specific replacement in the code would be a guess with a
    shelf life, so the error names the endpoint that is always current.
    """
    import httpx

    response = httpx.Response(
        404, json={"error": {"message": "model_not_found"}}, request=httpx.Request("POST", "https://x")
    )
    err = llm_client._translate_http_error(response, "some-vendor/withdrawn-model:free")
    assert err.status == 502
    assert "some-vendor/withdrawn-model:free" in err.message
    assert "/api/models" in err.hint
    assert "LLM_MODEL" in err.hint


def test_a_known_retired_model_explains_itself_when_one_is_recorded(monkeypatch):
    """`RETIRED_MODELS` is empty by design, so the path is tested by populating it."""
    import httpx

    from backend import config

    monkeypatch.setitem(config.RETIRED_MODELS, "vendor/gone:free", "Withdrawn on 2026-09-01.")
    response = httpx.Response(
        404, json={"error": {"message": "model_not_found"}}, request=httpx.Request("POST", "https://x")
    )
    err = llm_client._translate_http_error(response, "vendor/gone:free")
    assert "retired" in err.hint
    assert "Withdrawn on 2026-09-01." in err.hint


def test_no_model_id_is_compiled_into_the_application():
    """A baked-in default would rot, and rot into somebody else's confusing 404.

    Free ids are withdrawn without notice. Shipping one means a user who never
    chose it eventually gets an error about a model they have never heard of,
    so the app declines to guess and asks them to pick from the live list.
    """
    from backend.config import DEFAULT_MODELS

    assert set(DEFAULT_MODELS) == {"openrouter", "generic"}
    assert all(value == "" for value in DEFAULT_MODELS.values()), (
        f"a model id was compiled in: {DEFAULT_MODELS}"
    )


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
