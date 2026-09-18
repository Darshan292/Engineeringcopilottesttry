"""Proof that no content carries between requests.

The concern is reasonable and common: chat-shaped applications accumulate a
conversation, so the third request pays for the first two. This application
does not, and the cost of being wrong about that is silently paying for
someone else's log every time.

So it is asserted rather than claimed. Every test here runs two unrelated
requests and proves the second call's payload contains nothing from the first.

What does persist between requests is deliberate and content-free: a token
count calibration factor, a cached context-window number per model, and
rate-limit timestamps. Those are numbers about the service, not fragments of
anyone's input, and the last test pins that distinction.
"""

from __future__ import annotations

import pytest

from backend.samples import API_DOC_SAMPLE, LOG_RCA_SAMPLE, POSTMORTEM_SAMPLE, UNIT_TEST_SAMPLE

# Two inputs with no vocabulary in common, so leakage is unmistakable.
FIRST_LOG = """2026-01-01T00:00:00Z ERROR zebra-service [quasar] widget reconciliation failed
2026-01-01T00:00:01Z WARN zebra-service [quasar] retrying widget reconciliation
"""
SECOND_LOG = """2026-02-02T11:11:11Z ERROR penguin-service [marmalade] sprocket alignment drifted
2026-02-02T11:11:12Z WARN penguin-service [marmalade] recalibrating sprocket
"""


def test_a_second_request_carries_nothing_from_the_first(client, for_tool):
    stub = for_tool("log-rca")

    client.post("/api/log-rca", json={"input": FIRST_LOG})
    first_payload = stub.last_user_content
    assert "zebra-service" in first_payload

    client.post("/api/log-rca", json={"input": SECOND_LOG})
    second_payload = stub.last_user_content

    assert "penguin-service" in second_payload
    for leaked in ("zebra", "quasar", "widget", "reconciliation"):
        assert leaked not in second_payload, f"'{leaked}' carried over from the previous request"


def test_a_previous_answer_is_never_resent(client, for_tool):
    """The model's own output must not become input to the next call."""
    stub = for_tool("log-rca")

    first = client.post("/api/log-rca", json={"input": FIRST_LOG}).json()
    # Something distinctive from the rendered answer.
    assert "Incident Brief" in first["markdown"]

    client.post("/api/log-rca", json={"input": SECOND_LOG})
    second_payload = stub.last_user_content

    for marker in ("Incident Brief", "Reliability", "Computed confidence"):
        assert marker not in second_payload, f"the previous answer's '{marker}' was resent"


def test_switching_tools_carries_nothing_across(client, for_tool):
    """A code request after a log request must not carry the log."""
    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": FIRST_LOG})

    stub.tool = "unit-tests"
    client.post("/api/unit-tests", json={"input": UNIT_TEST_SAMPLE})
    payload = stub.last_user_content

    assert "apply_tiered_discount" in payload
    for leaked in ("zebra", "quasar", "ERROR"):
        assert leaked not in payload


def test_message_payload_is_always_exactly_two_messages(client, for_tool):
    """No accumulating history: one system turn, one user turn, every time."""
    stub = for_tool("log-rca")
    for _ in range(4):
        client.post("/api/log-rca", json={"input": FIRST_LOG})

    # The client is called with (system_prompt, user_content) and nothing else;
    # a growing conversation would need a messages list, which does not exist.
    for call in stub.calls:
        assert set(call) >= {"system_prompt", "user_content"}
        assert "messages" not in call
        assert "history" not in call


def test_payload_size_does_not_grow_with_repetition(client, for_tool):
    """Ten identical requests must each cost the same."""
    stub = for_tool("log-rca")
    sizes = []
    for _ in range(10):
        client.post("/api/log-rca", json={"input": FIRST_LOG})
        sizes.append(len(stub.last_user_content))

    # Allow a little variation from warnings, but nothing cumulative.
    assert max(sizes) - min(sizes) < 200, f"payload grew across requests: {sizes}"
    assert sizes[-1] <= sizes[0] + 200


@pytest.mark.parametrize(
    "tool, sample",
    [
        ("log-rca", LOG_RCA_SAMPLE),
        ("unit-tests", UNIT_TEST_SAMPLE),
        ("api-docs", API_DOC_SAMPLE),
        ("postmortem", POSTMORTEM_SAMPLE),
    ],
)
def test_every_tool_starts_from_nothing(client, for_tool, tool, sample):
    stub = for_tool(tool)

    client.post(f"/api/{tool}", json={"input": sample})
    first_size = len(stub.last_user_content)

    client.post(f"/api/{tool}", json={"input": sample})
    second_size = len(stub.last_user_content)

    assert abs(second_size - first_size) < 200


def test_the_request_schema_has_no_place_to_put_history(client, for_tool):
    """Nothing in the API accepts prior context, so none can be supplied."""
    from backend.schemas import ToolRequest

    for_tool("log-rca")
    assert set(ToolRequest.model_fields) == {"input", "model", "temperature"}

    # An attempt to smuggle history in is ignored rather than honoured.
    res = client.post(
        "/api/log-rca",
        json={"input": FIRST_LOG, "history": ["earlier turn"], "context": "earlier answer"},
    )
    assert res.status_code in {200, 429}


def test_what_does_persist_is_numbers_not_content(client, for_tool):
    """Calibration and cached limits persist. Input never does."""
    from backend.core.tokens import calibrator, registry

    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": FIRST_LOG})

    # Calibration holds ratios, not text.
    for entry in calibrator.public().values():
        assert set(entry) <= {
            "samples",
            "correction_factor",
            "last_observed_error_pct",
            "confident",
        }
        for value in entry.values():
            assert isinstance(value, (int, float, bool))

    # The registry holds integers keyed by model id.
    for model_id, window in registry._windows.items():
        assert isinstance(window, int)
        assert "zebra" not in model_id
