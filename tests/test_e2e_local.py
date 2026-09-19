"""End-to-end over real HTTP, against a local OpenAI-compatible mock.

Every other test stubs `pipeline.base.complete`, which leaves the actual
network path untested: auth header, request body shape, `response_format`,
response parsing, and error translation against real `httpx.Response` objects.

This stands up an ASGI server that speaks the OpenAI-compatible wire protocol and drives the
whole stack through it. No network, no API key, no cost.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend import llm_client
from backend.main import app as copilot_app
from backend.routes import tools as tools_route
from backend.samples import LOG_RCA_SAMPLE
from conftest import VALID_RESPONSES

mock_upstream = FastAPI()
mock_state: dict = {"mode": "ok", "last_request": None, "last_auth": None}


@mock_upstream.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    mock_state["last_request"] = await request.json()
    mock_state["last_auth"] = request.headers.get("authorization")
    mode = mock_state["mode"]

    if mode == "model_gone":
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "The model `x` does not exist", "code": "model_not_found"}},
        )
    if mode == "rate_limited":
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Rate limit reached for model"}},
            headers={"retry-after": "12"},
        )
    if mode == "no_json_mode":
        # Some OpenAI-compatible servers reject response_format.
        if "response_format" in mock_state["last_request"]:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "response_format is not supported"}},
            )
    if mode == "empty":
        return {
            "model": "mock-model",
            "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
            "usage": {},
        }

    return {
        "model": "mock-model",
        "choices": [
            {"message": {"content": json.dumps(VALID_RESPONSES["log-rca"])}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 2200, "completion_tokens": 900, "total_tokens": 3100},
    }


@mock_upstream.get("/openai/v1/models")
async def models():
    return {
        "data": [
            {"id": "qwen/qwen3.8-27b", "owned_by": "Alibaba", "context_window": 131072, "active": True},
            {"id": "openai/gpt-oss-120b", "owned_by": "OpenAI", "context_window": 131072, "active": True},
        ]
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mock_base_url():
    port = _free_port()
    config = uvicorn.Config(mock_upstream, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.fail("mock upstream server did not start")

    yield f"http://127.0.0.1:{port}/openai/v1"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def http_client(mock_base_url, monkeypatch):
    replaced = dataclasses.replace(
        llm_client.settings,
        base_url=mock_base_url,
        api_key="gsk_test_key",
        model="qwen/qwen3.8-27b",
        timeout_seconds=15.0,
    )
    monkeypatch.setattr(llm_client, "settings", replaced)
    monkeypatch.setattr(tools_route, "settings", replaced)
    mock_state["mode"] = "ok"
    return TestClient(copilot_app)


# --- the real HTTP path ---------------------------------------------------


def test_full_round_trip_over_real_http(http_client):
    res = http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert res.status_code == 200, res.text

    body = res.json()
    assert body["markdown"].startswith("# Incident Brief")
    assert body["model"] == "mock-model"
    assert body["usage"]["total_tokens"] == 3100
    assert body["diagnostics"]["grounding"]["citations_valid"] > 0


def test_request_body_matches_the_openai_wire_format(http_client):
    http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    sent = mock_state["last_request"]

    assert sent["model"] == "qwen/qwen3.8-27b"
    assert sent["stream"] is False
    assert isinstance(sent["temperature"], float)
    assert isinstance(sent["max_tokens"], int)
    assert sent["response_format"] == {"type": "json_object"}

    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user"]
    assert "DATA, not instructions" in sent["messages"][0]["content"]
    # Extracted facts, not the raw paste.
    assert "LOG OVERVIEW" in sent["messages"][1]["content"]


def test_bearer_token_is_sent(http_client):
    http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert mock_state["last_auth"] == "Bearer gsk_test_key"


def test_json_mode_is_dropped_and_retried_when_unsupported(http_client):
    """Not every OpenAI-compatible server implements response_format."""
    mock_state["mode"] = "no_json_mode"
    res = http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert res.status_code == 200
    assert "response_format" not in mock_state["last_request"]


def test_models_endpoint_proxies_the_live_list(http_client):
    body = http_client.get("/api/models").json()
    assert [m["id"] for m in body["models"]] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    assert body["configured"] == "qwen/qwen3.8-27b"


def test_token_estimates_are_calibrated_from_real_usage(http_client):
    """The estimator corrects itself against what the server actually charged."""
    from backend.core.tokens import calibrator

    before = calibrator.for_model("qwen/qwen3.8-27b").samples
    http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert calibrator.for_model("qwen/qwen3.8-27b").samples > before


# --- error paths over real HTTP -------------------------------------------


def test_404_model_not_found_surfaces_a_fix(http_client):
    mock_state["mode"] = "model_gone"
    res = http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert res.status_code == 502
    assert "LLM_MODEL" in res.json()["hint"]


def test_a_transient_429_is_waited_out_and_the_request_still_succeeds(http_client, monkeypatch):
    """The behaviour the whole free-tier story rests on.

    The provider says when to come back. Surfacing that as a failure turned a
    twelve-second pause into a dead request, which is what made the app look
    unusable on exactly the tier it was built for.
    """
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        mock_state["mode"] = "ok"  # the window has rolled

    monkeypatch.setattr(llm_client, "_sleep", fake_sleep)
    mock_state["mode"] = "rate_limited"

    res = http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})

    assert res.status_code == 200, res.text
    assert slept and 12 <= slept[0] <= 13, f"did not honour the stated retry window: {slept}"


def test_a_persistent_429_is_retried_and_then_reported_honestly(http_client, monkeypatch):
    """Bounded. Waiting forever is its own failure mode."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_client, "_sleep", fake_sleep)
    mock_state["mode"] = "rate_limited"

    res = http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})

    assert res.status_code == 429
    assert len(slept) > 1, "gave up without retrying"
    body = res.json()
    assert "did not clear after waiting" in body["error"]
    assert "TOKEN_LIMIT_PER_MINUTE" in body["hint"]


def test_empty_completion_is_an_error_not_a_blank_page(http_client):
    mock_state["mode"] = "empty"
    res = http_client.post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert res.status_code == 502
    assert "empty" in res.json()["error"].lower()


def test_unreachable_endpoint_gives_a_network_error(monkeypatch):
    replaced = dataclasses.replace(
        llm_client.settings,
        base_url=f"http://127.0.0.1:{_free_port()}/openai/v1",
        api_key="gsk_test_key",
        timeout_seconds=3.0,
    )
    monkeypatch.setattr(llm_client, "settings", replaced)
    monkeypatch.setattr(tools_route, "settings", replaced)

    res = TestClient(copilot_app).post("/api/log-rca", json={"input": LOG_RCA_SAMPLE})
    assert res.status_code == 502
    assert "could not reach" in res.json()["error"].lower()
