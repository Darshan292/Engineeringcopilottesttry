"""End-to-end test against a local OpenAI-compatible mock of the Groq API.

`tests/test_api.py` stubs at the `complete()` boundary, which leaves the actual
HTTP path untested: auth header, request body shape, response parsing, and the
error translation against real `httpx.Response` objects.

This file closes that gap. It stands up a real ASGI server on a local port that
speaks Groq's wire protocol, points GROQ_BASE_URL at it, and drives the whole
stack over real HTTP. No network, no API key, no cost.

Run with:  .venv/bin/pytest -q
"""

from __future__ import annotations

import dataclasses
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend import groq_client
from backend.main import app as copilot_app
from backend.routes import tools as tools_route

# --- a fake Groq ----------------------------------------------------------

mock_groq = FastAPI()
mock_state: dict = {"mode": "ok", "last_request": None, "last_auth": None}

FAKE_COMPLETION = """## Framework
Python, pytest.

## Tests

```python
def test_returns_zero_for_empty_list():
    assert total([]) == 0
```

| Case | Expected |
| --- | --- |
| empty | 0 |
"""


@mock_groq.post("/openai/v1/chat/completions")
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
    if mode == "empty":
        return {
            "model": "mock-model",
            "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
            "usage": {},
        }
    if mode == "truncated":
        return {
            "model": "mock-model",
            "choices": [{"message": {"content": "## Cut"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4096, "total_tokens": 4105},
        }

    return {
        "model": "mock-model",
        "choices": [{"message": {"content": FAKE_COMPLETION}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 64, "total_tokens": 184},
    }


@mock_groq.get("/openai/v1/models")
async def models():
    return {
        "data": [
            {"id": "qwen/qwen3.6-27b", "owned_by": "Alibaba", "context_window": 131072, "active": True},
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
    config = uvicorn.Config(mock_groq, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.fail("mock Groq server did not start")

    yield f"http://127.0.0.1:{port}/openai/v1"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def client(mock_base_url, monkeypatch):
    replaced = dataclasses.replace(
        groq_client.settings,
        groq_base_url=mock_base_url,
        groq_api_key="gsk_test_key",
        groq_model="qwen/qwen3.6-27b",
        timeout_seconds=15.0,
    )
    monkeypatch.setattr(groq_client, "settings", replaced)
    monkeypatch.setattr(tools_route, "settings", replaced)
    mock_state["mode"] = "ok"
    return TestClient(copilot_app)


# --- the real HTTP path ---------------------------------------------------


def test_full_round_trip_over_real_http(client):
    res = client.post("/api/unit-tests", json={"input": "def total(xs): return sum(xs)"})
    assert res.status_code == 200, res.text

    body = res.json()
    assert body["markdown"].startswith("## Framework")
    assert body["model"] == "mock-model"
    assert body["usage"]["total_tokens"] == 184
    assert body["elapsed_ms"] >= 0
    assert body["truncated"] is False


def test_request_body_matches_groq_wire_format(client):
    client.post("/api/postmortem", json={"input": "[02:18] pager fired"})
    sent = mock_state["last_request"]

    assert sent["model"] == "qwen/qwen3.6-27b"
    assert sent["stream"] is False
    assert isinstance(sent["temperature"], float)
    assert isinstance(sent["max_tokens"], int)

    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "user"]
    assert "blameless" in sent["messages"][0]["content"].lower()
    assert "[02:18] pager fired" in sent["messages"][1]["content"]


def test_bearer_token_is_sent(client):
    client.post("/api/api-docs", json={"input": "@app.get('/x')"})
    assert mock_state["last_auth"] == "Bearer gsk_test_key"


def test_per_request_model_override_reaches_the_wire(client):
    client.post("/api/log-rca", json={"input": "ERROR boom", "model": "openai/gpt-oss-120b"})
    assert mock_state["last_request"]["model"] == "openai/gpt-oss-120b"


def test_models_endpoint_proxies_the_live_list(client):
    body = client.get("/api/models").json()
    assert [m["id"] for m in body["models"]] == ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
    assert body["configured"] == "qwen/qwen3.6-27b"


# --- error paths over real HTTP -------------------------------------------


def test_404_model_not_found_surfaces_a_fix(client):
    mock_state["mode"] = "model_gone"
    res = client.post("/api/unit-tests", json={"input": "x"})
    assert res.status_code == 502
    assert "unavailable" in res.json()["error"].lower()
    assert "GROQ_MODEL" in res.json()["hint"]


def test_429_is_passed_through_with_the_retry_window(client):
    mock_state["mode"] = "rate_limited"
    res = client.post("/api/unit-tests", json={"input": "x"})
    assert res.status_code == 429
    assert "12s" in res.json()["hint"]


def test_empty_completion_is_an_error_not_a_blank_page(client):
    mock_state["mode"] = "empty"
    res = client.post("/api/unit-tests", json={"input": "x"})
    assert res.status_code == 502
    assert "empty" in res.json()["error"].lower()


def test_truncated_completion_is_flagged_for_the_ui(client):
    mock_state["mode"] = "truncated"
    body = client.post("/api/unit-tests", json={"input": "x"}).json()
    assert body["truncated"] is True


def test_unreachable_endpoint_gives_a_network_error(monkeypatch):
    replaced = dataclasses.replace(
        groq_client.settings,
        groq_base_url=f"http://127.0.0.1:{_free_port()}/openai/v1",
        groq_api_key="gsk_test_key",
        timeout_seconds=3.0,
    )
    monkeypatch.setattr(groq_client, "settings", replaced)
    monkeypatch.setattr(tools_route, "settings", replaced)

    res = TestClient(copilot_app).post("/api/unit-tests", json={"input": "x"})
    assert res.status_code == 502
    assert "could not reach groq" in res.json()["error"].lower()
