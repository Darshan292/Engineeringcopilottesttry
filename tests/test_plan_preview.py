"""The plan is shown before it is executed, and it is the real plan.

Two things make a preview worth having. It must cost nothing -- otherwise the
user pays twice to see what they are about to pay for. And it must be produced
by the same code that does the work, because a preview that can disagree with
the run is a lie with a progress bar attached.

Both are asserted here rather than assumed.
"""

from __future__ import annotations

import pytest

from backend.samples import API_DOC_SAMPLE, LOG_RCA_SAMPLE, POSTMORTEM_SAMPLE, UNIT_TEST_SAMPLE

TOOLS = [
    ("log-rca", LOG_RCA_SAMPLE),
    ("unit-tests", UNIT_TEST_SAMPLE),
    ("api-docs", API_DOC_SAMPLE),
    ("postmortem", POSTMORTEM_SAMPLE),
]


def _varied_log(lines: int) -> str:
    shapes = [
        "ERROR svc-a [db] timeout after {i}ms id={i}",
        "WARN svc-b [cache] miss key=user:{i} region=eu-{i}",
        "INFO svc-c [api] GET /orders/{i} 200 in {i}ms",
        "ERROR svc-d [queue] nack job={i} reason=deadline-{i}",
    ]
    return "\n".join(
        f"2026-09-14T02:{i // 60 % 60:02d}:{i % 60:02d}Z " + shapes[i % 4].format(i=i)
        for i in range(lines)
    )


@pytest.mark.parametrize("tool, sample", TOOLS)
def test_planning_costs_no_model_call(client, for_tool, tool, sample):
    stub = for_tool(tool)
    res = client.post(f"/api/plan/{tool}", json={"input": sample})
    assert res.status_code == 200, res.text
    assert stub.call_count == 0, "the preview called the model"


@pytest.mark.parametrize("tool, sample", TOOLS)
def test_the_preview_reports_what_was_extracted_before_the_model(client, for_tool, tool, sample):
    """The point is to show that real work happens without the model."""
    for_tool(tool)
    body = client.post(f"/api/plan/{tool}", json={"input": sample}).json()

    assert body["feasible"] is True
    assert body["extracted"], "nothing was reported as extracted"
    # Deterministic stages are listed alongside the model call, not hidden
    # behind it -- otherwise the preview reproduces the impression that the
    # answer is whatever the model said.
    kinds = {step["kind"] for step in body["steps"]}
    assert "deterministic" in kinds
    assert "model" in kinds
    deterministic = [s for s in body["steps"] if s["kind"] == "deterministic"]
    assert len(deterministic) > len([s for s in body["steps"] if s["kind"] == "model"])

    names = " ".join(s["name"] for s in body["steps"])
    for expected in ("Redact", "injection", "Verify every citation", "Compute confidence"):
        assert expected in names


@pytest.mark.parametrize("tool, sample", TOOLS)
def test_the_preview_matches_what_the_run_actually_does(client, for_tool, tool, sample):
    """The contract that makes the preview worth showing at all."""
    stub = for_tool(tool)
    preview = client.post(f"/api/plan/{tool}", json={"input": sample}).json()

    run = client.post(f"/api/{tool}", json={"input": sample})
    assert run.status_code == 200, run.text
    stages = {s["name"]: s for s in run.json()["trace"]["stages"]}

    assert stages["plan_context"]["strategy"] == preview["context"]["strategy"]
    assert stages["plan_context"]["chunks"] == preview["context"]["chunks"]
    assert stages["budget"]["available_for_input"] == preview["budget"]["available_for_input"]

    # The number of model calls promised is the number made.
    promised = len([s for s in preview["steps"] if s["kind"] == "model"])
    assert stub.call_count == promised, (
        f"preview promised {promised} model calls, the run made {stub.call_count}"
    )


def test_a_split_input_is_described_part_by_part(client, for_tool, monkeypatch):
    """'How is it splitting' is answerable before the first call, not after."""
    from backend.core import tokens

    monkeypatch.setitem(tokens.CONTEXT_WINDOWS, "qwen/qwen3.6-27b", 12_000)
    monkeypatch.setattr(tokens, "registry", tokens.ModelRegistry())
    for_tool("log-rca")

    body = client.post("/api/plan/log-rca", json={"input": _varied_log(600)}).json()
    assert body["context"]["strategy"] == "map_reduce"

    parts = [s for s in body["steps"] if s["name"].startswith("Analyse part")]
    assert len(parts) == body["context"]["chunks"] > 1
    # Each part says what it contains, so "how is it splitting" has an answer.
    for step in parts:
        assert "items" in step["detail"] and "tokens" in step["detail"]
    assert any(s["name"] == "Combine the partial findings" for s in body["steps"])

    # And the narrative addressed to the model says the same thing.
    assert "Split into" in body["narrative"]
    assert "Nothing was dropped" in body["narrative"]


def test_a_refused_plan_is_previewed_rather_than_raised(client, for_tool, monkeypatch):
    """'It would be refused, here is why' is an answer, not an error."""
    from backend.core import chunking, tokens

    monkeypatch.setitem(tokens.CONTEXT_WINDOWS, "qwen/qwen3.6-27b", 8_000)
    monkeypatch.setattr(tokens, "registry", tokens.ModelRegistry())
    monkeypatch.setattr(chunking, "DEFAULT_MAX_CHUNKS", 2)
    for_tool("unit-tests")

    code = "\n\n".join(
        f"def function_{i}(x, y):\n    if x > {i}:\n        raise ValueError('over')\n    return x * y"
        for i in range(400)
    )
    res = client.post("/api/plan/unit-tests", json={"input": code})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["feasible"] is False
    assert any(s["kind"] == "blocked" for s in body["steps"])
    assert "model calls" in body["context"]["reason"]


def test_the_preview_is_stateless_like_everything_else(client, for_tool):
    """A preview must not become context for the next request either."""
    stub = for_tool("log-rca")
    client.post("/api/plan/log-rca", json={"input": _varied_log(50)})
    client.post("/api/log-rca", json={"input": "2026-01-01T00:00:00Z ERROR zeta [x] boom"})
    assert "svc-a" not in stub.last_user_content
    assert "orders" not in stub.last_user_content


def test_an_unknown_tool_is_a_404(client, for_tool):
    for_tool("log-rca")
    assert client.post("/api/plan/nonsense", json={"input": "x"}).status_code == 404


def test_the_promised_call_count_holds_for_a_split_input(client, for_tool, monkeypatch):
    """The case that exposed a preview promising half the calls it made.

    On a small per-call budget the combine is a tree: thirteen partial answers
    do not fit one combine call, so they are merged in batches and the merges
    merged again. The planner counted one combine call, promised fourteen, and
    the run made twenty-seven. A preview that can be wrong by 2x about the thing
    it exists to report is worse than no preview, so the arithmetic is asserted
    against the run rather than reasoned about.
    """
    from backend.core import tokens

    monkeypatch.setitem(tokens.CONTEXT_WINDOWS, "qwen/qwen3.6-27b", 8_000)
    monkeypatch.setattr(tokens, "registry", tokens.ModelRegistry())
    stub = for_tool("log-rca")

    log = _varied_log(600)
    preview = client.post("/api/plan/log-rca", json={"input": log}).json()
    assert preview["context"]["strategy"] == "map_reduce"
    assert preview["context"]["reduce_levels"] > 1, "this input must exercise a tree combine"

    promised = len([s for s in preview["steps"] if s["kind"] == "model"])
    assert promised == preview["context"]["estimated_model_calls"]
    assert promised > preview["context"]["chunks"] + 1, "a tree combine costs more than one call"

    run = client.post("/api/log-rca", json={"input": log})
    assert run.status_code == 200, run.text

    # The promise is an upper bound, and the bound is the part that matters: a
    # terse model packs more partials into each combine and finishes in fewer
    # calls, which is a pleasant surprise. Being over is the failure -- it means
    # quota was spent that nobody agreed to.
    assert stub.call_count <= promised, (
        f"preview promised at most {promised} model calls, the run made {stub.call_count}"
    )
    # The map phase is exact: one call per part, no estimation involved.
    map_calls = len([s for s in preview["steps"] if s["name"].startswith("Analyse part")])
    assert map_calls == preview["context"]["chunks"]


def test_a_budget_too_small_to_combine_does_not_spend_the_map_calls_first(monkeypatch):
    """Splitting is pointless if the results can never be merged.

    A combine call must hold at least two partial answers or the tree never
    shrinks. Below that the old code split happily, spent one call per part,
    and failed at the last step with "did not converge" -- every one of those
    calls charged to a free-tier quota for nothing.
    """
    from backend.core import chunking
    from backend.core.chunking import plan_context
    from backend.core.tokens import build_budget
    from backend.parsers.code import parse_code

    # Code, so there is no relevance fallback to soften the outcome.
    code = "\n\n".join(
        f"def function_{i}(x):\n    if x > {i}:\n        raise ValueError('over')\n    return x"
        for i in range(400)
    )
    ir = parse_code(code)
    budget = build_budget("tiny-unknown-model", "sys", 1024)

    # Force partials so large that two cannot share a combine call.
    monkeypatch.setattr(chunking, "MIN_PARTIAL_TOKENS", 10_000)
    plan = plan_context(ir, budget)

    assert plan.strategy == "reject"
    assert "combine" in plan.reason
    assert "never be combined" in plan.reason or "could never converge" in plan.reason


def test_the_combine_budget_always_leaves_room_for_two_partials():
    """The property that makes the tree converge, asserted directly."""
    from backend.core.chunking import _map_output_budget, _reduce_tree_cost

    for ceiling in (1_000, 1_709, 3_709, 12_000, 90_000):
        partial = _map_output_budget(ceiling, 1_200)
        assert partial * 2 <= ceiling, f"a combine at {ceiling} cannot hold two partials"
        # And the tree therefore shrinks at every level rather than stalling.
        # Two per batch folds at most 2**4 = 16 parts inside the level cap, so
        # convergence is reported rather than promised -- and a plan that would
        # not converge is refused before any map call is paid for.
        per_batch = ceiling // max(1, partial)
        calls, levels, converges = _reduce_tree_cost(16, ceiling, max(1, partial))
        assert converges, f"16 parts should fold at ceiling {ceiling} (batch of {per_batch})"
        assert calls < 16


def test_the_projected_duration_counts_every_combine_round(client, for_tool, monkeypatch):
    """Understating the calls understated the wait that decides refusals."""
    from backend.core import tokens
    from backend.core.chunking import plan_context
    from backend.core.tokens import build_budget
    from backend.parsers.logs import parse_logs

    monkeypatch.setitem(tokens.CONTEXT_WINDOWS, "qwen/qwen3.6-27b", 8_000)
    monkeypatch.setattr(tokens, "registry", tokens.ModelRegistry())

    ir = parse_logs(_varied_log(600))
    plan = plan_context(
        ir, build_budget("qwen/qwen3.6-27b", "sys", 2_200, token_allowance_per_minute=8_000)
    )
    if plan.strategy != "map_reduce":
        pytest.skip("this budget no longer produces a split plan")

    # The projection must price the whole tree, not the map phase plus one.
    assert plan.estimated_calls > len(plan.chunks) + 1
    assert plan.estimated_seconds > 0
    naive_seconds = plan.estimated_seconds * (len(plan.chunks) + 1) / plan.estimated_calls
    assert plan.estimated_seconds > naive_seconds
