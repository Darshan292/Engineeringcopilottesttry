"""Map-reduce execution and evidence scoping.

These cover the two defects that made the previous version's large-input story
untrue rather than merely incomplete:

1. The planner produced N chunks and the executor sent only the first, so
   everything after chunk 1 was silently discarded while the response still
   presented itself as an analysis of the whole input.
2. The model was told every ID in the input was valid to cite, regardless of
   which ones its prompt actually contained. A citation of a real-but-unseen
   line therefore passed the grounding check — a false negative in the exact
   mechanism that exists to catch invention.
"""

from __future__ import annotations

import json

import pytest

from backend.core.chunking import plan_context
from backend.core.tokens import build_budget
from backend.parsers.logs import parse_logs
from conftest import VALID_RESPONSES, response_for


def _log(lines: int) -> str:
    return "\n".join(
        f"2026-09-14T02:{i // 60 % 60:02d}:{i % 60:02d}Z ERROR svc-a [db] "
        f"failure variant-{i % 7} id={i}"
        for i in range(lines)
    )


@pytest.fixture
def tiny_window(monkeypatch):
    """Force map-reduce by shrinking the context window.

    Big enough that the system prompt and output reserve still leave room for
    content -- otherwise the planner correctly refuses before chunking, which
    is a different code path.
    """
    from backend.core import tokens

    # 16,000 used to force chunking; right-sizing the per-tool output reserve
    # freed enough budget that a compressed single call now fits, which is the
    # better outcome but a different code path. 10,000 still forces the split.
    monkeypatch.setitem(tokens.CONTEXT_WINDOWS, "qwen/qwen3.6-27b", 10_000)
    monkeypatch.setattr(tokens, "registry", tokens.ModelRegistry())
    return None


# --- the planner and the executor must agree ------------------------------


def test_planner_produces_multiple_chunks_for_a_large_input(tiny_window):
    from backend.prompts import TOOL_PROMPTS

    ir = parse_logs(_log(600))
    # The real system prompt, so the budget matches what the pipeline computes.
    plan = plan_context(ir, build_budget("qwen/qwen3.6-27b", TOOL_PROMPTS["log-rca"], 4096))
    assert plan.strategy == "map_reduce"
    assert len(plan.chunks) > 1
    # Every chunk must be individually sendable.
    for chunk in plan.chunks:
        assert chunk.item_ids
        assert chunk.total == len(plan.chunks)


def test_every_chunk_is_actually_sent(client, for_tool, tiny_window):
    """The bug: N chunks planned, one sent, N-1 silently dropped."""
    stub = for_tool("log-rca")
    res = client.post("/api/log-rca", json={"input": _log(600)})
    assert res.status_code == 200, res.text

    body = res.json()
    stages = [s["name"] for s in body["trace"]["stages"]]
    map_stages = {s for s in stages if s.startswith("map.part")}
    assert len(map_stages) > 1, f"only {len(map_stages)} map stage(s) ran: {stages}"
    assert any(s.startswith("reduce") for s in stages), "no combine step ran"

    # One call per part, plus the combine. The combine is a tree, so with many
    # parts there is more than one combine call.
    reduce_calls = len([s for s in stages if s.startswith("reduce.") and "level" not in s])
    assert stub.call_count == len(map_stages) + reduce_calls
    assert reduce_calls >= 1


def test_no_input_is_silently_discarded(client, for_tool, tiny_window):
    """Every parsed line must appear in some call's payload."""
    stub = for_tool("log-rca")
    source = _log(400)
    client.post("/api/log-rca", json={"input": source})

    sent = "\n".join(call["user_content"] for call in stub.calls)
    ir = parse_logs(source)
    missing = [entry.id for entry in ir.entries if f"[{entry.id}]" not in sent]
    assert not missing, f"{len(missing)} lines never reached the model, e.g. {missing[:5]}"


def test_the_user_is_told_the_input_was_split(client, for_tool, tiny_window):
    for_tool("log-rca")
    body = client.post("/api/log-rca", json={"input": _log(600)}).json()
    assert any("analysed in" in w and "parts" in w for w in body["warnings"])
    assert any("No single call saw the whole input" in w for w in body["warnings"])


def test_cost_is_aggregated_across_all_calls(client, for_tool, tiny_window):
    for_tool("log-rca")
    body = client.post("/api/log-rca", json={"input": _log(600)}).json()
    stages = [s for s in body["trace"]["stages"] if s["name"].startswith(("map.", "reduce"))]
    # Usage must reflect every call, not just the last one.
    assert body["usage"]["total_tokens"] >= 150 * len(stages) / 2


# --- evidence scoping ------------------------------------------------------


def test_each_call_may_only_cite_what_that_call_contains(client, for_tool, tiny_window):
    stub = for_tool("log-rca")
    source = _log(400)
    client.post("/api/log-rca", json={"input": source})

    ir = parse_logs(source)
    all_ids = ir.evidence_ids()

    for call in stub.calls:
        content = call["user_content"]
        if "VALID EVIDENCE IDs" not in content:
            continue  # the combine turn has its own wording
        declared = content.split("VALID EVIDENCE IDs (cite only from these):")[1].split("\n")[0]
        for candidate in declared.replace("...", " ").split(","):
            candidate = candidate.strip().split(" ")[0].strip()
            if candidate and candidate in all_ids:
                assert f"[{candidate}]" in content, (
                    f"{candidate} was declared citable but is not present in this call's payload"
                )


def test_a_single_pass_call_declares_only_the_presented_ids(client, for_tool):
    """Compressed mode shows representative lines, not every line."""
    stub = for_tool("log-rca")
    # Large enough to compress, small enough to stay single-pass.
    client.post("/api/log-rca", json={"input": _log(20_000)})

    content = stub.last_user_content
    declared = content.split("VALID EVIDENCE IDs (cite only from these):")[1].split("\n")[0]
    # The whole input has 20k IDs; a compressed view must not claim all of them.
    assert "20000" not in declared
    assert "L19999" not in declared


def test_the_combine_step_cannot_introduce_new_evidence(client, for_tool, tiny_window):
    """The reduce call never saw raw content, so it may only reuse cited IDs."""
    stub = for_tool("log-rca")
    # Parts cite L1 and L2 only.
    partial = response_for(
        "log-rca",
        timeline=[{"evidence_id": "L1", "time": "t", "event": "e"}],
        root_cause={"statement": "cause", "evidence_ids": ["L2"]},
        supporting_evidence=[],
        contradicting_evidence=[],
        alternative_hypotheses=[],
    )
    stub.script(partial)

    client.post("/api/log-rca", json={"input": _log(600)})
    reduce_call = stub.calls[-1]["user_content"]

    assert "EVIDENCE IDs CITED BY THE PARTS" in reduce_call
    declared = reduce_call.split("EVIDENCE IDs CITED BY THE PARTS (you may cite only these):")[1]
    declared = declared.split("\n")[0]
    assert "L1" in declared and "L2" in declared
    assert "L300" not in declared, "the combine step was offered evidence no part cited"


def test_reduce_receives_the_global_overview_and_every_partial(client, for_tool, tiny_window):
    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": _log(600)})
    reduce_call = stub.calls[-1]["user_content"]

    assert "LOG OVERVIEW" in reduce_call
    assert "PARTIAL ANALYSIS 1" in reduce_call
    assert "PARTIAL ANALYSIS 2" in reduce_call
    # Partials are JSON, so the combine step reasons over structure.
    assert '"root_cause"' in reduce_call


def test_map_parts_know_they_are_partial(client, for_tool, tiny_window):
    stub = for_tool("log-rca")
    client.post("/api/log-rca", json={"input": _log(600)})
    first = stub.calls[0]["user_content"]
    assert "part 1 of" in first.lower()
    assert "describes the ENTIRE input" in first


def test_domain_validators_run_only_on_the_final_answer(client, for_tool, tiny_window, monkeypatch):
    """Executing generated tests against a partial view would be meaningless."""
    calls = {"n": 0}

    def counting_exec(*args, **kwargs):
        calls["n"] += 1
        from backend.validation.python_exec import ExecutionReport

        report = ExecutionReport(ran=True, syntax_ok=True, collected=1, passed=1)
        return report, []

    monkeypatch.setattr("backend.pipeline.tools.run_python_tests", counting_exec)

    stub = for_tool("unit-tests")
    big_code = "\n\n".join(
        f"def function_{i}(x):\n    if x > {i}:\n        raise ValueError('too big')\n    return x * {i}"
        for i in range(120)
    )
    res = client.post("/api/unit-tests", json={"input": big_code})
    if res.status_code != 200:
        pytest.skip(f"input did not reach map-reduce for this tool: {res.status_code}")

    stages = [s["name"] for s in res.json()["trace"]["stages"]]
    if not any(s.startswith("map.part") for s in stages):
        pytest.skip("input stayed single-pass")
    assert calls["n"] == 1, "test execution ran on partial views instead of the final answer"
