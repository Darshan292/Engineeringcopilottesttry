"""Run the golden evaluation set against a real model.

    python -m evals.run                    # one run per case
    python -m evals.run --repeat 3         # three runs, reports variance
    python -m evals.run --model openai/gpt-oss-120b
    python -m evals.run --case rca-cascade --verbose
    python -m evals.run --json results.json

Unlike `tests/`, this costs real API calls and needs `GROQ_API_KEY`. It is the
only thing here that measures the model's analytical quality rather than the
system's plumbing, and it is the answer to "how do you know the output is any
good?"

Why `--repeat` matters: a model is not deterministic, so a single green run
proves less than it appears to. Repeating exposes the cases that pass most of
the time, which is the difference between a tool you can rely on and one that
works when you demo it.

Critical checks are weighted separately and reported on their own line. A run
that leaks a name is a failed run whatever its aggregate score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.groq_client import GroqError  # noqa: E402
from backend.pipeline.tools import PIPELINES  # noqa: E402
from evals.cases import ALL_CASES, EvalCase  # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    critical: bool
    why: str = ""


@dataclass
class RunResult:
    case_id: str
    tool: str
    ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: int = 0
    tokens: int = 0
    attempts: int = 1
    computed_confidence: float | None = None
    markdown: str = ""

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def critical_failures(self) -> list[str]:
        return [c.name for c in self.checks if c.critical and not c.passed]


async def run_case(case: EvalCase, model: str | None) -> RunResult:
    started = time.perf_counter()
    try:
        result = await PIPELINES[case.tool](case.input, model=model, request_id=f"eval-{case.id}")
    except GroqError as exc:
        return RunResult(
            case_id=case.id, tool=case.tool, ok=False, error=f"{exc.message}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            case_id=case.id, tool=case.tool, ok=False, error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    checks: list[CheckResult] = []
    for check in case.checks:
        try:
            passed = bool(check.predicate(result.markdown, result.diagnostics))
        except Exception as exc:  # a broken predicate is a failed check, not a crash
            passed = False
            check = type(check)(check.name, check.predicate, check.critical, f"predicate raised {exc}")
        checks.append(CheckResult(check.name, passed, check.critical, check.why))

    return RunResult(
        case_id=case.id,
        tool=case.tool,
        ok=all(c.passed for c in checks),
        checks=checks,
        elapsed_ms=result.elapsed_ms,
        tokens=result.usage.get("total_tokens", 0),
        attempts=result.attempts,
        computed_confidence=(result.diagnostics.get("confidence") or {}).get("computed_score"),
        markdown=result.markdown,
    )


def print_run(result: RunResult, verbose: bool) -> None:
    if result.error:
        print(f"  {RED}ERROR{RESET} {result.error[:160]}")
        return

    for check in result.checks:
        if check.passed:
            if verbose:
                print(f"    {GREEN}pass{RESET} {check.name}")
        else:
            marker = f"{RED}FAIL (critical){RESET}" if check.critical else f"{YELLOW}fail{RESET}"
            print(f"    {marker} {check.name}")
            if check.why:
                print(f"         {DIM}{check.why}{RESET}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run golden evals against a real model.")
    parser.add_argument("--model", help="Override GROQ_MODEL for this run.")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per case; exposes non-determinism.")
    parser.add_argument("--case", action="append", help="Run only these case IDs.")
    parser.add_argument("--tool", action="append", help="Run only these tools.")
    parser.add_argument("--verbose", action="store_true", help="Show passing checks too.")
    parser.add_argument("--json", help="Write full results to this path.")
    parser.add_argument("--save-outputs", help="Directory to write each generated document into.")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print(f"{RED}GROQ_API_KEY is not set.{RESET} These evals make real API calls.")
        print("Get a free key at https://console.groq.com/keys, then: export GROQ_API_KEY=...")
        return 2

    cases = ALL_CASES
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    if args.tool:
        cases = [c for c in cases if c.tool in set(args.tool)]
    if not cases:
        print("No cases matched the filters.")
        return 2

    model = args.model or os.getenv("GROQ_MODEL") or "(configured default)"
    total_calls = len(cases) * args.repeat
    print(f"{BOLD}Golden evaluation{RESET}")
    print(f"  model:  {model}")
    print(f"  cases:  {len(cases)} x {args.repeat} = {total_calls} upstream calls")
    print(f"  {DIM}The free tier allows roughly 30 requests/minute.{RESET}\n")

    by_case: dict[str, list[RunResult]] = defaultdict(list)
    all_results: list[RunResult] = []

    for case in cases:
        print(f"{BOLD}{case.id}{RESET} ({case.tool}) - {case.description}")
        for attempt in range(args.repeat):
            result = await run_case(case, args.model)
            by_case[case.id].append(result)
            all_results.append(result)

            label = f"  run {attempt + 1}/{args.repeat}:"
            if result.error:
                print(f"{label} {RED}error{RESET}")
            else:
                colour = GREEN if result.ok else (RED if result.critical_failures else YELLOW)
                print(
                    f"{label} {colour}{result.score:.0%}{RESET} "
                    f"({sum(1 for c in result.checks if c.passed)}/{len(result.checks)} checks) "
                    f"{DIM}{result.elapsed_ms}ms, {result.tokens} tokens, "
                    f"{result.attempts} attempt(s), confidence "
                    f"{result.computed_confidence if result.computed_confidence is not None else 'n/a'}{RESET}"
                )
            print_run(result, args.verbose)

            if args.save_outputs and result.markdown:
                out = Path(args.save_outputs)
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{case.id}-run{attempt + 1}.md").write_text(result.markdown, encoding="utf-8")

            # Stay inside the free tier's per-minute allowance.
            if attempt + 1 < args.repeat or case is not cases[-1]:
                await asyncio.sleep(2.5)
        print()

    # --- summary ---
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}Summary{RESET}\n")

    header = f"{'case':28} {'mean':>7} {'min':>7} {'max':>7} {'stdev':>7}  critical failures"
    print(header)
    print("-" * len(header))

    all_critical: list[str] = []
    for case_id, runs in by_case.items():
        scores = [r.score for r in runs]
        criticals = sorted({name for r in runs for name in r.critical_failures})
        all_critical.extend(criticals)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        marker = RED if criticals else (GREEN if min(scores) == 1.0 else YELLOW)
        print(
            f"{case_id:28} {marker}{statistics.mean(scores):>6.0%}{RESET} "
            f"{min(scores):>6.0%} {max(scores):>6.0%} {stdev:>6.0%}  "
            f"{RED + ', '.join(criticals) + RESET if criticals else DIM + 'none' + RESET}"
        )

    mean_score = statistics.mean([r.score for r in all_results]) if all_results else 0.0
    errored = sum(1 for r in all_results if r.error)

    print()
    print(f"  overall mean score : {mean_score:.1%}")
    print(f"  runs               : {len(all_results)} ({errored} errored)")
    print(f"  total tokens       : {sum(r.tokens for r in all_results):,}")

    if args.repeat > 1:
        unstable = [
            case_id for case_id, runs in by_case.items()
            if len({r.ok for r in runs}) > 1
        ]
        if unstable:
            print(f"  {YELLOW}non-deterministic   : {', '.join(unstable)}{RESET}")
            print(f"  {DIM}These pass sometimes. A single green run would have hidden that.{RESET}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {
                        "case_id": r.case_id,
                        "tool": r.tool,
                        "ok": r.ok,
                        "score": r.score,
                        "error": r.error,
                        "elapsed_ms": r.elapsed_ms,
                        "tokens": r.tokens,
                        "attempts": r.attempts,
                        "computed_confidence": r.computed_confidence,
                        "checks": [
                            {"name": c.name, "passed": c.passed, "critical": c.critical} for c in r.checks
                        ],
                    }
                    for r in all_results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {args.json}")

    print()
    if all_critical:
        print(f"{RED}{BOLD}FAILED: critical checks did not pass.{RESET}")
        print(f"  {', '.join(sorted(set(all_critical)))}")
        return 1
    if mean_score < 0.8:
        print(f"{YELLOW}{BOLD}WEAK: mean score below 80%.{RESET} No critical failures, but quality is poor.")
        return 1
    print(f"{GREEN}{BOLD}PASSED{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
