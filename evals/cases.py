"""Golden evaluation cases.

Each case pairs an input with checks that a correct answer must satisfy. The
checks are deterministic predicates over the rendered document and the
pipeline's diagnostics -- no LLM-as-judge, because grading a model's output
with another model inherits the same failure modes and cannot be trusted on
exactly the cases that matter.

Three kinds of check:

- **must_contain / must_not_contain**: substring facts. Blunt, but a postmortem
  containing a real name is a hard failure whatever else it got right.
- **predicate**: a function over (markdown, diagnostics). This is where the
  interesting assertions live -- did the RCA identify the trigger rather than
  the loudest symptom, did it cite real evidence, did the tests execute.
- **weight**: how much the case counts toward the score. Safety properties are
  weighted higher than analytical quality, because a leak is worse than a
  mediocre summary.

Adding a case is the cheapest way to stop a regression from shipping twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from backend.samples import (
    API_DOC_SAMPLE,
    LOG_RCA_SAMPLE,
    POSTMORTEM_SAMPLE,
    UNIT_TEST_SAMPLE,
)


@dataclass
class Check:
    name: str
    predicate: Callable[[str, dict], bool]
    critical: bool = False
    why: str = ""


@dataclass
class EvalCase:
    id: str
    tool: str
    input: str
    description: str
    checks: list[Check] = field(default_factory=list)


def _contains(*needles: str) -> Callable[[str, dict], bool]:
    return lambda markdown, _d: all(n.lower() in markdown.lower() for n in needles)


def _absent(*needles: str) -> Callable[[str, dict], bool]:
    return lambda markdown, _d: not any(n.lower() in markdown.lower() for n in needles)


# --- log / RCA ------------------------------------------------------------


def _identifies_reindex_trigger(markdown: str, _d: dict) -> bool:
    """The reindex job is the cause; the pool exhaustion is the symptom."""
    lowered = markdown.lower()
    root_section = lowered.split("## root cause analysis")[-1].split("## next steps")[0]
    return "reindex" in root_section or "full scan" in root_section or "stock_levels" in root_section


def _cites_only_real_evidence(_m: str, diagnostics: dict) -> bool:
    return not diagnostics.get("grounding", {}).get("citations_fabricated")


def _timeline_is_evidence_backed(_m: str, diagnostics: dict) -> bool:
    grounding = diagnostics.get("grounding", {})
    return grounding.get("citations_total", 0) >= 3 and grounding.get("grounding_ratio", 0) >= 0.9


def _offers_alternatives(markdown: str, _d: dict) -> bool:
    section = markdown.lower().split("alternative hypotheses")[-1]
    return "none offered" not in section[:200]


def _not_fooled_by_the_loudest_error(markdown: str, _d: dict) -> bool:
    """Naming the SQLTransientConnectionException as the root cause is the trap."""
    lowered = markdown.lower()
    root_section = lowered.split("**most likely cause.**")[-1][:600]
    names_symptom = "sqltransientconnectionexception" in root_section
    names_cause = any(w in root_section for w in ("reindex", "full scan", "stock_levels", "primary"))
    return names_cause or not names_symptom


LOG_CASES = [
    EvalCase(
        id="rca-cascade",
        tool="log-rca",
        input=LOG_RCA_SAMPLE,
        description="Multi-service cascade where the trigger precedes the loudest error by four minutes.",
        checks=[
            Check("no fabricated citations", _cites_only_real_evidence, critical=True,
                  why="An invented line ID means the analysis confabulated."),
            Check("timeline is evidence-backed", _timeline_is_evidence_backed,
                  why="Rows must trace to real lines."),
            Check("identifies the reindex as the cause", _identifies_reindex_trigger,
                  why="The reindex job at 02:14 is the actual trigger."),
            Check("not fooled by the loudest error", _not_fooled_by_the_loudest_error,
                  why="The connection exception is a symptom, not the cause."),
            Check("offers an alternative hypothesis", _offers_alternatives,
                  why="A single unchallenged explanation is a weak analysis."),
            Check("names affected services", _contains("checkout-api"),
                  why="Blast radius must come from the log."),
            Check("invents no owner names", _absent("priya", "marcus", "aisha", "dan "),
                  why="Owners are role placeholders."),
        ],
    ),
    EvalCase(
        id="rca-insufficient-evidence",
        tool="log-rca",
        input=(
            "2026-09-14T02:16:45.331Z ERROR checkout-api [db] Connection is not available\n"
            "2026-09-14T02:16:51.874Z ERROR checkout-api [db] Connection is not available\n"
        ),
        description="Two identical lines with no context. Honest answer is 'not enough evidence'.",
        checks=[
            Check("no fabricated citations", _cites_only_real_evidence, critical=True),
            Check(
                "admits the evidence is thin",
                lambda m, d: (
                    d.get("confidence", {}).get("computed_band") in {"low", "medium"}
                    or any(w in m.lower() for w in ("insufficient", "not determinable", "cannot", "unknown", "gap"))
                ),
                why="Two lines cannot support a confident root cause.",
            ),
            Check("lists what is missing", _contains("evidence gaps"),
                  why="Should say what data would settle it."),
        ],
    ),
]


# --- postmortem -----------------------------------------------------------

REAL_NAMES = ("priya", "raghavan", "marcus", "webb", "aisha", "bello", "oyelaran")


def _no_real_names(markdown: str, _d: dict) -> bool:
    return not any(name in markdown.lower() for name in REAL_NAMES)


def _blames_no_individual(markdown: str, _d: dict) -> bool:
    lowered = markdown.lower()
    blame_phrases = ("someone changed", "he ", "she ", "his ", "her ", "failed to notice", "their mistake")
    return not any(phrase in lowered for phrase in blame_phrases)


def _distinguishes_trigger_from_root_cause(markdown: str, _d: dict) -> bool:
    lowered = markdown.lower()
    if "**trigger.**" not in lowered or "**root cause.**" not in lowered:
        return False
    trigger = lowered.split("**trigger.**")[1].split("**root cause.**")[0]
    root = lowered.split("**root cause.**")[1][:800]
    return trigger.strip() != root.strip() and len(root.strip()) > 40


def _flags_mitigation_not_fix(markdown: str, diagnostics: dict) -> bool:
    return "mitigation, not a fix" in markdown.lower()


def _has_a_detect_action(markdown: str, _d: dict) -> bool:
    section = markdown.lower().split("## action items")[-1]
    return "detect" in section


POSTMORTEM_CASES = [
    EvalCase(
        id="postmortem-blameless",
        tool="postmortem",
        input=POSTMORTEM_SAMPLE,
        description="Transcript full of real names, a wrong hypothesis, and a mitigation that is not a fix.",
        checks=[
            Check("no real names anywhere", _no_real_names, critical=True,
                  why="The core promise of the tool."),
            Check("no individual is blamed", _blames_no_individual, critical=True,
                  why="Blameless is the point."),
            Check("no fabricated citations", _cites_only_real_evidence, critical=True),
            Check("trigger differs from root cause", _distinguishes_trigger_from_root_cause,
                  why="The config change is the root cause; the workload growth triggered it."),
            Check("flags that this was a mitigation", _flags_mitigation_not_fix,
                  why="The reindex job is still misconfigured."),
            Check("includes a Detect action item", _has_a_detect_action,
                  why="The detection gap was the stated main miss."),
            Check("identifies the detection gap", _contains("detect"),
                  why="Nobody was alerted before customers were."),
            Check("uses role placeholders", _contains("["),
                  why="Owners are roles."),
        ],
    ),
]


# --- unit tests -----------------------------------------------------------


def _tests_actually_pass(_m: str, diagnostics: dict) -> bool:
    execution = diagnostics.get("execution") or {}
    return bool(execution.get("executed")) and execution.get("failed", 1) == 0 and execution.get("collected", 0) > 0


def _covers_most_boundaries(_m: str, diagnostics: dict) -> bool:
    return (diagnostics.get("coverage") or {}).get("ratio", 0) >= 0.6


def _covers_the_promo_boundary(markdown: str, _d: dict) -> bool:
    return "10000" in markdown or "10,000" in markdown or "100.00" in markdown


UNIT_TEST_CASES = [
    EvalCase(
        id="tests-pricing",
        tool="unit-tests",
        input=UNIT_TEST_SAMPLE,
        description="Pricing function with stacking rules, a cap, and boundary bugs.",
        checks=[
            Check("generated tests execute and pass", _tests_actually_pass, critical=True,
                  why="'Runnable' must be a result, not a claim."),
            Check("covers 60%+ of extracted boundaries", _covers_most_boundaries,
                  why="The AST enumerated them; the suite should cover them."),
            Check("tests the $100 promo boundary", _covers_the_promo_boundary,
                  why="subtotal > 10000 is the subtlest boundary in the function."),
            Check("tests the empty-order error", _contains("valueerror", "empty"),
                  why="An explicit raise in the source."),
            Check("tests the unknown-tier error", _contains("keyerror"),
                  why="The other explicit raise."),
            Check("names pytest", _contains("pytest")),
        ],
    ),
]


# --- API docs -------------------------------------------------------------


def _openapi_is_valid(_m: str, diagnostics: dict) -> bool:
    report = diagnostics.get("openapi") or {}
    return bool(report.get("parsed")) and bool(report.get("schema_valid"))


def _documents_every_route(_m: str, diagnostics: dict) -> bool:
    return not (diagnostics.get("openapi") or {}).get("routes_missing_from_spec")


def _invents_no_routes(_m: str, diagnostics: dict) -> bool:
    return not (diagnostics.get("openapi") or {}).get("routes_in_spec_but_not_in_code")


API_CASES = [
    EvalCase(
        id="docs-deployments",
        tool="api-docs",
        input=API_DOC_SAMPLE,
        description="Three routes with auth, conditional validation, and 409/422/404 error paths.",
        checks=[
            Check("OpenAPI is schema-valid", _openapi_is_valid, critical=True,
                  why="Valid-looking YAML is not a valid spec."),
            Check("documents every real route", _documents_every_route, critical=True,
                  why="A missing endpoint makes the doc unusable."),
            Check("invents no routes", _invents_no_routes, critical=True,
                  why="A documented endpoint that does not exist is worse than none."),
            Check("documents the 409 conflict", _contains("409"),
                  why="Raised inside the handler; hand-written docs always miss it."),
            Check("documents the 422 canary rule", _contains("422"),
                  why="Production deploys require canary_percent."),
            Check("documents the 404", _contains("404")),
            Check("mentions auth", _contains("bearer")),
        ],
    ),
]


ALL_CASES: list[EvalCase] = LOG_CASES + POSTMORTEM_CASES + UNIT_TEST_CASES + API_CASES
