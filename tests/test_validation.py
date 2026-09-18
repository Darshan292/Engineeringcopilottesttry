"""Validation-layer tests: grounding, confidence, OpenAPI, test execution.

This is the half of the architecture that exists because a model's output is a
proposal, not a result.
"""

from __future__ import annotations

import pytest

from backend.parsers.logs import parse_logs
from backend.samples import LOG_RCA_SAMPLE, UNIT_TEST_SAMPLE
from backend.validation.confidence import compute_confidence
from backend.validation.grounding import GroundingReport, check_citations, claims_from_rca, drop_fabricated
from backend.validation.openapi import validate_openapi
from backend.validation.python_exec import run_python_tests
from backend.validation.schemas import RCAOutput

VALID_SPEC = """
openapi: 3.1.0
info:
  title: Test API
  version: '1.0.0'
servers:
  - url: https://api.example.com
paths:
  /v1/deployments:
    get:
      summary: List
      responses:
        '200':
          description: OK
"""


@pytest.fixture
def log_ir():
    return parse_logs(LOG_RCA_SAMPLE)


# --- grounding ------------------------------------------------------------


def test_real_citations_are_accepted(log_ir):
    report = check_citations([("pool exhausted", ["L6", "L7"])], log_ir)
    assert report.grounding_ratio == 1.0
    assert not report.has_fabrications


def test_fabricated_citations_are_caught(log_ir):
    report = check_citations([("something happened", ["L6", "L9999"])], log_ir)
    assert report.invalid_citations == ["L9999"]
    assert report.has_fabrications
    assert report.grounding_ratio == 0.5


def test_template_ids_are_valid_citations_too(log_ir):
    report = check_citations([("repeated timeout", ["T1"])], log_ir)
    assert not report.has_fabrications


def test_uncited_claims_reduce_coverage(log_ir):
    report = check_citations(
        [("claim with evidence", ["L1"])], log_ir, uncited=["claim without evidence"]
    )
    assert report.claim_coverage == 0.5


def test_lexical_overlap_distinguishes_relevant_from_irrelevant_evidence(log_ir):
    relevant = check_citations([("hikari-main connection pool stats", ["L1"])], log_ir)
    irrelevant = check_citations([("kubernetes ingress certificate rotation", ["L1"])], log_ir)
    assert relevant.checks[0].lexical_overlap > irrelevant.checks[0].lexical_overlap


def test_rows_citing_nonexistent_evidence_are_removed(log_ir):
    output = RCAOutput(
        summary="s",
        severity="SEV2",
        severity_rationale="r",
        timeline=[
            {"evidence_id": "L3", "time": "02:15", "event": "real event"},
            {"evidence_id": "L9999", "time": "99:99", "event": "invented event"},
        ],
        root_cause={"statement": "cause", "evidence_ids": ["L2"]},
        model_confidence="medium",
    )
    claims, uncited = claims_from_rca(output)
    report = check_citations(claims, log_ir, uncited=uncited)
    dropped = drop_fabricated(output, report)

    assert len(dropped) == 1
    assert "invented event" in dropped[0]
    assert [row.evidence_id for row in output.timeline] == ["L3"]


def test_citation_ids_are_normalized_before_comparison():
    """Models emit [L12], 'L12,L13' and L12 interchangeably."""
    from backend.validation.schemas import Evidenced

    item = Evidenced(statement="x", evidence_ids=["[L12]", "L13, L14", " L15 "])
    assert item.evidence_ids == ["L12", "L13", "L14", "L15"]


# --- confidence -----------------------------------------------------------


def _grounding(ratio: float = 1.0, *, fabricated: bool = False, uncited: int = 0) -> GroundingReport:
    """Build a grounding report with a controlled shape.

    `ratio` is the fraction of citations that resolve, produced by mixing in
    IDs the parser never issued. `uncited` adds claims carrying no evidence at
    all, which lowers coverage without implying fabrication.
    """
    ir = parse_logs(LOG_RCA_SAMPLE)
    valid = ["L1", "L2", "L3", "L4"]
    keep = max(1, round(len(valid) * ratio))
    ids = valid[:keep] + [f"L{9000 + i}" for i in range(len(valid) - keep)]
    if fabricated:
        ids = ids + ["L9999"]
    return check_citations(
        [("hikari-main pool stats saturated", ids)],
        ir,
        uncited=[f"claim {i} with no evidence" for i in range(uncited)],
    )


def test_fabrication_forces_low_confidence():
    result = compute_confidence(
        _grounding(1.0, fabricated=True),
        parse_rate=1.0,
        context_strategy="full",
        model_claimed="high",
        contradicting_count=3,
        alternatives_count=3,
        root_cause_evidence_count=5,
    )
    assert result.band == "low"
    assert result.score <= 0.35
    assert any("Fabricated" in w for w in result.warnings)


def test_overconfidence_is_flagged_when_the_model_disagrees_with_the_score():
    """Model says 'high' on an analysis that is thin by every measurable signal."""
    result = compute_confidence(
        _grounding(0.25, uncited=4),
        parse_rate=0.3,
        context_strategy="map_reduce",
        model_claimed="high",
        contradicting_count=0,
        alternatives_count=0,
        root_cause_evidence_count=1,
    )
    assert result.band in {"low", "medium"}
    assert result.overconfident, f"score {result.score:.2f} vs claimed high"
    assert result.model_claimed == "high"


def test_well_evidenced_analysis_scores_high():
    result = compute_confidence(
        _grounding(1.0),
        parse_rate=1.0,
        context_strategy="full",
        model_claimed="high",
        contradicting_count=2,
        alternatives_count=2,
        root_cause_evidence_count=4,
    )
    assert result.band == "high"
    assert not result.overconfident


def test_thin_evidence_caps_the_score():
    result = compute_confidence(
        _grounding(1.0),
        parse_rate=1.0,
        context_strategy="full",
        contradicting_count=2,
        alternatives_count=2,
        root_cause_evidence_count=1,
    )
    assert result.score <= 0.60
    assert any("rests on" in w for w in result.warnings)


def test_compressed_context_lowers_confidence():
    kwargs = dict(parse_rate=1.0, contradicting_count=2, alternatives_count=2, root_cause_evidence_count=4)
    full = compute_confidence(_grounding(1.0), context_strategy="full", **kwargs)
    chunked = compute_confidence(_grounding(1.0), context_strategy="map_reduce", **kwargs)
    assert chunked.score < full.score


def test_confidence_factors_are_explainable():
    result = compute_confidence(
        _grounding(1.0), parse_rate=1.0, context_strategy="full",
        contradicting_count=1, alternatives_count=1, root_cause_evidence_count=3,
    )
    names = {f.name for f in result.factors}
    assert {"citation_validity", "claim_coverage", "input_parse_rate", "context_completeness"} <= names
    assert sum(f.weight for f in result.factors) == pytest.approx(1.0, abs=0.01)


# --- OpenAPI --------------------------------------------------------------


def test_valid_spec_passes():
    report = validate_openapi(VALID_SPEC)
    assert report.parsed and report.schema_valid
    assert report.version == "3.1.0"
    assert report.documented_operations == ["GET /v1/deployments"]


def test_malformed_yaml_is_caught():
    report = validate_openapi("openapi: 3.1.0\n  bad: [indent")
    assert not report.parsed
    assert report.errors
    assert "not parseable YAML" in report.repair_feedback()


def test_yaml_that_is_not_openapi_is_caught():
    """Parsing is not validating; this is the gap the review named."""
    report = validate_openapi("openapi: 3.1.0\ninfo:\n  title: x\npaths: {}")
    assert report.parsed
    assert not report.schema_valid  # `info.version` is required
    assert report.errors


def test_fenced_spec_is_unwrapped():
    report = validate_openapi(f"```yaml\n{VALID_SPEC}\n```")
    assert report.parsed and report.schema_valid


def test_spec_is_cross_checked_against_extracted_routes():
    from backend.parsers.routes import parse_routes
    from backend.samples import API_DOC_SAMPLE

    ir = parse_routes(API_DOC_SAMPLE)
    report = validate_openapi(VALID_SPEC, ir)
    # The spec documents one of three real routes.
    assert report.missing_routes
    assert not report.ok
    assert "missing from paths" in report.repair_feedback()


def test_invented_operations_are_detected():
    from backend.parsers.routes import parse_routes

    ir = parse_routes("from fastapi import APIRouter\nrouter = APIRouter()\n\n@router.get('/real')\ndef real(): ...\n")
    spec = VALID_SPEC.replace("/v1/deployments", "/totally-made-up")
    report = validate_openapi(spec, ir)
    assert "GET /totally-made-up" in report.extra_routes


# --- test execution -------------------------------------------------------

PASSING_TESTS = (
    "import pytest\n"
    "from pricing import apply_tiered_discount, LineItem\n\n"
    "def test_empty_raises():\n"
    "    with pytest.raises(ValueError):\n"
    "        apply_tiered_discount([], 'gold')\n"
)


def test_passing_tests_are_verified_by_running_them():
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, PASSING_TESTS, ["apply_tiered_discount", "LineItem"])
    assert report.ran and report.ok
    assert report.collected == 1 and report.passed == 1


def test_failing_tests_are_caught_with_the_assertion_text():
    failing = PASSING_TESTS + (
        "\ndef test_wrong_expectation():\n"
        "    r = apply_tiered_discount([LineItem('a', 100, 1)], 'gold')\n"
        "    assert r['applied_rate'] == 0.99\n"
    )
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, failing, ["apply_tiered_discount", "LineItem"])
    assert not report.ok
    assert report.failed == 1
    assert "failed" in report.repair_feedback()
    assert report.failure_details


def test_syntax_errors_are_reported_without_spawning_a_process():
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, "def test_x(:\n    pass")
    assert not report.syntax_ok
    assert not report.ran
    assert "syntax error" in report.repair_feedback()


def test_uncollectable_tests_are_reported():
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, "from nonexistent_module import thing\ndef test_a(): thing()")
    assert report.ran
    assert not report.ok
    assert report.errors or report.collection_error


def test_a_file_with_no_tests_is_reported():
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, "def helper():\n    return 1\n")
    assert report.ran
    assert report.collected == 0
    assert "no tests" in report.repair_feedback().lower()


def test_infinite_loops_are_killed_by_the_timeout():
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, "def test_hang():\n    while True:\n        pass\n", timeout=5)
    assert report.ran
    assert "exceeded" in (report.collection_error or "")


def test_execution_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_TEST_EXECUTION", "false")
    report, _ = run_python_tests(UNIT_TEST_SAMPLE, PASSING_TESTS)
    assert not report.ran
    assert report.skipped_reason
