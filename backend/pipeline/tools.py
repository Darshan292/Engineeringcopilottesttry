"""The four tool pipelines.

Every tool runs the same ordered stages, and the ordering is the design:

    redact -> detect -> scan for injection -> parse to IR -> budget -> plan
    context -> build prompt from facts -> structured LLM call (with repair)
    -> verify evidence -> run domain validators -> compute confidence
    -> render Markdown

The LLM sits in the middle, doing only the step that needs judgement. Secrets
are gone before it, structure is extracted before it, and everything it says is
checked after it. That is the difference between this and passing raw text to a
prompt: no single stage is novel, but the model is no longer the only thing
standing between a paste and an answer.

Stages before the call cannot be skipped, because redaction and parsing produce
what the prompt is built from. Stages after it cannot be skipped either -- the
renderer takes validated structures, so an unverified response has nothing to
render from.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..core import injection
from ..core.chunking import plan_context
from ..core.detect import detect_input_kind, detect_language, mismatch_warning
from ..core.redaction import POLICY_CODE, POLICY_LOGS, POLICY_STRICT, assert_no_secrets, redact
from ..core.tokens import build_budget
from ..groq_client import GroqError
from ..parsers.code import parse_code
from ..parsers.logs import parse_logs
from ..parsers.routes import parse_routes
from ..parsers.transcript import parse_transcript
from ..prompts import TOOL_PROMPTS, build_user_message
from ..render import markdown as render
from ..validation import grounding as ground
from ..validation.confidence import compute_confidence
from ..validation.openapi import validate_openapi
from ..validation.python_exec import run_python_tests
from ..validation.schemas import APIDocOutput, PostmortemOutput, RCAOutput, UnitTestOutput
from .base import Trace, call_structured, timed


@dataclass
class PipelineResult:
    tool: str
    markdown: str
    trace: Trace
    model: str
    elapsed_ms: int
    usage: dict
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    attempts: int = 1
    repairs: list[str] = field(default_factory=list)


_POLICIES = {
    "unit-tests": POLICY_CODE,
    "api-docs": POLICY_CODE,
    "log-rca": POLICY_LOGS,
    "postmortem": POLICY_STRICT,
}


def _preprocess(tool: str, raw: str, trace: Trace) -> tuple[str, dict, list[str]]:
    """Redact, classify and defang. Returns (safe_text, diagnostics, warnings)."""
    warnings: list[str] = []
    diagnostics: dict = {}

    with timed(trace, "redact") as stage:
        redaction = redact(raw, _POLICIES[tool])
        diagnostics["redaction"] = redaction.public()
        stage.summary = {"redacted": redaction.redacted_count}
        if redaction.has_credentials:
            warnings.append(
                f"{redaction.redacted_count} secret(s) or personal identifier(s) were removed "
                f"before the request left this machine: {redaction.counts_by_kind()}. They were "
                f"replaced with stable [[REDACTED:...]] placeholders."
            )

    with timed(trace, "detect") as stage:
        kind = detect_input_kind(redaction.text)
        diagnostics["input_kind"] = {"kind": kind.kind, "confidence": round(kind.confidence, 3), **kind.evidence}
        stage.summary = {"kind": kind.kind, "confidence": round(kind.confidence, 2)}
        mismatch = mismatch_warning(tool, kind)
        if mismatch:
            warnings.append(mismatch)

    with timed(trace, "injection_scan") as stage:
        scan = injection.neutralize(redaction.text)
        diagnostics["injection"] = scan.public()
        stage.summary = {"detected": scan.detected, "severity": scan.max_severity}
        if scan.detected:
            warnings.append(
                f"Prompt-injection patterns were detected in the input ({', '.join(scan.public()['kinds'])}, "
                f"severity {scan.max_severity}) and neutralized before the call. The content is still "
                f"analysed; the instructions in it were not followed."
            )

    safe = scan.neutralized_text

    # Final gate: nothing that was redacted may survive into the outbound text.
    leaked = assert_no_secrets(safe, redaction)
    if leaked:
        raise GroqError(
            f"Refusing to send: redacted values of kind {leaked} survived preprocessing.",
            status=500,
            hint="This is a bug in the redaction pipeline. Please report the input shape that caused it.",
        )

    return safe, diagnostics, warnings


async def _run(
    tool: str,
    ir,
    facts: str,
    schema,
    *,
    trace: Trace,
    model: str | None,
    temperature: float | None,
    warnings: list[str],
    extra_validators: list | None = None,
    context_note: str = "",
):
    """Budget, plan context and make the structured call."""
    system_prompt = TOOL_PROMPTS[tool]

    with timed(trace, "budget") as stage:
        from ..config import settings

        budget = build_budget(model or settings.groq_model, system_prompt, settings.max_tokens)
        stage.summary = {"available_for_input": budget.available_for_input}

    with timed(trace, "plan_context") as stage:
        plan = plan_context(ir, budget)
        stage.summary = {"strategy": plan.strategy, "chunks": len(plan.chunks)}

    if plan.strategy == "reject":
        raise GroqError(plan.reason, status=413, hint="See the reason above for the available options.")

    if plan.strategy != "full":
        warnings.append(plan.reason)

    # Single-pass covers full and summary. Map-reduce is handled by the caller
    # only for tools where partial analysis is meaningful; for the others the
    # planner's chunk cap has already turned oversize input into a rejection.
    body = plan.chunks[0].body if plan.chunks else facts

    user_message = build_user_message(
        tool,
        body,
        evidence_ids=sorted(ir.evidence_ids()),
        context_note=context_note or (plan.reason if plan.strategy != "full" else ""),
        warnings=warnings,
    )

    result = await call_structured(
        system_prompt,
        user_message,
        schema,
        trace=trace,
        model=model,
        temperature=temperature,
        extra_validators=extra_validators,
    )
    return result, plan


# --- log / RCA ------------------------------------------------------------


async def run_log_rca(raw: str, *, model=None, temperature=None, request_id=None) -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = _preprocess("log-rca", raw, trace)

    with timed(trace, "parse") as stage:
        ir = parse_logs(safe)
        stage.summary = {"lines": ir.total_lines, "templates": len(ir.templates)}
    diagnostics["parse"] = ir.stats()

    if not ir.entries:
        raise GroqError("No log lines could be parsed from this input.", status=422,
                        hint="The Log/RCA tool expects log output. Check you pasted the right thing.")

    result, plan = await _run(
        "log-rca", ir, ir.render("full"), RCAOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
    )
    output = result.value

    with timed(trace, "verify_evidence") as stage:
        claims, uncited = ground.claims_from_rca(output)
        grounding = ground.check_citations(claims, ir, uncited=uncited)
        dropped = ground.drop_fabricated(output, grounding)
        stage.summary = {"grounding_ratio": round(grounding.grounding_ratio, 2), "dropped_rows": len(dropped)}
    diagnostics["grounding"] = grounding.public()

    with timed(trace, "compute_confidence") as stage:
        confidence = compute_confidence(
            grounding,
            parse_rate=ir.stats()["parse_rate"],
            context_strategy=plan.strategy,
            model_claimed=output.model_confidence,
            contradicting_count=len(output.contradicting_evidence),
            alternatives_count=len(output.alternative_hypotheses),
            root_cause_evidence_count=len(output.root_cause.evidence_ids),
        )
        stage.summary = {"band": confidence.band, "score": round(confidence.score, 2)}
    diagnostics["confidence"] = confidence.public()

    with timed(trace, "render"):
        body = render.render_rca(
            output, ir_stats=ir.stats(), confidence=confidence,
            grounding=grounding.public(), context=plan.public(),
            dropped=dropped, warnings=warnings,
        )

    return PipelineResult(
        tool="log-rca", markdown=body, trace=trace, model=result.model,
        elapsed_ms=result.elapsed_ms, usage=result.usage, warnings=warnings,
        diagnostics=diagnostics, attempts=result.attempts, repairs=result.repairs,
    )


# --- postmortem -----------------------------------------------------------


async def run_postmortem(raw: str, *, model=None, temperature=None, request_id=None) -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = _preprocess("postmortem", raw, trace)

    with timed(trace, "parse") as stage:
        ir = parse_transcript(safe)
        stage.summary = {"utterances": len(ir.utterances), "participants": len(ir.speakers)}
    diagnostics["parse"] = ir.stats()

    if not ir.utterances:
        raise GroqError(
            "No speaker turns could be parsed from this transcript.", status=422,
            hint="Expected lines like '[02:18] Name: message'. Check the format of what you pasted.",
        )

    allowed_roles = ir.allowed_placeholders()

    def check_roles(value) -> str:
        """Timeline actors and the IC must be people who were actually present.

        Action-item owners are checked separately and only for shape, because a
        follow-up can legitimately be owned by a role that never appeared in
        the channel.
        """
        unknown = [
            role for role in value.participant_roles()
            if role not in allowed_roles and role not in {"[UNKNOWN]", "[UNASSIGNED]"}
        ]
        if unknown:
            # Deliberately does NOT echo the offending values. They are model
            # output, they may be real names, and this message is returned to
            # the caller in `repairs` -- echoing it would leak exactly what the
            # anonymization upstream exists to prevent.
            return (
                f"{len(set(unknown))} role value(s) you used as incident participants do not "
                f"correspond to anyone in this transcript. For the incident commander and every "
                f"timeline actor, use only these placeholders: "
                f"{', '.join(sorted(allowed_roles))}, or [UNKNOWN]. Never use a person's name. "
                f"(Action-item owners are not restricted to participants.)"
            )
        malformed = [r for r in value.owner_roles() if not (r.startswith("[") and r.endswith("]"))]
        if malformed:
            return f"Action-item owners must be bracketed role placeholders, not: {', '.join(malformed[:5])}."
        return ""

    result, plan = await _run(
        "postmortem", ir, ir.render("full"), PostmortemOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
        extra_validators=[check_roles],
    )
    output = result.value

    with timed(trace, "verify_evidence") as stage:
        claims, uncited = ground.claims_from_postmortem(output)
        grounding = ground.check_citations(claims, ir, uncited=uncited)
        dropped = ground.drop_fabricated(output, grounding)
        stage.summary = {"grounding_ratio": round(grounding.grounding_ratio, 2), "dropped_rows": len(dropped)}
    diagnostics["grounding"] = grounding.public()

    with timed(trace, "compute_confidence") as stage:
        confidence = compute_confidence(
            grounding,
            parse_rate=(ir.parsed_lines / ir.total_lines) if ir.total_lines else 0.0,
            context_strategy=plan.strategy,
            contradicting_count=len(output.contributing_factors),
            alternatives_count=len(output.open_questions),
            root_cause_evidence_count=len(output.root_cause.evidence_ids),
        )
    diagnostics["confidence"] = confidence.public()

    with timed(trace, "render"):
        body = render.render_postmortem(
            output, ir_stats=ir.stats(), confidence=confidence,
            grounding=grounding.public(), context=plan.public(),
            dropped=dropped, warnings=warnings,
        )

    return PipelineResult(
        tool="postmortem", markdown=body, trace=trace, model=result.model,
        elapsed_ms=result.elapsed_ms, usage=result.usage, warnings=warnings,
        diagnostics=diagnostics, attempts=result.attempts, repairs=result.repairs,
    )


# --- unit tests -----------------------------------------------------------


async def run_unit_tests(raw: str, *, model=None, temperature=None, request_id=None) -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = _preprocess("unit-tests", raw, trace)

    with timed(trace, "parse") as stage:
        language = detect_language(safe).value
        ir = parse_code(safe, language)
        stage.summary = {"language": ir.language, "functions": len(ir.all_functions())}
    diagnostics["parse"] = ir.stats()

    if ir.syntax_error and ir.language == "python":
        warnings.append(
            f"The source does not parse as valid Python ({ir.syntax_error}). Structural extraction "
            f"was skipped, so the model received raw text and the result is materially less reliable."
        )
    if not ir.all_functions():
        warnings.append("No functions or methods were extracted; there may be nothing to test.")

    boundaries = [c for f in ir.all_functions() for c in f.comparisons]
    defined = [f.name for f in ir.all_functions()] + [c.name for c in ir.classes]

    # The model's own claim about running the tests is checked by running them,
    # and a failure feeds the same repair loop as a schema error.
    execution_holder: dict = {}

    def check_execution(value) -> str:
        if ir.language != "python" or not value.test_code.strip():
            return ""
        if ir.syntax_error:
            # The source cannot be imported, so no generated test could pass.
            # Retrying would spend the whole repair budget proving that.
            execution_holder["report"] = None
            return ""
        report, notes = run_python_tests(safe, value.test_code, defined)
        execution_holder["report"] = report
        execution_holder["notes"] = notes
        if report.skipped_reason:
            return ""
        return report.repair_feedback()

    result, plan = await _run(
        "unit-tests", ir, ir.render("full"), UnitTestOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
        extra_validators=[check_execution],
    )
    output = result.value
    execution = execution_holder.get("report")
    if execution_holder.get("notes"):
        warnings.append(
            "The generated test file's imports had to be rewritten to run against the pasted "
            f"source ({'; '.join(execution_holder['notes'])}). Adjust them for your real module layout."
        )
    if execution:
        diagnostics["execution"] = execution.public()

    with timed(trace, "coverage_check") as stage:
        claimed = {c.covers_boundary.strip() for c in output.cases if c.covers_boundary.strip()}
        covered = [b for b in boundaries if b in claimed]
        uncovered = [b for b in boundaries if b not in claimed]
        coverage = {
            "total_boundaries": len(boundaries),
            "covered": len(covered),
            "uncovered": uncovered,
            "ratio": (len(covered) / len(boundaries)) if boundaries else 1.0,
        }
        stage.summary = {"boundary_coverage": round(coverage["ratio"], 2)}
    diagnostics["coverage"] = {k: v for k, v in coverage.items() if k != "uncovered"}

    # Cases pointing at functions that do not exist are the code-tool
    # equivalent of a fabricated citation.
    valid_ids = {f.id for f in ir.all_functions()}
    bogus = sorted({c.target_function_id for c in output.cases if c.target_function_id and c.target_function_id not in valid_ids})
    if bogus:
        warnings.append(f"Some cases target function IDs that were not extracted: {', '.join(bogus)}.")

    with timed(trace, "render"):
        body = render.render_unit_tests(
            output, ir_stats=ir.stats(), execution=execution,
            coverage=coverage, warnings=warnings, context=plan.public(),
        )

    return PipelineResult(
        tool="unit-tests", markdown=body, trace=trace, model=result.model,
        elapsed_ms=result.elapsed_ms, usage=result.usage, warnings=warnings,
        diagnostics=diagnostics, attempts=result.attempts, repairs=result.repairs,
    )


# --- API docs -------------------------------------------------------------


async def run_api_docs(raw: str, *, model=None, temperature=None, request_id=None) -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = _preprocess("api-docs", raw, trace)

    with timed(trace, "parse") as stage:
        ir = parse_routes(safe)
        stage.summary = {"framework": ir.framework, "routes": len(ir.routes)}
    diagnostics["parse"] = ir.stats()

    if not ir.routes:
        warnings.append(
            "No routes could be extracted. The model received the raw source, so the OpenAPI "
            "cross-check cannot verify that the spec matches the code."
        )

    openapi_holder: dict = {}

    def check_openapi(value) -> str:
        report = validate_openapi(value.openapi_yaml, ir if ir.routes else None)
        openapi_holder["report"] = report
        return report.repair_feedback()

    result, plan = await _run(
        "api-docs", ir, ir.render("full"), APIDocOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
        extra_validators=[check_openapi],
    )
    output = result.value
    openapi_report = openapi_holder.get("report")
    if openapi_report:
        diagnostics["openapi"] = openapi_report.public()

    valid_route_ids = {r.id for r in ir.routes}
    bogus = sorted({e.route_id for e in output.endpoints if e.route_id and e.route_id not in valid_route_ids})
    if bogus:
        warnings.append(f"Documented endpoints reference route IDs that were not extracted: {', '.join(bogus)}.")

    with timed(trace, "render"):
        body = render.render_api_docs(
            output, ir_stats=ir.stats(), openapi_report=openapi_report,
            warnings=warnings, context=plan.public(),
        )

    return PipelineResult(
        tool="api-docs", markdown=body, trace=trace, model=result.model,
        elapsed_ms=result.elapsed_ms, usage=result.usage, warnings=warnings,
        diagnostics=diagnostics, attempts=result.attempts, repairs=result.repairs,
    )


PIPELINES = {
    "unit-tests": run_unit_tests,
    "api-docs": run_api_docs,
    "log-rca": run_log_rca,
    "postmortem": run_postmortem,
}
