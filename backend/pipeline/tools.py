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

import asyncio
import uuid
from dataclasses import dataclass, field

from ..core import injection
from ..core.chunking import plan_context
from ..core.detect import detect_input_kind, detect_language, mismatch_warning
from ..core.redaction import (
    POLICY_CODE,
    POLICY_LOGS,
    POLICY_STRICT,
    assert_no_secrets,
    fail_closed,
    redact,
)
from ..config import is_routing_model
from ..core.tokens import (
    MIN_USABLE_CONTEXT,
    ROUTING_MODEL_PRIOR,
    build_budget,
    estimate_tokens,
    model_suggestion,
    non_chat_reason,
    suggested_models,
)
from ..groq_client import GroqError
from ..parsers.code import parse_code
from ..parsers.logs import parse_logs
from ..parsers.routes import parse_routes
from ..parsers.transcript import parse_transcript
from ..prompts import (
    TOOL_PROMPTS,
    build_reduce_message,
    build_reduce_system_prompt,
    build_user_message,
)
from ..render import markdown as render
from ..validation import grounding as ground
from ..validation.confidence import compute_confidence
from ..validation.openapi import validate_openapi
from ..validation.python_exec import run_python_tests
from ..validation.schemas import APIDocOutput, PostmortemOutput, RCAOutput, UnitTestOutput
from .base import StructuredResult as StructuredResultLike
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


async def _preprocess_async(tool: str, raw: str, trace: Trace) -> tuple[str, dict, list[str]]:
    """Run preprocessing off the event loop.

    Redaction scans ~20 regexes over the whole input and the injection scan
    another 7. On a multi-megabyte paste that is seconds of CPU, and doing it
    inline blocks every other request in the process.
    """
    return await asyncio.to_thread(_preprocess, tool, raw, trace)


def _preprocess(tool: str, raw: str, trace: Trace) -> tuple[str, dict, list[str]]:
    """Redact, classify and defang. Returns (safe_text, diagnostics, warnings)."""
    warnings: list[str] = []
    diagnostics: dict = {}

    with timed(trace, "redact") as stage:
        redaction = redact(raw, _POLICIES[tool])
        diagnostics["redaction"] = redaction.public()
        stage.summary = {"redacted": redaction.redacted_count}
        if redaction.has_credentials and fail_closed():
            raise GroqError(
                f"Refusing to process: this input contains credentials "
                f"({redaction.counts_by_kind()}) and REDACTION_FAIL_CLOSED is enabled.",
                status=422,
                hint=(
                    "Redaction is pattern matching and cannot guarantee it recognises every "
                    "secret shape. This deployment is configured to refuse rather than rely "
                    "on it. Remove the credentials from the input, or set "
                    "REDACTION_FAIL_CLOSED=false to accept redaction as sufficient."
                ),
            )
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


@dataclass
class ExecutionPlan:
    """The whole deterministic decision, made before any model is called.

    Split out of `_run` so it can also be produced on its own, with no
    upstream call, for a caller that wants to see the plan before paying for
    it. One code path, so the preview cannot describe a plan the run would
    not actually follow.
    """

    plan: object
    budget: object
    model: str
    output_reserve: int
    map_reserve: int
    per_call_ceiling: int

    def public(self) -> dict:
        return {
            "model": self.model,
            "context": self.plan.public(),
            "narrative": self.plan.describe(),
            "budget": self.budget.public(),
            "output_reserved": self.output_reserve,
            "map_output_reserved": self.map_reserve,
            "per_call_ceiling": self.per_call_ceiling,
        }


async def _plan_and_budget(
    tool: str,
    ir,
    system_prompt: str,
    *,
    trace: Trace,
    model: str | None,
    warnings: list[str],
    client_key: str = "local",
) -> ExecutionPlan:
    """Everything decided without the model: window, reserves, and the plan."""
    from ..config import settings
    from ..groq_client import refresh_model_registry_if_stale

    # Context windows come from the provider where possible. One cheap GET per
    # hour beats budgeting against a table that was correct when it was written.
    with timed(trace, "model_metadata") as stage:
        refreshed = await refresh_model_registry_if_stale()
        stage.summary = {"refreshed": refreshed}

    chosen = model or settings.groq_model

    # A model that cannot do chat completions fails somewhere confusing and
    # late. Say so here, by name, with something that works.
    reason = non_chat_reason(chosen)
    if reason:
        raise GroqError(
            f"'{chosen}' is {reason}. This application needs a chat model.",
            status=422,
            hint=(
                f"Set LLM_MODEL to one of: {model_suggestion()}. "
                f"The provider's /models endpoint lists every model your key can call, "
                f"including ones that do not serve chat completions."
            ),
        )

    from ..config import output_tokens_for
    from ..governance import limiter as request_limiter
    from ..governance import token_limiter

    # A flat output reservation is wasteful where it is too large and truncating
    # where it is too small. On a hard tokens-per-minute ceiling the waste is
    # decisive: reserving 4,096 for a reply that runs to 1,800 spends a quarter
    # of an 8,000-token minute on nothing.
    output_reserve = output_tokens_for(tool)
    # A map-reduce partial answers a narrower question over a slice of the
    # input, so it needs far less room than the final combined document.
    map_reserve = output_tokens_for(tool, is_map_step=True)

    with timed(trace, "budget") as stage:
        # The per-minute token allowance caps what one call can carry, whatever
        # the context window says. Budgeting without it produces plans that the
        # token budget then refuses -- which is what "context window too large"
        # actually meant.
        budget = build_budget(
            chosen,
            system_prompt,
            output_reserve,
            token_allowance_per_minute=token_limiter.per_minute,
        )
        stage.summary = {
            "available_for_input": budget.available_for_input,
            "effective_window": budget.effective_window,
            "window_source": budget.window_source,
            "output_reserved": output_reserve,
        }

    # `build_budget` has already folded the token allowance into the usable
    # window, so this is the ceiling for one call.
    per_call_ceiling = budget.available_for_input

    if budget.context_window < MIN_USABLE_CONTEXT:
        raise GroqError(
            f"'{chosen}' has a {budget.context_window:,}-token context window, which is too "
            f"small for this application. The instructions and output contract alone need "
            f"about {budget.reserved_for_system:,} tokens before any of your input.",
            status=422,
            hint=(
                f"At least ~{MIN_USABLE_CONTEXT:,} tokens are needed. Set LLM_MODEL to one of: "
                f"{model_suggestion()}. Window source: {budget.window_source}."
            ),
        )

    if budget.available_for_input <= 0:
        # Name the limit that actually bound. Blaming a 131,072-token context
        # window when the real constraint was an 8,000-token minute sent people
        # looking for a bigger model, which would not have helped at all.
        window_bound = budget.effective_window >= budget.context_window
        constraint = (
            f"'{chosen}' has a {budget.context_window:,}-token context window"
            if window_bound
            else (
                f"a {token_limiter.per_minute:,}-token-per-minute allowance caps each call at "
                f"{budget.effective_window:,} tokens (the model's context window is "
                f"{budget.context_window:,} and is not the constraint here)"
            )
        )
        routing_note = ""
        remedy = (
            f"Use a model with a larger window: {model_suggestion()}."
            if window_bound
            else f"Raise TOKEN_LIMIT_PER_MINUTE above {budget.reserved_for_system + budget.reserved_for_output + budget.safety_margin:,}."
        )
        if is_routing_model(chosen):
            routing_note = (
                f" '{chosen}' is a routing model, so the instruction reserve is inflated by about "
                f"{ROUTING_MODEL_PRIOR:g}x to cover the tool schemas it sends on your behalf; a "
                f"plain chat model needs far less; try {model_suggestion()}."
            )
        raise GroqError(
            f"No budget left for your input: {constraint}, and this request reserves "
            f"{budget.reserved_for_output:,} for the reply, {budget.reserved_for_system:,} for "
            f"instructions and {budget.safety_margin:,} as margin.",
            status=422,
            hint=f"{remedy}{routing_note}",
        )

    if budget.window_source.startswith("conservative"):
        warnings.append(
            f"The context window for '{model or settings.groq_model}' is not known to this "
            f"deployment and could not be read from the provider, so a conservative "
            f"{budget.context_window:,}-token window was assumed. Large inputs may be "
            f"compressed or refused more aggressively than necessary."
        )

    # Splitting is the answer to "this input is bigger than one call". It is
    # not the answer to "one call is bigger than one minute's allowance", and
    # conflating the two wastes a lot of somebody's quota: every part re-sends
    # the system prompt and re-reserves the output, so N parts cost N times the
    # fixed overhead. When that overhead already dominates, more parts is
    # strictly worse than fewer, and the only real fixes are a cheaper model or
    # a higher allowance.
    fixed = budget.reserved_for_system + budget.reserved_for_output + budget.safety_margin
    per_call = fixed + budget.available_for_input
    if per_call > 0 and budget.available_for_input < per_call * 0.25:
        routing_note = ""
        if is_routing_model(chosen):
            routing_note = (
                f" '{chosen}' is a routing model: it sends its own instructions and tool "
                f"schemas alongside your text, so one call costs roughly "
                f"{ROUTING_MODEL_PRIOR:g}x what the visible input suggests. A plain chat "
                f"model does not carry that overhead; try {model_suggestion()}."
            )
        warnings.append(
            f"Only {budget.available_for_input:,} of the {per_call:,} tokens available per call "
            f"can hold your input; the other {fixed:,} are the instructions, the reserved reply "
            f"and the safety margin. Splitting a large input will not help here, because every "
            f"part pays that {fixed:,} again.{routing_note} Raise TOKEN_LIMIT_PER_MINUTE if your "
            f"account allows, or switch LLM_MODEL."
        )

    # Parsing and planning are CPU-bound and run on multi-megabyte inputs.
    # Doing that on the event loop stalls every other request, including
    # /api/health, for the whole duration.
    with timed(trace, "plan_context") as stage:
        # How many upstream calls are left today, so the planner can refuse a
        # plan that would spend more than the caller actually has. Only binds
        # where the provider rations requests; elsewhere the allowance is large
        # enough that it never does.
        _, calls_left = request_limiter.remaining(client_key)
        plan = await asyncio.to_thread(
            plan_context, ir, budget, calls_remaining_today=calls_left
        )
        stage.summary = {
            "strategy": plan.strategy,
            "chunks": len(plan.chunks),
            "effective_per_call": plan.effective_per_call,
        }

    return ExecutionPlan(
        plan=plan,
        budget=budget,
        model=chosen,
        output_reserve=output_reserve,
        map_reserve=map_reserve,
        per_call_ceiling=per_call_ceiling,
    )


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
    client_key: str = "local",
):
    """Budget, plan context, then run either a single call or map-reduce."""
    system_prompt = TOOL_PROMPTS[tool]

    decided = await _plan_and_budget(
        tool, ir, system_prompt, trace=trace, model=model, warnings=warnings,
        client_key=client_key,
    )
    plan = decided.plan
    budget = decided.budget
    output_reserve = decided.output_reserve
    per_call_ceiling = decided.per_call_ceiling
    # How large a partial answer may be is the planner's decision, not a
    # constant: it is what makes the combine tree converge, and it is what the
    # promised call count was computed from. A runtime that uses a different
    # number quietly makes the plan a work of fiction.
    map_reserve = plan.map_output_tokens or decided.map_reserve

    if plan.strategy == "reject":
        # Carry the budget warnings into the hint. A refusal discards `warnings`
        # along with the rest of the response, which threw away the one line
        # that explained *why* only 52 tokens per call were left -- leaving a
        # message that said "raise the limit" without saying what was eating it.
        diagnosis = " ".join(w for w in warnings if "tokens available per call" in w)
        raise GroqError(
            plan.reason,
            status=413,
            hint=(diagnosis or "See the reason above for the available options."),
        )

    if plan.strategy != "full":
        warnings.append(plan.reason)

    # The plan is part of the answer, not an implementation detail: the caller
    # is entitled to know their input was reduced and by what rule.
    plan_note = plan.describe()

    # --- single pass: full or compressed -----------------------------------
    if plan.strategy != "map_reduce":
        chunk = plan.chunks[0]
        result = await call_structured(
            system_prompt,
            build_user_message(
                tool,
                chunk.body,
                # Only what this call actually contains. Declaring the whole
                # input's ID space would let the model cite a line it never
                # saw, and that citation would pass the grounding check.
                evidence_ids=chunk.item_ids,
                context_note=plan_note,
                warnings=warnings,
            ),
            schema,
            trace=trace,
            model=model,
            temperature=temperature,
            max_tokens=budget.reserved_for_output,
            extra_validators=extra_validators,
            client_key=client_key,
        )
        return result, plan

    # --- map-reduce ---------------------------------------------------------
    # Each part is analysed against only its own content and its own ID space,
    # then a combine step reasons over the partial findings plus the global
    # overview. Without this, everything after the first chunk was silently
    # discarded while the response still claimed to have analysed the input.
    partials: list[str] = []
    cited_ids: set[str] = set()
    accumulated: StructuredResultLike | None = None

    for chunk in plan.chunks:
        # Wait for token budget rather than failing mid-run. A plan that was
        # accepted should complete; the planner already refused anything that
        # would wait longer than MAX_PLAN_SECONDS.
        await _await_token_budget(
            client_key,
            chunk.estimated_tokens + budget.reserved_for_output,
            trace,
            f"part {chunk.index + 1}",
        )
        part = await call_structured(
            system_prompt,
            build_user_message(
                tool,
                chunk.body,
                evidence_ids=chunk.item_ids,
                context_note=(
                    f"{plan_note}\n\nThis is part {chunk.index + 1} of {chunk.total}. Analyse "
                    f"only what is present here; a combining step will resolve anything that "
                    f"spans parts."
                ),
                warnings=warnings,
            ),
            schema,
            trace=trace,
            model=model,
            temperature=temperature,
            max_tokens=map_reserve,
            stage_name=f"map.part{chunk.index + 1}",
            client_key=client_key,
            # Domain validators (executing tests, validating OpenAPI) apply to
            # the final answer, not to a partial view of the input.
            extra_validators=None,
        )
        partials.append(part.value.model_dump_json(indent=None))
        cited_ids.update(_cited_ids(tool, part.value))
        accumulated = part if accumulated is None else accumulated.merge(part)

    # --- combine ------------------------------------------------------------
    # With many parts the combined findings exceed a single call's budget on
    # their own, so the combine is a tree: batches are merged, then the merges
    # are merged, until one answer remains. A flat combine would either
    # overflow the budget or silently drop partials, which is the failure this
    # whole layer exists to prevent.
    reduce_system = build_reduce_system_prompt(tool)
    # From the plan, not recomputed here. The plan promised a call count based
    # on these two numbers, and a runtime that batches by its own arithmetic
    # makes that promise false -- which is how the preview came to advertise
    # fourteen calls for a run that made twenty-seven.
    reduce_ceiling = plan.reduce_ceiling or max(
        1_000, per_call_ceiling - estimate_tokens(plan.skeleton) - 400
    )

    def batches_for(items: list[str]) -> list[list[str]]:
        """Group partials so each combine call stays inside the budget."""
        groups: list[list[str]] = []
        current: list[str] = []
        used = 0
        for item in items:
            cost = estimate_tokens(item)
            if current and used + cost > reduce_ceiling:
                groups.append(current)
                current, used = [], 0
            current.append(item)
            used += cost
        if current:
            groups.append(current)
        return groups

    level = 0
    pending = partials
    reduced = None

    while True:
        groups = batches_for(pending)
        level += 1

        with timed(trace, f"reduce.level{level}") as stage:
            stage.summary = {
                "inputs": len(pending),
                "batches": len(groups),
                "distinct_ids_cited": len(cited_ids),
            }

        if len(groups) == 1:
            reduced = await call_structured(
                reduce_system,
                build_reduce_message(tool, plan.skeleton, groups[0], evidence_ids=sorted(cited_ids)),
                schema,
                trace=trace,
                model=model,
                temperature=temperature,
                max_tokens=output_reserve,
                stage_name="reduce.final",
                extra_validators=extra_validators,
                client_key=client_key,
            )
            break

        if level > 4:  # pathological input; stop rather than loop
            raise GroqError(
                f"Combining {len(partials)} partial analyses did not converge within 4 levels.",
                status=413,
                hint=(
                    "Narrow the input, or raise TOKEN_LIMIT_PER_MINUTE so the analysis needs "
                    "fewer parts."
                ),
            )

        merged: list[str] = []
        for index, group in enumerate(groups, start=1):
            partial = await call_structured(
                reduce_system,
                build_reduce_message(tool, plan.skeleton, group, evidence_ids=sorted(cited_ids)),
                schema,
                trace=trace,
                model=model,
                temperature=temperature,
                max_tokens=map_reserve,
                stage_name=f"reduce.L{level}.batch{index}",
                client_key=client_key,
            )
            merged.append(partial.value.model_dump_json(indent=None))
            accumulated = partial if accumulated is None else accumulated.merge(partial)
        pending = merged

    warnings.append(
        f"This input was analysed in {len(plan.chunks)} parts and combined over {level} "
        f"level(s), {trace.upstream_calls} model calls in total. No single call saw the "
        f"whole input, so "
        f"relationships spanning parts rest on the combining step rather than on direct "
        f"observation. Confidence is reduced accordingly."
    )

    final = accumulated.merge(reduced) if accumulated else reduced
    return final, plan


async def _await_token_budget(client_key: str, needed: int, trace: Trace, label: str) -> None:
    """Block until the token allowance can carry `needed`, or give up cleanly.

    A multi-call plan that starts and then dies half way through has spent real
    quota for nothing. Waiting keeps the run whole; the planner has already
    refused plans whose total wait would be unreasonable.
    """
    from ..governance import token_limiter

    deadline = asyncio.get_running_loop().time() + 300
    waited = 0.0
    while True:
        allowed, _reason, retry_after = token_limiter.check(client_key, needed)
        if allowed:
            if waited:
                trace.record(f"paced.{label}", asyncio.get_running_loop().time() - waited, {"waited_s": round(waited, 1)})
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise GroqError(
                f"Gave up waiting for token budget before {label}.",
                status=429,
                hint="Raise TOKEN_LIMIT_PER_MINUTE, or narrow the input so fewer calls are needed.",
            )
        pause = min(max(1, retry_after), 15)
        waited += pause
        await asyncio.sleep(pause)


def _cited_ids(tool: str, value) -> set[str]:
    """Every evidence ID a partial answer referenced."""
    getter = getattr(value, "all_evidence_ids", None)
    if callable(getter):
        return {i for i in getter() if i}
    # Tools whose schema carries no evidence IDs (tests, docs) contribute none.
    return set()


# --- log / RCA ------------------------------------------------------------


async def run_log_rca(raw: str, *, model=None, temperature=None, request_id=None, client_key="local") -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = await _preprocess_async("log-rca", raw, trace)

    with timed(trace, "parse") as stage:
        ir = await asyncio.to_thread(parse_logs, safe)
        stage.summary = {"lines": ir.total_lines, "templates": len(ir.templates)}
    diagnostics["parse"] = ir.stats()

    if not ir.entries:
        raise GroqError("No log lines could be parsed from this input.", status=422,
                        hint="The Log/RCA tool expects log output. Check you pasted the right thing.")

    result, plan = await _run(
        "log-rca", ir, ir.render("full"), RCAOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
        client_key=client_key,
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


async def run_postmortem(raw: str, *, model=None, temperature=None, request_id=None, client_key="local") -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = await _preprocess_async("postmortem", raw, trace)

    with timed(trace, "parse") as stage:
        ir = await asyncio.to_thread(parse_transcript, safe)
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
        extra_validators=[check_roles], client_key=client_key,
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


async def run_unit_tests(raw: str, *, model=None, temperature=None, request_id=None, client_key="local") -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = await _preprocess_async("unit-tests", raw, trace)

    with timed(trace, "parse") as stage:
        language = detect_language(safe).value
        ir = await asyncio.to_thread(parse_code, safe, language)
        stage.summary = {"language": ir.language, "functions": len(ir.all_functions())}
    diagnostics["parse"] = ir.stats()

    if ir.syntax_error and ir.language == "python":
        warnings.append(
            f"The source does not parse as valid Python ({ir.syntax_error}). Structural extraction "
            f"was skipped, so the model received raw text and the result is materially less reliable."
        )
    if not ir.all_functions():
        warnings.append("No functions or methods were extracted; there may be nothing to test.")
    elif ir.confidence < 0.6:
        warnings.append(
            f"Structural extraction for {ir.language} uses {ir.parser}, which recovers "
            f"signatures but not boundary conditions, raised exceptions or dependency "
            f"boundaries (confidence {ir.confidence:.0%}). The model is working from a "
            f"weaker picture than it would for Python, so coverage claims below are less "
            f"reliable. Review the generated cases rather than trusting the coverage figure."
        )

    boundaries = [c for f in ir.all_functions() for c in f.comparisons]
    defined = [f.name for f in ir.all_functions()] + [c.name for c in ir.classes]

    # The model's own claim about running the tests is checked by running them,
    # and a failure feeds the same repair loop as a schema error.
    execution_holder: dict = {}

    async def check_execution(value) -> str:
        if ir.language != "python" or not value.test_code.strip():
            return ""
        if ir.syntax_error:
            # The source cannot be imported, so no generated test could pass.
            # Retrying would spend the whole repair budget proving that.
            execution_holder["report"] = None
            return ""
        # subprocess.run blocks for up to the timeout; off the loop it goes.
        report, notes = await asyncio.to_thread(run_python_tests, safe, value.test_code, defined)
        execution_holder["report"] = report
        execution_holder["notes"] = notes
        if report.skipped_reason:
            return ""
        return report.repair_feedback()

    result, plan = await _run(
        "unit-tests", ir, ir.render("full"), UnitTestOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
        extra_validators=[check_execution], client_key=client_key,
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


async def run_api_docs(raw: str, *, model=None, temperature=None, request_id=None, client_key="local") -> PipelineResult:
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    safe, diagnostics, warnings = await _preprocess_async("api-docs", raw, trace)

    with timed(trace, "parse") as stage:
        ir = await asyncio.to_thread(parse_routes, safe)
        stage.summary = {"framework": ir.framework, "routes": len(ir.routes)}
    diagnostics["parse"] = ir.stats()

    if not ir.routes:
        warnings.append(
            "No routes could be extracted. The model received the raw source, so the OpenAPI "
            "cross-check cannot verify that the spec matches the code."
        )
    elif "regex" in ir.parser:
        warnings.append(
            f"Routes were extracted with {ir.parser} rather than a real parser, so parameter "
            f"types, request bodies and error paths are likely incomplete. The OpenAPI "
            f"cross-check can only verify against what was extracted, so a missing route "
            f"here will not be reported as missing."
        )

    openapi_holder: dict = {}

    def check_openapi(value) -> str:
        report = validate_openapi(value.openapi_yaml, ir if ir.routes else None)
        openapi_holder["report"] = report
        return report.repair_feedback()

    result, plan = await _run(
        "api-docs", ir, ir.render("full"), APIDocOutput,
        trace=trace, model=model, temperature=temperature, warnings=warnings,
        extra_validators=[check_openapi], client_key=client_key,
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


# --- plan preview ---------------------------------------------------------
#
# The same deterministic front half, run on its own with no upstream call.
#
# It exists because "how is this being split, and why" is a question the caller
# should be able to ask before spending quota, not only read afterwards in a
# diagnostics panel. Answering it from a separate estimator would be worse than
# not answering it: a preview that disagrees with the run is a lie with a
# progress bar. So this calls the same parsers and the same planner, and the
# only thing it leaves out is the model.


_PARSERS = {
    "log-rca": lambda safe: parse_logs(safe),
    "postmortem": lambda safe: parse_transcript(safe),
    "unit-tests": lambda safe: parse_code(safe, detect_language(safe).value),
    "api-docs": lambda safe: parse_routes(safe),
}

# What each tool's deterministic stages extracted, in the caller's language.
# The point of the preview is to show that real work happens before the model
# is involved; "parsed 2,318 lines into 41 message patterns" makes that
# concrete in a way that "preprocessing complete" does not.
_EXTRACTION_SUMMARY = {
    "log-rca": lambda ir: [
        f"{ir.total_lines:,} log lines parsed and given stable IDs",
        f"{len(ir.templates):,} distinct message patterns mined by variable masking",
        f"{ir.stats()['parse_rate']:.0%} of lines matched a known timestamp/level format",
    ],
    "postmortem": lambda ir: [
        f"{len(ir.utterances):,} speaker turns parsed",
        f"{len(ir.speakers):,} participants found and replaced with role placeholders",
    ],
    "unit-tests": lambda ir: [
        f"language detected as {ir.language} (parser: {ir.parser})",
        f"{len(ir.all_functions()):,} functions and methods extracted",
        f"{sum(len(f.comparisons) for f in ir.all_functions()):,} boundary comparisons extracted from branches",
        f"{sum(len(f.raises) for f in ir.all_functions()):,} explicit raise sites found",
        f"{sum(len(f.external_calls) for f in ir.all_functions()):,} external calls that will need mocking",
    ],
    "api-docs": lambda ir: [
        f"framework detected as {ir.framework} (parser: {ir.parser})",
        f"{len(ir.routes):,} routes extracted",
        f"{len(ir.models):,} request/response schemas found",
    ],
}


async def preview(tool: str, raw: str, *, model=None, request_id=None) -> dict:
    """Run everything up to the model call and report what would happen.

    Costs no tokens and makes no upstream request beyond the hourly model
    metadata refresh, which is cached.
    """
    trace = Trace(request_id or uuid.uuid4().hex[:12])
    warnings: list[str] = []

    safe, diagnostics, warnings = await _preprocess_async(tool, raw, trace)

    with timed(trace, "parse") as stage:
        ir = await asyncio.to_thread(_PARSERS[tool], safe)
        stage.summary = {"kind": type(ir).__name__}
    diagnostics["parse"] = ir.stats()

    decided = await _plan_and_budget(
        tool, ir, TOOL_PROMPTS[tool], trace=trace, model=model, warnings=warnings
    )  # preview: no client key, so the daily budget does not narrow the plan
    plan = decided.plan

    # A plan that would be refused is reported as a plan, not raised as an
    # error: the caller asked what would happen, and "it would be refused,
    # here is the arithmetic" is the answer.
    steps = _preview_steps(tool, plan)

    return {
        "tool": tool,
        "request_id": trace.request_id,
        "feasible": plan.strategy != "reject",
        "extracted": _EXTRACTION_SUMMARY[tool](ir),
        "steps": steps,
        "warnings": warnings,
        "diagnostics": diagnostics,
        **decided.public(),
    }


def _preview_steps(tool: str, plan) -> list[dict]:
    """The ordered work, deterministic stages and model calls alike.

    Both are listed because both are real work, and showing only the model
    calls would reproduce exactly the impression this architecture exists to
    correct -- that the answer is what the model said.
    """
    steps: list[dict] = [
        {"kind": "deterministic", "name": "Redact secrets", "detail": "~20 credential patterns, before anything leaves this process"},
        {"kind": "deterministic", "name": "Scan for prompt injection", "detail": "instruction-shaped text in the input is neutralised, not obeyed"},
        {"kind": "deterministic", "name": "Parse to structured facts", "detail": "stable IDs assigned so every later claim can be checked against one"},
        {"kind": "deterministic", "name": "Plan the context", "detail": plan.reason},
    ]

    if plan.strategy == "reject":
        steps.append({"kind": "blocked", "name": "Refused", "detail": plan.reason})
        return steps

    if plan.strategy == "map_reduce":
        for chunk in plan.chunks:
            steps.append({
                "kind": "model",
                "name": f"Analyse part {chunk.index + 1} of {chunk.total}",
                "detail": (
                    f"{len(chunk.item_ids):,} items ({chunk.item_ids[0] if chunk.item_ids else '?'}"
                    f"..{chunk.item_ids[-1] if chunk.item_ids else '?'}), "
                    f"~{chunk.estimated_tokens:,} tokens, carrying the global overview"
                ),
            })
        # The combine is a tree when the partials do not fit one call together,
        # so every round is listed. Showing one step for what is really a dozen
        # calls is the same lie as showing one call for a dozen parts.
        combine_calls = plan.estimated_calls - len(plan.chunks)
        for index in range(combine_calls):
            final = index == combine_calls - 1
            steps.append({
                "kind": "model",
                "name": "Combine the partial findings" if final else f"Merge partial findings ({index + 1})",
                "detail": (
                    "reasons over every part's findings plus the overview; may cite only what the parts cited"
                    if final
                    else "a batch of partial answers merged into one, because they do not fit a single combine call"
                ),
            })
    else:
        steps.append({
            "kind": "model",
            "name": "Analyse",
            "detail": f"one call, ~{plan.estimated_input_tokens:,} input tokens ({plan.strategy})",
        })

    steps += [
        {"kind": "deterministic", "name": "Verify every citation exists", "detail": "claims citing an ID that was never sent are dropped, not rendered"},
        {"kind": "deterministic", "name": "Run the domain validators", "detail": _VALIDATOR_BLURB[tool]},
        {"kind": "deterministic", "name": "Compute confidence", "detail": "from measured signals -- grounding ratio, parse rate, strategy, contradictions -- not from the model's own estimate"},
        {"kind": "deterministic", "name": "Render", "detail": "from validated structures, so an unverified response has nothing to render from"},
    ]
    return steps


_VALIDATOR_BLURB = {
    "unit-tests": "the generated tests are executed with pytest in a sandbox; failures are reported, not hidden",
    "api-docs": "the OpenAPI document is schema-validated offline and cross-checked against the parsed routes",
    "log-rca": "timeline ordering and evidence-per-claim are checked against the parsed log",
    "postmortem": "every name is checked against the allowed role placeholders; invented names are rejected",
}
