"""Render validated structures to Markdown.

The model never writes a document. It fills a typed structure; this renders it.

That is what makes the output contract real. Section order, heading text, table
columns and placeholder handling are decided by code, so they are identical on
every run regardless of how the model felt about formatting that day. A missing
section is impossible -- the schema required it, or the renderer omits it
deliberately and says why.

It also means the reliability metadata is not optional. Computed confidence,
fabricated citations, execution results and validation failures are rendered
into the document by the same pass that renders the content, so a reader cannot
receive the conclusion without the caveats attached to it.
"""

from __future__ import annotations

from ..validation.confidence import ConfidenceResult

_BAND_MARK = {"high": "High", "medium": "Medium", "low": "Low"}


def _escape_cell(text: str) -> str:
    """Table cells must not contain raw pipes or newlines."""
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None._\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_escape_cell(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _bullets(items: list[str], empty: str = "_None identified._") -> str:
    if not items:
        return empty + "\n"
    return "\n".join(f"- {item}" for item in items) + "\n"


def _cite(ids: list[str]) -> str:
    return f" `[{', '.join(ids)}]`" if ids else ""


def render_reliability(
    confidence: ConfidenceResult | None,
    *,
    grounding: dict | None = None,
    context: dict | None = None,
    extra_warnings: list[str] | None = None,
) -> str:
    """The block every document carries, stating how far to trust it."""
    lines = ["## Reliability\n"]

    if confidence:
        lines.append(
            f"**Computed confidence: {_BAND_MARK.get(confidence.band, confidence.band)} "
            f"({confidence.score:.0%})** — derived in code from citation validity, evidence "
            f"coverage and input completeness. It measures how well-evidenced this analysis "
            f"is, not whether it is correct.\n"
        )
        if confidence.model_claimed:
            agreement = (
                "The model rated its own confidence "
                f"**{confidence.model_claimed}**"
            )
            if confidence.overconfident:
                agreement += (
                    " — materially higher than the computed score. Treat the conclusions "
                    "below with corresponding caution."
                )
            else:
                agreement += ", which is consistent with the computed score."
            lines.append(agreement + "\n")

        rows = [
            [f.name.replace("_", " "), f"{f.value:.0%}", f"{f.weight:.2f}", f.detail]
            for f in confidence.factors
        ]
        lines.append(_table(["Factor", "Score", "Weight", "Basis"], rows))

    if grounding:
        fabricated = grounding.get("citations_fabricated") or []
        lines.append(
            f"**Evidence:** {grounding.get('citations_valid', 0)} of "
            f"{grounding.get('citations_total', 0)} citations verified against the parsed input "
            f"({grounding.get('evidence_ids_available', 0)} referenceable items).\n"
        )
        if fabricated:
            lines.append(
                f"> **Fabricated citations removed:** `{', '.join(fabricated)}` — these IDs do "
                f"not exist in the input. Any claim resting only on them has been dropped.\n"
            )

    if context:
        lines.append(
            f"**Context:** `{context.get('strategy')}` strategy, "
            f"{context.get('estimated_input_tokens', 0):,} input tokens of "
            f"{context.get('available_for_input', 0):,} available. {context.get('reason', '')}\n"
        )

    warnings = list(confidence.warnings if confidence else []) + list(extra_warnings or [])
    if warnings:
        lines.append("**Caveats:**\n")
        lines.append(_bullets(warnings))

    return "\n".join(lines)


# --- log / RCA ------------------------------------------------------------


def render_rca(output, *, ir_stats: dict, confidence, grounding, context, dropped: list[str], warnings: list[str]) -> str:
    parts = ["# Incident Brief\n", "## Summary\n", output.summary + "\n"]

    parts.append("## Severity & Impact\n")
    parts.append(
        _table(
            ["Field", "Value"],
            [
                ["Suspected severity", output.severity],
                ["Rationale", output.severity_rationale],
                ["Affected components", ", ".join(output.affected_components) or "[NONE IDENTIFIED]"],
                ["User-visible effect", output.user_impact],
            ],
        )
    )

    parts.append("## Timeline\n")
    if output.timeline:
        parts.append(
            _table(
                ["Time", "Event", "Evidence"],
                [[row.time, row.event, f"`{row.evidence_id}`"] for row in output.timeline],
            )
        )
    else:
        parts.append("_No timeline rows survived evidence verification._\n")

    if dropped:
        parts.append(
            "> The following timeline rows were removed because they cited evidence that does "
            "not exist in the input:\n"
        )
        parts.append(_bullets(dropped))

    parts.append("## Root Cause Analysis\n")
    parts.append(f"**Most likely cause.** {output.root_cause.statement}{_cite(output.root_cause.evidence_ids)}\n")
    if output.confidence_rationale:
        parts.append(f"**The model's stated reasoning about confidence.** {output.confidence_rationale}\n")

    parts.append("**Supporting evidence.**\n")
    parts.append(
        _bullets([f"{item.statement}{_cite(item.evidence_ids)}" for item in output.supporting_evidence])
    )

    parts.append("**Contradicting evidence.**\n")
    parts.append(
        _bullets(
            [f"{item.statement}{_cite(item.evidence_ids)}" for item in output.contradicting_evidence],
            empty="_None found in the provided excerpt._",
        )
    )

    parts.append("**Alternative hypotheses.**\n")
    if output.alternative_hypotheses:
        for index, hypothesis in enumerate(output.alternative_hypotheses, start=1):
            parts.append(
                f"{index}. {hypothesis.statement}{_cite(hypothesis.supporting_evidence_ids)}\n"
                f"   - To confirm or rule out: {hypothesis.how_to_confirm or '[NOT STATED]'}\n"
            )
    else:
        parts.append("_None offered — treat the single hypothesis above with caution._\n")

    parts.append("## Next Steps\n")
    parts.append("**Immediate (mitigate)**\n")
    parts.append(
        _table(
            ["Action", "Owner role", "Why"],
            [[a.action, a.owner_role, a.rationale] for a in output.immediate_actions],
        )
    )
    parts.append("**Follow-up (diagnose & prevent)**\n")
    parts.append(
        _table(
            ["Action", "Owner role", "Why"],
            [[a.action, a.owner_role, a.rationale] for a in output.followup_actions],
        )
    )

    parts.append("## Evidence Gaps\n")
    parts.append(_bullets(output.evidence_gaps))

    parts.append(render_reliability(confidence, grounding=grounding, context=context, extra_warnings=warnings))

    parts.append("## Input Analysis (deterministic)\n")
    parts.append(
        _table(
            ["Metric", "Value"],
            [[k.replace("_", " "), str(v)] for k, v in ir_stats.items() if v is not None],
        )
    )

    return "\n".join(parts)


# --- postmortem -----------------------------------------------------------


def render_postmortem(output, *, ir_stats: dict, confidence, grounding, context, dropped: list[str], warnings: list[str]) -> str:
    parts = [f"# Postmortem: {output.title}\n", "## Incident Summary\n"]

    parts.append(
        _table(
            ["Field", "Value"],
            [
                ["Date", output.date],
                ["Duration", output.duration],
                ["Severity", output.severity],
                ["Status", output.status],
                ["Incident commander", output.incident_commander_role],
                ["Services affected", ", ".join(output.services_affected) or "[UNKNOWN]"],
            ],
        )
    )

    parts.append("## Impact\n")
    parts.append(
        _table(
            ["Field", "Value"],
            [
                ["Users affected", output.users_affected],
                ["Duration of user impact", output.impact_duration],
                ["Business/technical impact", output.business_impact],
                ["Data integrity", output.data_integrity],
            ],
        )
    )

    parts.append("## Timeline\n")
    if output.timeline:
        parts.append(
            _table(
                ["Time", "Actor (role)", "Event", "Evidence"],
                [[r.time, r.actor_role, r.event, f"`{r.evidence_id}`"] for r in output.timeline],
            )
        )
    else:
        parts.append("_No timeline rows survived evidence verification._\n")

    if dropped:
        parts.append("> Removed rows citing nonexistent evidence:\n")
        parts.append(_bullets(dropped))

    parts.append("## Root Cause\n")
    parts.append(f"**Trigger.** {output.trigger.statement}{_cite(output.trigger.evidence_ids)}\n")
    parts.append(f"**Root cause.** {output.root_cause.statement}{_cite(output.root_cause.evidence_ids)}\n")
    parts.append("**Contributing factors.**\n")
    parts.append(
        _bullets([f"{f.statement}{_cite(f.evidence_ids)}" for f in output.contributing_factors])
    )
    parts.append(
        f"**Why it was not caught earlier.** {output.detection_gap.statement}"
        f"{_cite(output.detection_gap.evidence_ids)}\n"
    )

    parts.append("## Resolution\n")
    parts.append(output.resolution + "\n")
    if not output.resolution_is_permanent_fix:
        parts.append(
            "> **This was a mitigation, not a fix.** The underlying condition remains open and "
            "is tracked in the action items below.\n"
        )

    parts.append("## What Went Well\n")
    parts.append(_bullets(output.what_went_well, empty="_Nothing notable identified in the transcript._"))
    parts.append("## What Went Poorly\n")
    parts.append(_bullets(output.what_went_poorly))
    parts.append("## Where We Got Lucky\n")
    parts.append(_bullets(output.where_we_got_lucky, empty="_Nothing identified._"))

    parts.append("## Action Items\n")
    parts.append(
        _table(
            ["#", "Action", "Type", "Owner role", "Priority", "Rationale"],
            [
                [str(i), a.action, a.type, a.owner_role, a.priority, a.rationale]
                for i, a in enumerate(output.action_items, start=1)
            ],
        )
    )

    parts.append("## Open Questions\n")
    parts.append(_bullets(output.open_questions))

    parts.append(render_reliability(confidence, grounding=grounding, context=context, extra_warnings=warnings))

    parts.append("## Input Analysis (deterministic)\n")
    parts.append(
        _table(["Metric", "Value"], [[k.replace("_", " "), str(v)] for k, v in ir_stats.items() if v is not None])
    )

    return "\n".join(parts)


# --- unit tests -----------------------------------------------------------


def render_unit_tests(output, *, ir_stats: dict, execution, coverage: dict, warnings: list[str], context: dict) -> str:
    parts = ["# Generated Test Suite\n", "## Framework\n"]
    parts.append(f"{output.language} with **{output.framework}**. Run with `{output.run_command}`.\n")

    # Execution result goes first: it is the only part of this document that is
    # a fact rather than a proposal.
    parts.append("## Execution Result\n")
    if execution is None or not execution.ran:
        reason = (execution.skipped_reason if execution else None) or "not attempted for this language"
        parts.append(
            f"> **These tests were not executed** ({reason}). Everything below is unverified "
            f"model output — treat 'runnable' as a claim, not a result.\n"
        )
    elif execution.ok:
        parts.append(
            f"**Verified: {execution.passed} of {execution.collected} tests pass** against the "
            f"source you provided, executed in {execution.duration_seconds:.1f}s.\n"
        )
    else:
        parts.append(
            f"**These tests do not currently pass.** {execution.passed} passed, "
            f"{execution.failed} failed, {execution.errors} errored "
            f"(of {execution.collected} collected).\n"
        )
        if execution.collection_error:
            parts.append(f"Collection problem: `{execution.collection_error}`\n")
        if execution.failure_details:
            parts.append("Failures as reported by pytest:\n")
            parts.append("```text\n" + "\n\n".join(execution.failure_details[:4])[:3000] + "\n```\n")
        parts.append(
            "> A failing generated test is not automatically a bad test. It may have found a "
            "real defect. Read the failures above before changing either side.\n"
        )

    parts.append("## Boundary Coverage\n")
    parts.append(
        f"The parser extracted **{coverage.get('total_boundaries', 0)}** boundary conditions from "
        f"the source. The suite claims to cover **{coverage.get('covered', 0)}** of them "
        f"({coverage.get('ratio', 0):.0%}).\n"
    )
    if coverage.get("uncovered"):
        parts.append("Boundaries with no corresponding test case:\n")
        parts.append(_bullets(list(coverage["uncovered"])))

    parts.append("## Cases\n")
    by_category: dict[str, list] = {}
    for case in output.cases:
        by_category.setdefault(case.category, []).append(case)
    for category in ("happy_path", "edge_case", "error_handling", "side_effects", "concurrency"):
        cases = by_category.get(category, [])
        parts.append(f"**{category.replace('_', ' ').title()}**\n")
        parts.append(
            _bullets(
                [
                    f"`{c.name}` — {c.behaviour}"
                    + (f" (covers `{c.covers_boundary}`)" if c.covers_boundary else "")
                    for c in cases
                ]
            )
        )

    parts.append("## Tests\n")
    parts.append(f"```{output.language.lower()}\n{output.test_code}\n```\n")

    parts.append("## Gaps & Assumptions\n")
    parts.append("**Assumptions made:**\n")
    parts.append(_bullets(output.assumptions))
    parts.append("**Hard to test as written:**\n")
    parts.append(_bullets(output.untestable))

    if warnings:
        parts.append("## Caveats\n")
        parts.append(_bullets(warnings))

    parts.append("## Input Analysis (deterministic)\n")
    parts.append(
        _table(["Metric", "Value"], [[k.replace("_", " "), str(v)] for k, v in ir_stats.items() if v is not None])
    )

    return "\n".join(parts)


# --- API docs -------------------------------------------------------------


def render_api_docs(output, *, ir_stats: dict, openapi_report, warnings: list[str], context: dict) -> str:
    parts = ["# API Reference\n", "## Overview\n", output.overview + "\n"]

    parts.append("## Specification Validation\n")
    if openapi_report is None:
        parts.append("_Not validated._\n")
    elif openapi_report.ok:
        parts.append(
            f"**Valid OpenAPI {openapi_report.version}.** Parsed, schema-validated, and "
            f"cross-checked against the {len(openapi_report.documented_operations)} route(s) "
            f"found in the source — no gaps in either direction.\n"
        )
    else:
        parts.append("**This specification did not fully validate.**\n")
        rows = [
            ["Parses as YAML", "yes" if openapi_report.parsed else "no"],
            ["Valid OpenAPI schema", "yes" if openapi_report.schema_valid else "no"],
            ["Routes missing from spec", ", ".join(openapi_report.missing_routes) or "none"],
            ["Spec operations not in code", ", ".join(openapi_report.extra_routes) or "none"],
        ]
        parts.append(_table(["Check", "Result"], rows))
        if openapi_report.errors:
            parts.append("Validator errors:\n")
            parts.append(_bullets(openapi_report.errors[:6]))

    parts.append("## Endpoints\n")
    parts.append(
        _table(
            ["Method", "Path", "Auth", "Purpose"],
            [[e.method, f"`{e.path}`", e.auth, e.purpose] for e in output.endpoints],
        )
    )

    parts.append("## OpenAPI 3.1\n")
    parts.append(f"```yaml\n{output.openapi_yaml.strip()}\n```\n")

    parts.append("## Reference\n")
    for endpoint in output.endpoints:
        parts.append(f"### {endpoint.method} {endpoint.path}\n")
        parts.append(endpoint.purpose + "\n")
        parts.append("**Request**\n")
        parts.append(
            _table(
                ["Name", "In", "Type", "Required", "Description"],
                [
                    [p.name, p.location, p.type, "yes" if p.required else "no", p.description]
                    for p in endpoint.parameters
                ],
            )
        )
        if endpoint.success_example:
            parts.append("**Response**\n")
            parts.append(f"```json\n{endpoint.success_example.strip()}\n```\n")
        parts.append("**Errors**\n")
        parts.append(
            _table(
                ["Status", "Condition", "Body"],
                [[str(e.status), e.condition, e.body] for e in endpoint.errors],
            )
        )
        if endpoint.curl_example:
            parts.append("**Example**\n")
            parts.append(f"```bash\n{endpoint.curl_example.strip()}\n```\n")

    parts.append("## Notes\n")
    parts.append(_bullets(output.notes))

    if warnings:
        parts.append("## Caveats\n")
        parts.append(_bullets(warnings))

    parts.append("## Input Analysis (deterministic)\n")
    parts.append(
        _table(["Metric", "Value"], [[k.replace("_", " "), str(v)] for k, v in ir_stats.items() if v is not None])
    )

    return "\n".join(parts)
