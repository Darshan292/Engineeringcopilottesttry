"""Evidence verification.

A prompt can ask for citations and get them: models are good at producing
plausible identifiers. `[L47]` looks like evidence whether or not line 47
exists or says anything related.

This module checks every cited ID against the set the parser actually produced.
An ID that is not in the IR is a fabrication, full stop, and is reported as
one. That check is only possible because the IR assigns stable IDs upstream --
it is the concrete payoff of parsing before prompting.

What it cannot check is semantic support: that line L47 genuinely backs the
claim made about it. Doing that properly needs either entailment scoring or a
second model pass, and this module does not pretend otherwise. What it does
give is a hard floor -- no claim can cite something that does not exist -- plus
a lexical overlap signal that catches the weakest cases, reported separately
and never conflated with existence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.ir import LogIR, TranscriptIR

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "was", "were", "are", "be", "been", "that", "this", "it",
    "its", "as", "by", "from", "had", "has", "have", "not", "no", "which",
    "when", "then", "than", "there", "their", "we", "our",
}


@dataclass
class CitationCheck:
    claim: str
    cited: list[str]
    valid: list[str]
    invalid: list[str]
    # Fraction of the claim's content words that appear in the cited evidence.
    lexical_overlap: float = 0.0

    @property
    def grounded(self) -> bool:
        return bool(self.valid) and not self.invalid


@dataclass
class GroundingReport:
    checks: list[CitationCheck] = field(default_factory=list)
    available_ids: int = 0
    # Claims that should carry evidence and do not.
    uncited_claims: list[str] = field(default_factory=list)

    @property
    def total_citations(self) -> int:
        return sum(len(c.cited) for c in self.checks)

    @property
    def invalid_citations(self) -> list[str]:
        seen: list[str] = []
        for check in self.checks:
            for bad in check.invalid:
                if bad not in seen:
                    seen.append(bad)
        return seen

    @property
    def grounding_ratio(self) -> float:
        """Fraction of citations that point at something real."""
        total = self.total_citations
        if not total:
            return 0.0
        valid = sum(len(c.valid) for c in self.checks)
        return valid / total

    @property
    def claim_coverage(self) -> float:
        """Fraction of evidence-bearing claims that carry at least one valid ID."""
        claims = len(self.checks) + len(self.uncited_claims)
        if not claims:
            return 0.0
        return sum(1 for c in self.checks if c.valid) / claims

    @property
    def has_fabrications(self) -> bool:
        return bool(self.invalid_citations)

    def public(self) -> dict:
        return {
            "citations_total": self.total_citations,
            "citations_valid": sum(len(c.valid) for c in self.checks),
            "citations_fabricated": self.invalid_citations,
            "grounding_ratio": round(self.grounding_ratio, 3),
            "claim_coverage": round(self.claim_coverage, 3),
            "uncited_claims": self.uncited_claims[:10],
            "evidence_ids_available": self.available_ids,
            "mean_lexical_overlap": (
                round(sum(c.lexical_overlap for c in self.checks) / len(self.checks), 3)
                if self.checks
                else 0.0
            ),
        }


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_.\-]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _evidence_text(ir, evidence_id: str) -> str:
    if isinstance(ir, LogIR):
        for entry in ir.entries:
            if entry.id == evidence_id:
                return f"{entry.level or ''} {entry.service or ''} {entry.component or ''} {entry.message}"
        for template in ir.templates:
            if template.id == evidence_id:
                return template.template
    elif isinstance(ir, TranscriptIR):
        for utterance in ir.utterances:
            if utterance.id == evidence_id:
                return utterance.text
    return ""


def _overlap(claim: str, ir, valid_ids: list[str]) -> float:
    if not valid_ids:
        return 0.0
    claim_words = _content_words(claim)
    if not claim_words:
        return 0.0
    evidence_words: set[str] = set()
    for evidence_id in valid_ids:
        evidence_words |= _content_words(_evidence_text(ir, evidence_id))
    if not evidence_words:
        return 0.0
    return len(claim_words & evidence_words) / len(claim_words)


def check_citations(claims: list[tuple[str, list[str]]], ir, *, uncited: list[str] | None = None) -> GroundingReport:
    """Verify (claim, cited_ids) pairs against the IR's real ID set."""
    available = ir.evidence_ids()
    checks: list[CitationCheck] = []

    for claim, cited in claims:
        cleaned = [c for c in cited if c]
        valid = [c for c in cleaned if c in available]
        invalid = [c for c in cleaned if c not in available]
        checks.append(
            CitationCheck(
                claim=claim[:300],
                cited=cleaned,
                valid=valid,
                invalid=invalid,
                lexical_overlap=_overlap(claim, ir, valid),
            )
        )

    return GroundingReport(
        checks=checks,
        available_ids=len(available),
        uncited_claims=[c[:300] for c in (uncited or [])],
    )


def claims_from_rca(output) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Pull every evidence-bearing claim out of an RCAOutput."""
    claims: list[tuple[str, list[str]]] = [
        (output.root_cause.statement, output.root_cause.evidence_ids)
    ]
    uncited: list[str] = []

    for item in output.supporting_evidence:
        claims.append((item.statement, item.evidence_ids))
    for item in output.contradicting_evidence:
        claims.append((item.statement, item.evidence_ids))
    for row in output.timeline:
        claims.append((row.event, [row.evidence_id] if row.evidence_id else []))
        if not row.evidence_id:
            uncited.append(f"timeline row: {row.event}")
    for hypothesis in output.alternative_hypotheses:
        combined = hypothesis.supporting_evidence_ids + hypothesis.contradicting_evidence_ids
        claims.append((hypothesis.statement, combined))
        if not combined:
            uncited.append(f"hypothesis: {hypothesis.statement}")

    if not output.root_cause.evidence_ids:
        uncited.append(f"root cause: {output.root_cause.statement}")

    return claims, uncited


def claims_from_postmortem(output) -> tuple[list[tuple[str, list[str]]], list[str]]:
    claims: list[tuple[str, list[str]]] = [
        (output.trigger.statement, output.trigger.evidence_ids),
        (output.root_cause.statement, output.root_cause.evidence_ids),
        (output.detection_gap.statement, output.detection_gap.evidence_ids),
    ]
    uncited: list[str] = []

    for factor in output.contributing_factors:
        claims.append((factor.statement, factor.evidence_ids))
        if not factor.evidence_ids:
            uncited.append(f"contributing factor: {factor.statement}")
    for row in output.timeline:
        claims.append((row.event, [row.evidence_id] if row.evidence_id else []))
        if not row.evidence_id:
            uncited.append(f"timeline row: {row.event}")

    for label, evidenced in (
        ("trigger", output.trigger),
        ("root cause", output.root_cause),
        ("detection gap", output.detection_gap),
    ):
        if not evidenced.evidence_ids:
            uncited.append(f"{label}: {evidenced.statement}")

    return claims, uncited


def drop_fabricated(output, report: GroundingReport) -> list[str]:
    """Remove rows whose only citation was fabricated. Returns what was removed.

    A timeline row that cites a line which does not exist is not a small
    formatting problem -- it is an invented event in a document people will
    treat as a record. Dropping it and saying so is the only honest option.
    """
    fabricated = set(report.invalid_citations)
    if not fabricated:
        return []

    removed: list[str] = []
    timeline = getattr(output, "timeline", None)
    if timeline is None:
        return removed

    kept = []
    for row in timeline:
        if row.evidence_id and row.evidence_id in fabricated:
            removed.append(f"{row.time} {row.event} (cited {row.evidence_id}, which does not exist)")
        else:
            kept.append(row)
    output.timeline = kept
    return removed
