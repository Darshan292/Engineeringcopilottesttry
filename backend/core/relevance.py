"""Deterministic relevance selection.

When the budget cannot hold the whole input, something has to choose what goes.
The lazy answers are "the first N lines" (truncation, which discards the
resolution) and "ask the model" (which requires sending it everything first,
defeating the purpose).

This scores every line in code, using signals that correlate with diagnostic
value, and keeps the highest-scoring set that fits. No model involvement, no
randomness, identical output for identical input.

The scoring signals, and why each one earns its weight:

- **Severity.** An ERROR carries more diagnostic weight than an INFO. Obvious,
  and not sufficient on its own: a log that is 90% errors gets no signal from
  severity alone, which is why it is one term among several.
- **Rarity.** A line from a template that appears twice says more than the
  40,000th instance of a template that appears 40,000 times. This is the
  information-theoretic term -- surprise is proportional to -log(frequency) --
  and it is what stops a flood of identical timeouts from crowding out the one
  line that explains them.
- **Novelty.** The first and last occurrence of any template are boundaries:
  when a behaviour started and when it stopped. Those two lines usually carry
  the whole story of that behaviour; the ones between are repetition.
- **Transitions.** A line whose level differs from its predecessor marks a
  change in system state. Onsets and recoveries live here.
- **Proximity.** Lines near the earliest anomaly are disproportionately likely
  to explain it, so distance from that point decays the score.
- **Numeric outliers.** A duration far from its template's median is the line
  where something actually went wrong, as distinct from the same message at a
  normal value.
- **Continuity.** Neighbours of a selected line are boosted slightly, because
  an isolated line without its context is hard to reason about.

The result is auditable: every kept line carries the reasons it was kept, and
the response reports how many lines were considered and how many survived.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .ir import LogEntry, LogIR

_SEVERITY_WEIGHT = {
    "FATAL": 1.0, "CRITICAL": 1.0, "PANIC": 1.0, "SEVERE": 0.95,
    "ERROR": 0.9, "WARN": 0.6, "NOTICE": 0.35, "INFO": 0.2, "DEBUG": 0.1, "TRACE": 0.05,
}


@dataclass
class ScoredLine:
    entry: LogEntry
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class SelectionReport:
    kept: list[LogEntry]
    considered: int
    dropped: int
    reasons: Counter = field(default_factory=Counter)
    coverage_note: str = ""

    def public(self) -> dict:
        return {
            "lines_considered": self.considered,
            "lines_selected": len(self.kept),
            "lines_dropped": self.dropped,
            "selection_ratio": round(len(self.kept) / self.considered, 4) if self.considered else 1.0,
            "top_reasons": dict(self.reasons.most_common(6)),
            "note": self.coverage_note,
        }


def _numeric_values(message: str) -> list[float]:
    """Numbers in a message, for outlier detection within a template."""
    out: list[float] = []
    token = ""
    for char in message:
        if char.isdigit() or (char == "." and token):
            token += char
        else:
            if token:
                try:
                    out.append(float(token))
                except ValueError:
                    pass
                token = ""
    if token:
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def score_entries(ir: LogIR) -> list[ScoredLine]:
    """Score every parsed line by diagnostic value. Deterministic."""
    entries = ir.entries
    if not entries:
        return []

    total = len(entries)
    template_counts = Counter(e.template_id for e in entries if e.template_id)

    # First and last occurrence of each template are the behaviour's boundaries.
    first_of_template: dict[str, str] = {}
    last_of_template: dict[str, str] = {}
    for entry in entries:
        if entry.template_id:
            first_of_template.setdefault(entry.template_id, entry.id)
            last_of_template[entry.template_id] = entry.id

    # Median numeric value per template, for outlier detection.
    per_template_values: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        if entry.template_id:
            per_template_values[entry.template_id].extend(_numeric_values(entry.message)[:4])
    medians = {
        tid: statistics.median(values)
        for tid, values in per_template_values.items()
        if len(values) >= 4
    }

    anomaly_index = next((i for i, e in enumerate(entries) if e.id == ir.first_anomaly_id), None)

    scored: list[ScoredLine] = []
    previous_level: str | None = None

    for index, entry in enumerate(entries):
        reasons: list[str] = []
        score = 0.0

        level = (entry.level or "INFO").upper()
        severity = _SEVERITY_WEIGHT.get(level, 0.25)
        score += severity * 2.0
        if severity >= 0.9:
            reasons.append("severe")

        # Rarity: -log(p) of this template, normalized.
        if entry.template_id and template_counts:
            probability = template_counts[entry.template_id] / total
            rarity = -math.log(max(probability, 1e-9)) / math.log(max(total, 2))
            score += rarity * 2.0
            if rarity > 0.5:
                reasons.append("rare pattern")

        if entry.template_id and first_of_template.get(entry.template_id) == entry.id:
            score += 1.5
            reasons.append("first of its kind")
        if entry.template_id and last_of_template.get(entry.template_id) == entry.id:
            score += 1.0
            reasons.append("last of its kind")

        if previous_level is not None and level != previous_level:
            score += 1.2
            reasons.append("level transition")
        previous_level = level

        if anomaly_index is not None:
            distance = abs(index - anomaly_index)
            score += 1.5 * math.exp(-distance / max(20, total * 0.02))
            if distance <= 5:
                reasons.append("near earliest anomaly")

        median = medians.get(entry.template_id or "")
        if median:
            values = _numeric_values(entry.message)[:4]
            if values and median > 0:
                deviation = max(abs(v - median) / median for v in values)
                if deviation > 1.0:
                    score += min(1.5, deviation * 0.4)
                    reasons.append("numeric outlier")

        # The very first and very last lines anchor the window.
        if index == 0 or index == total - 1:
            score += 1.0
            reasons.append("window boundary")

        scored.append(ScoredLine(entry=entry, score=score, reasons=reasons))

    return scored


def select_within_budget(
    ir: LogIR,
    budget_tokens: int,
    *,
    estimate: callable,
    render: callable,
    neighbour_bonus: bool = True,
) -> SelectionReport:
    """Keep the highest-value lines that fit, in original order.

    `estimate` costs a rendered line; `render` turns an entry into its prompt
    form. Both are injected so this module stays independent of the renderer.
    """
    scored = score_entries(ir)
    if not scored:
        return SelectionReport(kept=[], considered=0, dropped=0)

    ranked = sorted(scored, key=lambda s: (-s.score, s.entry.line_no))

    chosen: dict[str, ScoredLine] = {}
    spent = 0
    reasons: Counter = Counter()

    for candidate in ranked:
        cost = estimate(render(candidate.entry))
        if spent + cost > budget_tokens:
            continue
        chosen[candidate.entry.id] = candidate
        spent += cost
        for reason in candidate.reasons:
            reasons[reason] += 1

    # A selected line is easier to reason about with its immediate neighbours.
    if neighbour_bonus and spent < budget_tokens * 0.9:
        by_index = {s.entry.id: i for i, s in enumerate(scored)}
        for entry_id in list(chosen):
            index = by_index.get(entry_id)
            if index is None:
                continue
            for offset in (-1, 1):
                neighbour_index = index + offset
                if 0 <= neighbour_index < len(scored):
                    neighbour = scored[neighbour_index]
                    if neighbour.entry.id in chosen:
                        continue
                    cost = estimate(render(neighbour.entry))
                    if spent + cost > budget_tokens:
                        continue
                    chosen[neighbour.entry.id] = neighbour
                    spent += cost
                    reasons["context for a selected line"] += 1

    kept = [s.entry for s in sorted(chosen.values(), key=lambda s: s.entry.line_no)]
    dropped = len(scored) - len(kept)

    note = (
        f"{len(kept)} of {len(scored)} lines were selected by a deterministic relevance "
        f"score (severity, pattern rarity, first/last occurrence, level transitions, "
        f"proximity to the earliest anomaly, numeric outliers). The dropped lines are "
        f"accounted for in the pattern-frequency table, so their existence and counts are "
        f"still visible even though their text is not."
    ) if dropped else ""

    return SelectionReport(
        kept=kept,
        considered=len(scored),
        dropped=dropped,
        reasons=reasons,
        coverage_note=note,
    )
