"""Context planning: fit an IR into a token budget without losing the plot.

Four strategies, chosen by measurement rather than by a character threshold:

1. **full** -- everything fits; send it all.
2. **summary** -- the IR renders at reduced detail and fits. For logs this is
   template counts plus representative lines, which is a controlled loss with a
   stated shape, unlike truncation.
3. **map_reduce** -- still too large. The IR is split into ordered windows and
   each is analysed separately, then the partial results are combined.
4. **reject** -- a single indivisible unit exceeds the budget on its own.
   Failing loudly beats silently analysing 3% of a log.

The property that matters for map_reduce is **global context preservation**,
which is where naive chunking produces confidently wrong answers. Split an
incident in half and the first chunk sees a database slowing down with no
errors, while the second sees errors with no cause; both halves get a
reasonable-sounding, wrong explanation.

Every chunk therefore carries the IR's skeleton: the full time window, every
service, total level counts, the earliest anomaly, and the frequency table of
message shapes across the *whole* input. A chunk is explicitly told which
window it covers and that it is looking at a part. The reduce step then gets
the skeleton plus all partial findings, so cross-chunk relationships are
resolved where the whole picture is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .ir import CodeIR, LogIR, RouteIR, TranscriptIR
from .tokens import TokenBudget, estimate_tokens

Strategy = Literal["full", "summary", "map_reduce", "reject"]


@dataclass
class Chunk:
    index: int
    total: int
    body: str
    # Which IR items this chunk contains, so grounding can tell whether a
    # citation belongs to the chunk that produced it.
    item_ids: list[str] = field(default_factory=list)
    label: str = ""
    estimated_tokens: int = 0


@dataclass
class ContextPlan:
    strategy: Strategy
    chunks: list[Chunk]
    skeleton: str
    budget: TokenBudget
    estimated_input_tokens: int
    reason: str
    detail_level: str = "full"

    @property
    def is_multi_pass(self) -> bool:
        return self.strategy == "map_reduce"

    def public(self) -> dict:
        return {
            "strategy": self.strategy,
            "detail_level": self.detail_level,
            "chunks": len(self.chunks),
            "estimated_input_tokens": self.estimated_input_tokens,
            "available_for_input": self.budget.available_for_input,
            "reason": self.reason,
        }


def _items_for(ir) -> tuple[list, str]:
    """The ordered, splittable units of an IR and a label for them."""
    if isinstance(ir, LogIR):
        return list(ir.entries), "lines"
    if isinstance(ir, TranscriptIR):
        return list(ir.utterances), "messages"
    if isinstance(ir, CodeIR):
        return list(ir.all_functions()), "functions"
    if isinstance(ir, RouteIR):
        return list(ir.routes), "routes"
    return [], "items"


def _render_item(item) -> str:
    """One unit rendered for a chunk body."""
    for attr in ("id",):
        if not hasattr(item, attr):
            return str(item)

    if hasattr(item, "raw") and hasattr(item, "message"):  # LogEntry
        bits = [f"[{item.id}]"]
        if item.raw_timestamp:
            bits.append(item.raw_timestamp)
        if item.level:
            bits.append(item.level)
        if item.service:
            bits.append(item.service)
        if item.component:
            bits.append(f"[{item.component}]")
        bits.append(item.message)
        return " ".join(bits)

    if hasattr(item, "speaker_placeholder"):  # Utterance
        stamp = f"{item.raw_timestamp} " if item.raw_timestamp else ""
        tags = f"  <{','.join(item.signals)}>" if item.signals else ""
        return f"[{item.id}] {stamp}{item.speaker_placeholder}: {item.text}{tags}"

    if hasattr(item, "signature"):  # FunctionSpec
        lines = [f"FUNCTION [{item.id}] {item.signature}"]
        if item.comparisons:
            lines.append(f"  boundary conditions: {'; '.join(item.comparisons[:12])}")
        if item.raises:
            lines.append(f"  raises: {', '.join(sorted(set(item.raises)))}")
        if item.external_calls:
            lines.append(f"  external calls: {', '.join(sorted(set(item.external_calls)))}")
        if item.source:
            lines.append(item.source)
        return "\n".join(lines)

    if hasattr(item, "method") and hasattr(item, "path"):  # RouteSpec
        return f"ROUTE [{item.id}] {item.method} {item.path} -> {item.handler}()"

    return str(item)


def _skeleton_for(ir) -> str:
    if isinstance(ir, LogIR):
        return ir.skeleton()
    if isinstance(ir, TranscriptIR):
        return ir.render("skeleton")
    if isinstance(ir, CodeIR):
        names = ", ".join(f"[{f.id}] {f.name}" for f in ir.all_functions())
        return (
            f"MODULE OVERVIEW (global context)\n"
            f"  language: {ir.language} (parser: {ir.parser})\n"
            f"  third-party imports: {', '.join(ir.third_party_imports) or 'none'}\n"
            f"  all functions in this module: {names or 'none'}"
        )
    if isinstance(ir, RouteIR):
        paths = ", ".join(f"[{r.id}] {r.method} {r.path}" for r in ir.routes)
        return (
            f"API OVERVIEW (global context)\n"
            f"  framework: {ir.framework} (parser: {ir.parser})\n"
            f"  all routes in this surface: {paths or 'none'}\n"
            f"  schemas: {', '.join(m.name for m in ir.models) or 'none'}"
        )
    return ""


# A map-reduce plan costs one upstream call per chunk plus one to combine.
# The Groq free tier allows roughly 30 requests/minute and 1,000/day, so a plan
# with hundreds of chunks would exhaust a day's quota on a single request. The
# cap makes that a refusal with a stated remedy instead of a silent bill.
DEFAULT_MAX_CHUNKS = 12


def plan_context(
    ir,
    budget: TokenBudget,
    *,
    detail_hint: str = "full",
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> ContextPlan:
    """Decide how this IR will be presented to the model."""
    skeleton = _skeleton_for(ir)

    full_body = ir.render("full")
    full_tokens = estimate_tokens(full_body)

    if budget.fits(full_tokens):
        return ContextPlan(
            strategy="full",
            chunks=[Chunk(0, 1, full_body, sorted(ir.evidence_ids()), "whole input", full_tokens)],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=f"Whole input fits: {full_tokens:,} of {budget.available_for_input:,} available tokens.",
            detail_level="full",
        )

    summary_body = ir.render("summary")
    summary_tokens = estimate_tokens(summary_body)

    if budget.fits(summary_tokens):
        return ContextPlan(
            strategy="summary",
            chunks=[Chunk(0, 1, summary_body, sorted(ir.evidence_ids()), "compressed input", summary_tokens)],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=summary_tokens,
            reason=(
                f"Full input needs {full_tokens:,} tokens, over the {budget.available_for_input:,} "
                f"available. Compressed to {summary_tokens:,} by grouping repeated message shapes; "
                f"counts and first/last occurrences are preserved."
            ),
            detail_level="summary",
        )

    # Map-reduce. Each chunk pays for the skeleton, so that comes off the top.
    items, label = _items_for(ir)
    if not items:
        return ContextPlan(
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason="Input exceeds the context budget and has no splittable structure.",
        )

    skeleton_tokens = estimate_tokens(skeleton)
    # Leave room for the per-chunk header and the model's partial answer.
    per_chunk_budget = budget.available_for_input - skeleton_tokens - 256
    if per_chunk_budget < 512:
        return ContextPlan(
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"The global context alone needs {skeleton_tokens:,} tokens of the "
                f"{budget.available_for_input:,} available, leaving no room for content. "
                f"Use a model with a larger context window."
            ),
        )

    chunks: list[Chunk] = []
    current: list[str] = []
    current_ids: list[str] = []
    current_tokens = 0
    oversized: list[str] = []

    for item in items:
        rendered = _render_item(item)
        cost = estimate_tokens(rendered)

        if cost > per_chunk_budget:
            # A single unit that cannot fit: record it rather than silently
            # dropping it, and keep going so the report is complete.
            oversized.append(getattr(item, "id", "?"))
            continue

        if current and current_tokens + cost > per_chunk_budget:
            chunks.append(Chunk(len(chunks), 0, "\n".join(current), list(current_ids), "", current_tokens))
            current, current_ids, current_tokens = [], [], 0

        current.append(rendered)
        current_ids.append(getattr(item, "id", ""))
        current_tokens += cost

    if current:
        chunks.append(Chunk(len(chunks), 0, "\n".join(current), list(current_ids), "", current_tokens))

    if not chunks:
        return ContextPlan(
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"Every individual {label[:-1]} exceeds the per-chunk budget of "
                f"{per_chunk_budget:,} tokens. Nothing could be sent."
            ),
        )

    if len(chunks) > max_chunks:
        needed = len(chunks) + 1
        return ContextPlan(
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"This input would require {len(chunks)} chunks ({needed} upstream calls) to "
                f"analyse with a {budget.context_window:,}-token context window, over the "
                f"limit of {max_chunks}. That would consume a large share of a free-tier "
                f"daily quota for one request. Options: use a model with a larger context "
                f"window, narrow the excerpt to the incident window, or raise max_chunks "
                f"deliberately."
            ),
        )

    # Finalize: stamp totals and prepend the global context to every chunk.
    total = len(chunks)
    for chunk in chunks:
        chunk.total = total
        first, last = (chunk.item_ids[0], chunk.item_ids[-1]) if chunk.item_ids else ("?", "?")
        chunk.label = f"part {chunk.index + 1} of {total} ({first}..{last})"
        chunk.body = (
            f"{skeleton}\n\n"
            f"YOU ARE LOOKING AT PART {chunk.index + 1} OF {total} of the input "
            f"({len(chunk.item_ids)} {label}, {first} through {last}).\n"
            f"The overview above describes the ENTIRE input, not just this part. Do not "
            f"conclude that something is absent because it is not in this part -- say so "
            f"explicitly and let the combining step decide.\n\n"
            f"{chunk.body}"
        )
        chunk.estimated_tokens = estimate_tokens(chunk.body)

    reason = (
        f"Input needs {full_tokens:,} tokens even compressed ({summary_tokens:,}), over the "
        f"{budget.available_for_input:,} available. Split into {total} ordered parts; each "
        f"carries the full global overview so cross-part relationships survive."
    )
    if oversized:
        reason += f" {len(oversized)} oversized {label} could not be included: {oversized[:5]}."

    return ContextPlan(
        strategy="map_reduce",
        chunks=chunks,
        skeleton=skeleton,
        budget=budget,
        estimated_input_tokens=sum(c.estimated_tokens for c in chunks),
        reason=reason,
        detail_level="chunked",
    )
