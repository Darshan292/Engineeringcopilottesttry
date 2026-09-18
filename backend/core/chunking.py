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

from ..config import _env_int
from .ir import CodeIR, LogIR, RouteIR, TranscriptIR
from .tokens import TokenBudget, estimate_tokens

Strategy = Literal["full", "summary", "selected", "map_reduce", "reject"]


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
    # What executing this plan costs, so the caller can decide before paying.
    estimated_calls: int = 1
    estimated_seconds: int = 0
    selection: dict = field(default_factory=dict)

    # What the caller may spend on one call, after the context window and the
    # token allowance are both taken into account.
    effective_per_call: int = 0

    # How many rounds the combine takes. One means a single combine call; more
    # means partial answers are merged, then the merges are merged.
    reduce_levels: int = 1

    # What one combine call may carry, and how large a partial answer may be.
    # Both are decided here rather than in the runtime, because the promised
    # call count is only true if the run uses the same numbers to batch with.
    reduce_ceiling: int = 0
    map_output_tokens: int = 0

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
            "estimated_model_calls": self.estimated_calls,
            "reduce_levels": self.reduce_levels,
            "estimated_seconds": self.estimated_seconds,
            "binding_constraint": self.budget.public()["binding_constraint"],
            "selection": self.selection,
            "reason": self.reason,
        }

    def describe(self) -> str:
        """A plain-language account of the plan, for the user and the model.

        Both need it. The user is entitled to know their input was reduced and
        by what rule; the model reasons better when it knows it is looking at a
        selection rather than the whole thing, and says so in its conclusions.
        """
        lines = [f"PROCESSING PLAN: {self.strategy}"]

        # Report the budget that actually applied. Naming the context window
        # when a tighter token allowance was the real constraint would tell
        # both the reader and the model something false about why the input
        # was reduced.
        window_budget = self.budget.available_for_input
        effective = self.effective_per_call or window_budget
        if effective < window_budget:
            constraint = (
                f"per-minute token allowance; the model's context window would have "
                f"allowed {window_budget:,}"
            )
        else:
            constraint = self.budget.public()["binding_constraint"]
        lines.append(f"  Budget: {effective:,} input tokens per call (limited by the {constraint}).")

        if self.strategy == "full":
            lines.append("  The entire input fits in one call. Nothing was omitted.")
        elif self.strategy == "summary":
            lines.append(
                "  Repeated messages were grouped into patterns with counts and first/last "
                "occurrence. Every line is accounted for; not every line is reproduced."
            )
        elif self.strategy == "selected":
            selected = self.selection.get("lines_selected", 0)
            considered = self.selection.get("lines_considered", 0)
            # This is the one lossy strategy, so it says so first. A model told
            # it is reading a subset qualifies its conclusions; a model that
            # believes it read everything states them flatly and is wrong.
            lines.append(
                "  DEGRADED: reading every line was out of budget, so this is a subset. "
                "Full coverage was ruled out before selection, not preferred against it."
            )
            lines.append(
                f"  {selected:,} of {considered:,} lines were chosen by a deterministic "
                f"relevance score -- severity, how rare the message pattern is, first and last "
                f"occurrence of each pattern, level transitions, closeness to the earliest "
                f"anomaly, and numeric outliers. No model was involved in choosing."
            )
            lines.append(
                "  The lines you cannot see are counted in the pattern table above, so their "
                "existence is known even though their text is absent. Say so if a conclusion "
                "would need them."
            )
        elif self.strategy == "map_reduce":
            lines.append(
                f"  Split into {len(self.chunks)} ordered parts, analysed separately and then "
                f"combined over {self.reduce_levels} round(s) ({self.estimated_calls} model "
                f"calls in total, roughly {self.estimated_seconds}s at the current token rate). "
                f"Every part carries the global overview so relationships spanning parts survive."
            )
            lines.append(
                "  Nothing was dropped to make this fit: every line is read by exactly one "
                "part. What no single call sees is the relationship between parts, which the "
                "combining step resolves from the partial findings plus the overview."
            )
        return "\n".join(lines)


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


# Every chunk pays for the skeleton, so an unbounded enumeration of a large
# input's items makes the global context cost more than the content -- which
# defeats chunking entirely for exactly the inputs that need it.
_SKELETON_LIST_LIMIT = 40


def _bounded_list(items: list[str], limit: int = _SKELETON_LIST_LIMIT) -> str:
    if not items:
        return "none"
    if len(items) <= limit:
        return ", ".join(items)
    head = ", ".join(items[: limit - 5])
    tail = ", ".join(items[-5:])
    return f"{head}, ... ({len(items) - limit} more omitted from this list) ..., {tail}"


def _selected_ids(ir, body: str) -> list[str]:
    """IDs actually present in a rendered selection, in order."""
    return [item.id for item in _items_for(ir)[0] if f"[{item.id}]" in body]


def _skeleton_for(ir) -> str:
    if isinstance(ir, LogIR):
        return ir.skeleton()
    if isinstance(ir, TranscriptIR):
        return ir.render("skeleton")
    if isinstance(ir, CodeIR):
        return (
            f"MODULE OVERVIEW (global context)\n"
            f"  language: {ir.language} (parser: {ir.parser})\n"
            f"  third-party imports: {', '.join(ir.third_party_imports) or 'none'}\n"
            f"  functions in this module: "
            f"{_bounded_list([f'[{f.id}] {f.name}' for f in ir.all_functions()])}"
        )
    if isinstance(ir, RouteIR):
        return (
            f"API OVERVIEW (global context)\n"
            f"  framework: {ir.framework} (parser: {ir.parser})\n"
            f"  routes in this surface: "
            f"{_bounded_list([f'[{r.id}] {r.method} {r.path}' for r in ir.routes])}\n"
            f"  schemas: {_bounded_list([m.name for m in ir.models])}"
        )
    return ""


# A map-reduce plan costs one upstream call per chunk plus one to combine.
# The Groq free tier allows roughly 30 requests/minute and 1,000/day, so a plan
# with hundreds of chunks would exhaust a day's quota on a single request. The
# cap makes that a refusal with a stated remedy instead of a silent bill.
DEFAULT_MAX_CHUNKS = 40


def _pacing_seconds(calls: int, tokens_per_call: int, allowance_per_minute: int) -> int:
    """How long `calls` will take, given a tokens-per-minute ceiling."""
    if not allowance_per_minute or allowance_per_minute <= 0:
        return 0
    calls_per_minute = max(1.0, allowance_per_minute / max(tokens_per_call, 1))
    return int((calls / calls_per_minute) * 60)


# How long a single request may spend pacing itself against a token allowance
# before the planner gives up on reading every line and degrades to a scored
# subset instead.
#
# The default is generous on purpose: reading the whole input is worth waiting
# for, and a paced multi-call plan on a free-tier allowance is minutes, not
# seconds. It is not unbounded, because the request is synchronous -- a browser,
# a reverse proxy, or uvicorn's own timeout will eventually close the
# connection, and work that nobody receives is worse than a degraded answer
# that arrives. Raise it together with the server and proxy read timeouts.
MAX_PLAN_SECONDS = _env_int("MAX_PLAN_SECONDS", 600)



# The map output reserve, mirrored from `config.MAP_OUTPUT_TOKENS`. Each partial
# answer costs roughly this much when it is fed back into a combine call, which
# is what decides how many partials one combine can carry.
# A partial answer too small to hold a finding is not worth making. Below this
# the combine cannot be given room to converge and map-reduce is not viable at
# the budget on offer.
MIN_PARTIAL_TOKENS = 400


def _map_output_budget(reduce_ceiling: int, preferred: int) -> int:
    """How large a partial answer may be, so the combine can actually converge.

    A combine batch must hold at least two partials, or a level of the tree
    merges one answer into one answer and the tree never shrinks. That is not a
    theoretical concern: at a 1,709-token combine ceiling with partials allowed
    to reach 1,200, every batch holds exactly one, and a twelve-part analysis
    spends twelve calls before failing with "did not converge". Capping the
    partial at half the ceiling makes convergence a property of the plan rather
    than a lucky consequence of the model being terse.
    """
    return max(0, min(preferred, reduce_ceiling // 2))


# The runtime stops after this many rounds of merging rather than looping on a
# pathological input. A deeper tree also means every finding has been rewritten
# more times, so depth is a fidelity cost, not only a token cost.
MAX_REDUCE_LEVELS = 4


def _reduce_tree_cost(
    chunk_count: int, reduce_ceiling: int, partial_tokens: int
) -> tuple[int, int, bool]:
    """Combine calls, levels, and whether the tree converges inside the cap.

    The combine is a tree, not a single call: at a small per-call budget a dozen
    partial answers do not fit together, so they are merged in batches and the
    merges are merged again. Estimating it as one call understated a thirteen
    part analysis by thirteen calls -- the plan promised fourteen and the run
    made twenty-seven, which is exactly the kind of preview that is worse than
    no preview.

    Convergence is reported rather than assumed. With two partials per batch a
    tree can only fold 2**4 = 16 parts inside the level cap; beyond that the
    runtime gives up, and it gives up *after* paying for every map call. The
    planner needs to know that before it starts.

    Mirrors the loop in `pipeline.tools`; the two must agree, and
    `test_the_promised_call_count_holds_for_a_split_input` fails if they drift.
    """
    per_batch = max(1, reduce_ceiling // max(1, partial_tokens))
    pending = max(1, chunk_count)
    calls = 0
    levels = 0
    while True:
        batches = -(-pending // per_batch)  # ceil
        levels += 1
        if batches <= 1:
            return calls + 1, levels, True
        calls += batches
        pending = batches
        if levels > MAX_REDUCE_LEVELS:
            return calls, levels, False


def plan_context(
    ir,
    budget: TokenBudget,
    *,
    detail_hint: str = "full",
    max_chunks: int | None = None,
    max_plan_seconds: int | None = None,
    per_call_ceiling: int | None = None,
) -> ContextPlan:
    """Decide how this IR will be presented to the model.

    `per_call_ceiling` is what one call may spend from the account's token
    allowance, which on a rate-limited plan is usually far smaller than the
    model's context window. Planning against the window alone produces chunks
    that fit the model and not the quota: the plan looks valid and every call
    is refused. The effective budget is the smaller of the two.
    """
    # Resolved at call time, not captured as a default, so the module-level
    # values stay overridable by configuration and by tests.
    max_chunks = DEFAULT_MAX_CHUNKS if max_chunks is None else max_chunks
    max_plan_seconds = MAX_PLAN_SECONDS if max_plan_seconds is None else max_plan_seconds
    skeleton = _skeleton_for(ir)

    # The binding limit is whichever is smaller: what the context window
    # allows, or what a single call may spend from the token allowance.
    effective_budget = budget.available_for_input
    if per_call_ceiling is not None:
        effective_budget = min(effective_budget, max(0, per_call_ceiling))

    def fits(tokens: int) -> bool:
        return tokens <= effective_budget

    full_body = ir.render("full")
    full_tokens = estimate_tokens(full_body)

    if fits(full_tokens):
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="full",
            chunks=[Chunk(0, 1, full_body, sorted(ir.evidence_ids()), "whole input", full_tokens)],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"Whole input fits in one call: {full_tokens:,} of {effective_budget:,} "
                f"usable tokens."
            ),
            detail_level="full",
        )

    summary_body = ir.render("summary")
    summary_tokens = estimate_tokens(summary_body)

    if fits(summary_tokens):
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="summary",
            chunks=[Chunk(0, 1, summary_body, sorted(ir.evidence_ids()), "compressed input", summary_tokens)],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=summary_tokens,
            reason=(
                f"Full input needs {full_tokens:,} tokens, over the {effective_budget:,} usable "
                f"per call. Compressed to {summary_tokens:,} by grouping repeated message shapes; "
                f"counts and first/last occurrences are preserved."
            ),
            detail_level="summary",
        )

    # Relevance selection is DELIBERATELY NOT TRIED HERE.
    #
    # It is the only lossy strategy in the ladder: it keeps the highest-scoring
    # lines and drops the text of the rest. Map-reduce costs N+1 calls and real
    # waiting, but every line is read. Preferring the cheap strategy meant a log
    # that could have been fully analysed in six paced calls was instead thinned
    # to forty lines and answered in one -- faster, cheaper, and a worse answer.
    #
    # So selection is the fallback below, taken only when full coverage is
    # genuinely out of reach, and the caller is told what it cost them.
    def selected_fallback(reason_prefix: str) -> ContextPlan | None:
        """A lossy one-call plan, for when every line cannot be read."""
        if not hasattr(ir, "render_selected"):
            return None
        # Select against the budget that actually applies, not the context
        # window. Passing the window here made the selector keep every line
        # and then report "900 of 900 selected" while the call still could not
        # fit inside the token allowance.
        selected_body, selection = ir.render_selected(effective_budget)
        selected_tokens = estimate_tokens(selected_body)
        if not fits(selected_tokens) or selection.get("lines_selected", 0) <= 0:
            return None
        kept_ids = _selected_ids(ir, selected_body)
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="selected",
            chunks=[Chunk(0, 1, selected_body, kept_ids, "relevance-selected", selected_tokens)],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=selected_tokens,
            reason=(
                f"{reason_prefix} {selection['lines_selected']:,} of "
                f"{selection['lines_considered']:,} lines were selected by relevance score; "
                f"the rest remain counted in the pattern table. One model call."
            ),
            detail_level="selected",
            estimated_calls=1,
            estimated_seconds=0,
            selection=selection,
        )

    # Map-reduce. Each chunk pays for the skeleton, so that comes off the top.
    items, label = _items_for(ir)
    if not items:
        degraded = selected_fallback(
            f"Input needs {full_tokens:,} tokens even compressed ({summary_tokens:,}), over "
            f"the {effective_budget:,} usable per call, and has no splittable structure to "
            f"chunk, so every line could not be read."
        )
        if degraded:
            return degraded
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason="Input exceeds the context budget and has no splittable structure.",
        )

    skeleton_tokens = estimate_tokens(skeleton)
    # Leave room for the per-chunk header and the model's partial answer.
    per_chunk_budget = effective_budget - skeleton_tokens - 256
    if per_chunk_budget < 512:
        degraded = selected_fallback(
            f"The global overview alone needs {skeleton_tokens:,} tokens of the "
            f"{effective_budget:,} usable per call, leaving too little room to split the "
            f"input into parts, so every line could not be read."
        )
        if degraded:
            return degraded
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"The global overview alone needs {skeleton_tokens:,} tokens of the "
                f"{effective_budget:,} usable per call, leaving no room for content. "
                f"This is a per-call token allowance limit, not the model's context window "
                f"({budget.context_window:,}). Raise TOKEN_LIMIT_PER_MINUTE if your account "
                f"permits, or use a model with a higher allowance."
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
        degraded = selected_fallback(
            f"Every individual {label[:-1]} exceeds the per-chunk budget of "
            f"{per_chunk_budget:,} tokens, so the input could not be split and every line "
            f"could not be read."
        )
        if degraded:
            return degraded
        return ContextPlan(
            effective_per_call=effective_budget,
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

    per_call = int(sum(estimate_tokens(c.body) for c in chunks) / max(1, len(chunks))) + budget.reserved_for_output

    # Combining is a tree whenever the partials do not fit one call together,
    # so the cost is the map calls plus every level of the reduce -- not the
    # map calls plus one.
    from ..config import MAP_OUTPUT_TOKENS

    reduce_ceiling = max(1_000, effective_budget - skeleton_tokens - 400)
    map_output_tokens = _map_output_budget(reduce_ceiling, MAP_OUTPUT_TOKENS)

    if len(chunks) > 1 and map_output_tokens < MIN_PARTIAL_TOKENS:
        # Splitting would work; combining the results would not. Say that,
        # rather than spending every map call and failing at the last step.
        degraded = selected_fallback(
            f"Splitting this input would produce {len(chunks)} parts, but one combine call can "
            f"only carry {reduce_ceiling:,} tokens -- too little to merge two partial answers, "
            f"so the combining step could never converge and full coverage is not reachable at "
            f"this budget."
        )
        if degraded:
            degraded.reason += (
                " Raise TOKEN_LIMIT_PER_MINUTE, or use a model with a larger context window, "
                "to read every line instead."
            )
            return degraded
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"This input needs {len(chunks)} parts, but one combine call can only carry "
                f"{reduce_ceiling:,} tokens -- not enough to merge two partial answers, so the "
                f"results could never be combined. Splitting would spend {len(chunks)} model "
                f"calls and then fail."
            ),
            estimated_calls=len(chunks) + 1,
        )

    reduce_calls, reduce_levels, converges = _reduce_tree_cost(
        len(chunks), reduce_ceiling, max(1, map_output_tokens)
    )
    calls = len(chunks) + reduce_calls

    if not converges:
        per_batch = max(1, reduce_ceiling // max(1, map_output_tokens))
        foldable = per_batch ** MAX_REDUCE_LEVELS
        detail = (
            f"{len(chunks)} parts cannot be combined: one combine call holds {per_batch} "
            f"partial answers at this budget, which folds at most {foldable:,} parts within "
            f"{MAX_REDUCE_LEVELS} rounds of merging."
        )
        # Refusing here costs nothing. Discovering it at the last combine costs
        # every map call that came before it.
        degraded = selected_fallback(f"{detail} Full coverage was therefore not attempted.")
        if degraded:
            degraded.reason += (
                " Narrow the excerpt, raise TOKEN_LIMIT_PER_MINUTE, or use a model with a "
                "larger context window to read every line instead."
            )
            return degraded
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"{detail} Splitting would spend {len(chunks)} model calls and then fail at the "
                f"combining step, so it is refused before any are spent."
            ),
            estimated_calls=calls,
        )
    projected_seconds = _pacing_seconds(calls, per_call, budget.token_allowance_per_minute)

    # Both limits mean the same thing to a caller -- this plan costs more than
    # it should -- so they produce one message with the full arithmetic rather
    # than two differently worded refusals.
    too_many = len(chunks) > max_chunks
    too_slow = projected_seconds > max_plan_seconds

    if too_many or too_slow:
        limits = []
        if too_many:
            limits.append(f"{len(chunks)} parts exceeds the limit of {max_chunks}")
        if too_slow:
            limits.append(
                f"{projected_seconds // 60}m{projected_seconds % 60:02d}s exceeds the "
                f"{max_plan_seconds}s limit"
            )
        rate = (
            f" at {budget.token_allowance_per_minute:,} tokens/minute"
            if budget.token_allowance_per_minute
            else ""
        )
        # Reading every line is out of reach here, but a scored subset is not.
        # A degraded answer that says what it is beats a refusal, and the
        # caller is shown the arithmetic that ruled full coverage out so they
        # can raise the limit and get it.
        degraded = selected_fallback(
            f"Reading every line would need {calls} model calls of about {per_call:,} tokens "
            f"each, roughly {projected_seconds // 60}m{projected_seconds % 60:02d}s{rate} "
            f"({'; '.join(limits)}), so full coverage was not attempted."
        )
        if degraded:
            degraded.reason += (
                " To read every line instead, narrow the excerpt, raise "
                "TOKEN_LIMIT_PER_MINUTE if your account allows more, or raise MAX_PLAN_SECONDS."
            )
            return degraded
        return ContextPlan(
            effective_per_call=effective_budget,
            strategy="reject",
            chunks=[],
            skeleton=skeleton,
            budget=budget,
            estimated_input_tokens=full_tokens,
            reason=(
                f"This input needs {calls} model calls of about {per_call:,} tokens each, "
                f"roughly {projected_seconds // 60}m{projected_seconds % 60:02d}s{rate} "
                f"({'; '.join(limits)}). Options: narrow the excerpt to the window you care "
                f"about, raise TOKEN_LIMIT_PER_MINUTE if your account allows more, or use a "
                f"model with a larger context window."
            ),
            estimated_calls=calls,
            estimated_seconds=projected_seconds,
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
        f"{effective_budget:,} usable per call. Split into {total} ordered parts analysed "
        f"one at a time, then combined; each part carries the full global overview so "
        f"cross-part relationships survive."
    )
    if oversized:
        reason += f" {len(oversized)} oversized {label} could not be included: {oversized[:5]}."

    return ContextPlan(
        effective_per_call=effective_budget,
        strategy="map_reduce",
        chunks=chunks,
        skeleton=skeleton,
        budget=budget,
        estimated_input_tokens=sum(c.estimated_tokens for c in chunks),
        reason=reason,
        detail_level="chunked",
        estimated_calls=calls,
        estimated_seconds=projected_seconds,
        reduce_levels=reduce_levels,
        reduce_ceiling=reduce_ceiling,
        map_output_tokens=map_output_tokens,
    )
