"""Intermediate representations.

This module is the architectural centre of the system. Without it the LLM is
handed a raw string and asked to do three jobs at once -- parse it, reason
about it, and format the answer. Everything that can be decided mechanically
is decided here instead, by real parsers, before a prompt is ever built.

Three properties matter:

**Everything addressable.** Every log line, utterance, function and route
carries a stable ID. The model cites those IDs, and `validation/grounding.py`
checks each citation against the IR. A claim whose evidence does not exist is
detectable rather than plausible-sounding.

**Deterministic facts stay deterministic.** Which services appear, when the
first error occurred, how many branches a function has, which status codes a
handler can raise -- these are extracted, not inferred. The model is left with
the part that genuinely needs judgement.

**Compressible with a known loss.** An IR can be rendered at several levels of
detail, so a 200k-line log fits a context window by dropping detail in a
controlled way instead of being truncated at an arbitrary character count.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# --- shared ---------------------------------------------------------------


class IRBase(BaseModel):
    """Common surface every IR exposes to the pipeline."""

    kind: str

    def evidence_ids(self) -> set[str]:
        """Every ID the model is allowed to cite."""
        raise NotImplementedError

    def render(self, detail: Literal["full", "summary", "skeleton"] = "full") -> str:
        """Render for prompt inclusion at a chosen level of detail."""
        raise NotImplementedError

    def stats(self) -> dict:
        """Facts worth returning to the caller and asserting on in tests."""
        raise NotImplementedError


# --- logs -----------------------------------------------------------------


class LogEntry(BaseModel):
    id: str
    line_no: int
    timestamp: datetime | None = None
    raw_timestamp: str | None = None
    level: str | None = None
    service: str | None = None
    component: str | None = None
    message: str
    raw: str
    template_id: str | None = None


class LogTemplate(BaseModel):
    """A cluster of log lines sharing a structure, with variables masked.

    This is where large logs become tractable: 40,000 lines of the same
    connection-timeout message collapse to one template with a count, and the
    model sees the shape plus the frequency instead of 40,000 near-duplicates.
    """

    id: str
    template: str
    count: int
    level: str | None = None
    services: list[str] = Field(default_factory=list)
    first_line: int
    last_line: int
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    example_ids: list[str] = Field(default_factory=list)


class LogIR(IRBase):
    kind: Literal["logs"] = "logs"
    entries: list[LogEntry] = Field(default_factory=list)
    templates: list[LogTemplate] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    level_counts: dict[str, int] = Field(default_factory=dict)
    start_time: datetime | None = None
    end_time: datetime | None = None
    first_error_id: str | None = None
    first_anomaly_id: str | None = None
    total_lines: int = 0
    parsed_lines: int = 0
    truncated: bool = False

    def evidence_ids(self) -> set[str]:
        return {e.id for e in self.entries} | {t.id for t in self.templates}

    def stats(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "parsed_lines": self.parsed_lines,
            "parse_rate": round(self.parsed_lines / self.total_lines, 3) if self.total_lines else 0.0,
            "distinct_templates": len(self.templates),
            "services": self.services,
            "level_counts": self.level_counts,
            "time_span_seconds": (
                round((self.end_time - self.start_time).total_seconds(), 1)
                if self.start_time and self.end_time
                else None
            ),
            "first_error_id": self.first_error_id,
            "first_anomaly_id": self.first_anomaly_id,
            "truncated": self.truncated,
        }

    def skeleton(self) -> str:
        """Global context that must accompany every chunk of a large log."""
        lines = [
            "LOG OVERVIEW (global context - true for the whole log, not just this excerpt)",
            f"  lines: {self.total_lines} ({self.parsed_lines} parsed into fields)",
            f"  window: {self.start_time or 'unknown'} .. {self.end_time or 'unknown'}",
            f"  services: {', '.join(self.services) or 'none identified'}",
            f"  levels: {self.level_counts or 'none identified'}",
        ]
        if self.first_anomaly_id:
            lines.append(f"  earliest anomaly: {self.first_anomaly_id}")
        if self.first_error_id:
            lines.append(f"  earliest ERROR:   {self.first_error_id}")
        if self.templates:
            lines.append("  most frequent message shapes:")
            for tpl in sorted(self.templates, key=lambda t: -t.count)[:12]:
                lines.append(f"    [{tpl.id}] x{tpl.count} {tpl.level or '-'} :: {tpl.template[:110]}")
        return "\n".join(lines)

    def render_selected(self, budget_tokens: int) -> tuple[str, dict]:
        """Skeleton plus the highest-value lines that fit `budget_tokens`.

        Replaces "keep the first N of each pattern" with an explicit relevance
        score, so a tight budget spends itself on the lines that carry
        diagnostic weight rather than on whichever happened to come first.
        """
        from .relevance import select_within_budget
        from .tokens import estimate_tokens

        skeleton = self.skeleton()

        def assemble(kept, note_text):
            body = "\n".join(_render_entry(e) for e in kept)
            note = f"\n\n{note_text}" if note_text else ""
            return f"{skeleton}\n\nSELECTED LINES\n{body}{note}"

        # Budget for the lines themselves, leaving room for the skeleton and
        # the section framing.
        overhead = estimate_tokens(skeleton) + 160
        report = select_within_budget(
            self, max(0, budget_tokens - overhead), estimate=estimate_tokens, render=_render_entry
        )

        # Verify the assembled result rather than trusting the headroom
        # estimate. Guessing the framing cost overshot by 10-130 tokens, which
        # meant the caller's fit check failed every time and this strategy
        # never activated -- a large log fell through to chunking and was then
        # refused for needing too many parts.
        kept = list(report.kept)
        rendered = assemble(kept, report.coverage_note)
        while kept and estimate_tokens(rendered) > budget_tokens:
            # Drop from the end: entries are ordered by line number, and the
            # selector already ranked by relevance, so the tail is the least
            # valuable material that survived scoring.
            kept.pop()
            rendered = assemble(kept, report.coverage_note)

        if len(kept) != len(report.kept):
            report.kept = kept

        return rendered, report.public()

    def render(
        self,
        detail: Literal["full", "summary", "skeleton"] = "full",
        *,
        max_severe: int = 200,
    ) -> str:
        if detail == "skeleton":
            return self.skeleton()
        if detail == "summary":
            keep, elided = _representative_entries(self, max_severe=max_severe)
            body = "\n".join(_render_entry(e) for e in keep)
            note = (
                f"\n... {elided} further severe lines omitted; their shapes and counts "
                f"are in the template list above ..."
                if elided
                else ""
            )
            return f"{self.skeleton()}\n\nREPRESENTATIVE LINES\n{body}{note}"
        body = "\n".join(_render_entry(e) for e in self.entries)
        return f"{self.skeleton()}\n\nALL LINES\n{body}"


def _render_entry(entry: LogEntry) -> str:
    parts = [f"[{entry.id}]"]
    if entry.raw_timestamp:
        parts.append(entry.raw_timestamp)
    if entry.level:
        parts.append(entry.level)
    if entry.service:
        parts.append(entry.service)
    if entry.component:
        parts.append(f"[{entry.component}]")
    parts.append(entry.message)
    return " ".join(parts)


def _representative_entries(
    ir: LogIR, per_template: int = 2, *, max_severe: int = 200
) -> tuple[list[LogEntry], int]:
    """First and last occurrence of each template, plus bounded severe lines.

    Keeps the boundaries of each pattern -- when a message started and stopped
    appearing is usually the signal -- without keeping the repetitions.

    Severe lines are capped: an incident with 3,000 identical stack traces
    would otherwise reproduce all 3,000 and overflow the window, which is the
    exact failure this layer exists to prevent. The head and tail are kept
    because the onset and the recovery are what the analysis turns on, and the
    count of what was dropped is reported so nothing disappears silently.
    """
    by_id = {e.id: e for e in ir.entries}
    keep: dict[str, LogEntry] = {}
    for tpl in ir.templates:
        for entry_id in tpl.example_ids[:per_template]:
            if entry_id in by_id:
                keep[entry_id] = by_id[entry_id]

    severe = [e for e in ir.entries if (e.level or "").upper() in {"ERROR", "FATAL", "CRITICAL", "SEVERE"}]
    elided = 0
    if len(severe) > max_severe:
        half = max_severe // 2
        elided = len(severe) - max_severe
        severe = severe[:half] + severe[-half:]
    for entry in severe:
        keep[entry.id] = entry

    return sorted(keep.values(), key=lambda e: e.line_no), elided


# --- code -----------------------------------------------------------------


class ParamSpec(BaseModel):
    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "positional"


class FunctionSpec(BaseModel):
    id: str
    name: str
    qualname: str
    signature: str
    params: list[ParamSpec] = Field(default_factory=list)
    returns: str | None = None
    docstring: str | None = None
    is_async: bool = False
    decorators: list[str] = Field(default_factory=list)
    line_start: int
    line_end: int
    # Mechanically derived testability facts. These are what let the prompt ask
    # for specific cases instead of "think of some edge cases".
    raises: list[str] = Field(default_factory=list)
    branch_count: int = 0
    loop_count: int = 0
    return_count: int = 0
    cyclomatic_complexity: int = 1
    calls: list[str] = Field(default_factory=list)
    external_calls: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    magic_numbers: list[str] = Field(default_factory=list)
    mutates_arguments: bool = False
    has_side_effects: bool = False
    source: str = ""


class ClassSpec(BaseModel):
    id: str
    name: str
    bases: list[str] = Field(default_factory=list)
    docstring: str | None = None
    methods: list[FunctionSpec] = Field(default_factory=list)
    attributes: list[ParamSpec] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    line_start: int
    line_end: int


class CodeIR(IRBase):
    kind: Literal["code"] = "code"
    language: str = "unknown"
    parser: str = "none"
    confidence: float = 0.0
    imports: list[str] = Field(default_factory=list)
    third_party_imports: list[str] = Field(default_factory=list)
    functions: list[FunctionSpec] = Field(default_factory=list)
    classes: list[ClassSpec] = Field(default_factory=list)
    module_docstring: str | None = None
    syntax_error: str | None = None
    total_lines: int = 0
    source: str = ""

    def all_functions(self) -> list[FunctionSpec]:
        out = list(self.functions)
        for cls in self.classes:
            out.extend(cls.methods)
        return out

    def evidence_ids(self) -> set[str]:
        return {f.id for f in self.all_functions()} | {c.id for c in self.classes}

    def stats(self) -> dict:
        funcs = self.all_functions()
        return {
            "language": self.language,
            "parser": self.parser,
            "parse_confidence": self.confidence,
            "functions": len(funcs),
            "classes": len(self.classes),
            "total_complexity": sum(f.cyclomatic_complexity for f in funcs),
            "external_dependencies": sorted({c for f in funcs for c in f.external_calls}),
            "explicit_raises": sorted({r for f in funcs for r in f.raises}),
            "syntax_error": self.syntax_error,
        }

    def render(self, detail: Literal["full", "summary", "skeleton"] = "full") -> str:
        blocks = [
            f"LANGUAGE: {self.language} (extracted by: {self.parser})",
        ]
        if self.syntax_error:
            blocks.append(f"PARSE WARNING: {self.syntax_error}")
        if self.third_party_imports:
            blocks.append(f"THIRD-PARTY IMPORTS (mock at these boundaries): {', '.join(self.third_party_imports)}")

        for cls in self.classes:
            blocks.append(f"\nCLASS [{cls.id}] {cls.name}({', '.join(cls.bases)}) lines {cls.line_start}-{cls.line_end}")
            if cls.attributes:
                attrs = ", ".join(f"{a.name}: {a.annotation or '?'}" for a in cls.attributes)
                blocks.append(f"  attributes: {attrs}")

        for func in self.all_functions():
            blocks.append(_render_function(func, detail))

        # The extracted facts already inline each function's source. Appending
        # the whole file again sent the same code twice in one prompt, which on
        # a tokens-per-minute budget is a straight waste of a third of it. Only
        # module-level code the function renders cannot show is added back.
        if detail == "full" and self.source:
            leftover = _module_level_source(self)
            if leftover.strip():
                blocks.append(f"\nMODULE-LEVEL CODE (outside any function)\n{leftover}")
        return "\n".join(blocks)


def _module_level_source(ir: "CodeIR") -> str:
    """Source lines not already covered by a rendered function or class."""
    covered: set[int] = set()
    for func in ir.all_functions():
        covered.update(range(func.line_start, func.line_end + 1))
    for cls in ir.classes:
        covered.update(range(cls.line_start, cls.line_end + 1))
    lines = ir.source.split("\n")
    return "\n".join(
        line for number, line in enumerate(lines, start=1) if number not in covered
    )


def _render_function(func: FunctionSpec, detail: str) -> str:
    lines = [f"\nFUNCTION [{func.id}] {func.signature}  (lines {func.line_start}-{func.line_end})"]
    if func.docstring:
        lines.append(f"  docstring: {func.docstring.strip()[:400]}")
    facts = [
        f"complexity={func.cyclomatic_complexity}",
        f"branches={func.branch_count}",
        f"loops={func.loop_count}",
        f"returns={func.return_count}",
    ]
    lines.append(f"  control flow: {', '.join(facts)}")
    if func.raises:
        lines.append(f"  raises (explicit, from the code): {', '.join(sorted(set(func.raises)))}")
    if func.external_calls:
        lines.append(f"  external calls to mock: {', '.join(sorted(set(func.external_calls)))}")
    if func.comparisons:
        lines.append(f"  boundary conditions in source: {'; '.join(func.comparisons[:12])}")
    if func.magic_numbers:
        lines.append(f"  literal values used: {', '.join(func.magic_numbers[:12])}")
    if func.mutates_arguments:
        lines.append("  NOTE: mutates one of its arguments")
    return "\n".join(lines)


# --- API routes -----------------------------------------------------------


class FieldSpec(BaseModel):
    name: str
    type: str = "string"
    required: bool = True
    default: str | None = None
    constraints: dict = Field(default_factory=dict)
    description: str | None = None
    type_inferred: bool = False


class ModelSpec(BaseModel):
    id: str
    name: str
    fields: list[FieldSpec] = Field(default_factory=list)
    docstring: str | None = None


class ResponseSpec(BaseModel):
    status: int
    description: str = ""
    model: str | None = None
    # True when found by reading `raise HTTPException(...)` rather than a
    # declared response model -- i.e. a real error path the docs would miss.
    from_raise: bool = False


class RouteSpec(BaseModel):
    id: str
    method: str
    path: str
    handler: str
    summary: str | None = None
    docstring: str | None = None
    path_params: list[FieldSpec] = Field(default_factory=list)
    query_params: list[FieldSpec] = Field(default_factory=list)
    header_params: list[FieldSpec] = Field(default_factory=list)
    body_model: str | None = None
    response_model: str | None = None
    success_status: int = 200
    responses: list[ResponseSpec] = Field(default_factory=list)
    auth_dependencies: list[str] = Field(default_factory=list)
    line_start: int = 0


class RouteIR(IRBase):
    kind: Literal["routes"] = "routes"
    framework: str = "unknown"
    parser: str = "none"
    base_prefix: str = ""
    routes: list[RouteSpec] = Field(default_factory=list)
    models: list[ModelSpec] = Field(default_factory=list)
    syntax_error: str | None = None
    source: str = ""

    def evidence_ids(self) -> set[str]:
        return {r.id for r in self.routes} | {m.id for m in self.models}

    def stats(self) -> dict:
        return {
            "framework": self.framework,
            "parser": self.parser,
            "routes": len(self.routes),
            "models": len(self.models),
            "methods": sorted({r.method for r in self.routes}),
            "error_statuses_found": sorted(
                {resp.status for r in self.routes for resp in r.responses if resp.status >= 400}
            ),
            "authenticated_routes": sum(1 for r in self.routes if r.auth_dependencies),
            "inferred_types": sum(
                1 for m in self.models for f in m.fields if f.type_inferred
            ),
            "syntax_error": self.syntax_error,
        }

    def render(self, detail: Literal["full", "summary", "skeleton"] = "full") -> str:
        blocks = [f"FRAMEWORK: {self.framework} (extracted by: {self.parser})"]
        if self.base_prefix:
            blocks.append(f"ROUTER PREFIX: {self.base_prefix}")

        for model in self.models:
            blocks.append(f"\nSCHEMA [{model.id}] {model.name}")
            for field in model.fields:
                bits = [f"    {field.name}: {field.type}"]
                bits.append("required" if field.required else f"optional (default {field.default})")
                if field.constraints:
                    bits.append(f"constraints={field.constraints}")
                if field.type_inferred:
                    bits.append("TYPE INFERRED - not declared in source")
                blocks.append("  ".join(bits))

        for route in self.routes:
            blocks.append(f"\nROUTE [{route.id}] {route.method} {route.path}  -> {route.handler}()")
            if route.docstring:
                blocks.append(f"  docstring: {route.docstring.strip()[:300]}")
            if route.auth_dependencies:
                blocks.append(f"  auth: {', '.join(route.auth_dependencies)}")
            if route.path_params:
                blocks.append(f"  path params: {', '.join(f'{p.name}: {p.type}' for p in route.path_params)}")
            if route.query_params:
                qp = ", ".join(
                    f"{p.name}: {p.type}{'' if p.required else ' (optional)'}"
                    f"{' ' + str(p.constraints) if p.constraints else ''}"
                    for p in route.query_params
                )
                blocks.append(f"  query params: {qp}")
            if route.body_model:
                blocks.append(f"  request body: {route.body_model}")
            blocks.append(f"  success: {route.success_status}" + (f" -> {route.response_model}" if route.response_model else ""))
            for resp in route.responses:
                if resp.status >= 400:
                    origin = "raised in handler" if resp.from_raise else "declared"
                    blocks.append(f"  error {resp.status}: {resp.description} ({origin})")
        return "\n".join(blocks)


# --- incident transcripts -------------------------------------------------


class Speaker(BaseModel):
    """A participant, already anonymized.

    `placeholder` is the only identifier that ever leaves the process. The real
    name lives in a side table that the pipeline never serializes, which is
    what makes the blameless guarantee mechanical rather than a prompt request.
    """

    id: str
    placeholder: str
    inferred_role: str | None = None
    message_count: int = 0
    first_utterance_id: str | None = None


class Utterance(BaseModel):
    id: str
    line_no: int
    raw_timestamp: str | None = None
    timestamp: datetime | None = None
    speaker_id: str
    speaker_placeholder: str
    text: str
    # Deterministic signals used to seed the timeline and action items.
    is_question: bool = False
    signals: list[str] = Field(default_factory=list)


class TranscriptIR(IRBase):
    kind: Literal["transcript"] = "transcript"
    channel: str | None = None
    speakers: list[Speaker] = Field(default_factory=list)
    utterances: list[Utterance] = Field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    detection_id: str | None = None
    mitigation_id: str | None = None
    resolution_id: str | None = None
    total_lines: int = 0
    parsed_lines: int = 0

    def evidence_ids(self) -> set[str]:
        return {u.id for u in self.utterances}

    def allowed_placeholders(self) -> set[str]:
        return {s.placeholder for s in self.speakers}

    def stats(self) -> dict:
        return {
            "participants": len(self.speakers),
            "utterances": len(self.utterances),
            "parsed_lines": self.parsed_lines,
            "total_lines": self.total_lines,
            "window": [self.start_time, self.end_time],
            "roles": {s.placeholder: s.inferred_role for s in self.speakers},
            "detection_id": self.detection_id,
            "mitigation_id": self.mitigation_id,
            "resolution_id": self.resolution_id,
        }

    def render(self, detail: Literal["full", "summary", "skeleton"] = "full") -> str:
        header = [
            "PARTICIPANTS (use these placeholders; real names are not available to you)",
        ]
        for speaker in self.speakers:
            role = f" - acted as {speaker.inferred_role}" if speaker.inferred_role else ""
            header.append(f"  {speaker.placeholder}{role} ({speaker.message_count} messages)")

        markers = []
        for label, ref in (
            ("detection", self.detection_id),
            ("mitigation", self.mitigation_id),
            ("resolution", self.resolution_id),
        ):
            if ref:
                markers.append(f"  {label}: {ref}")
        if markers:
            header.append("DETERMINISTICALLY DETECTED MARKERS")
            header.extend(markers)

        if detail == "skeleton":
            return "\n".join(header)

        body = []
        for utt in self.utterances:
            stamp = f"{utt.raw_timestamp} " if utt.raw_timestamp else ""
            tags = f"  <{','.join(utt.signals)}>" if utt.signals else ""
            body.append(f"[{utt.id}] {stamp}{utt.speaker_placeholder}: {utt.text}{tags}")
        return "\n".join(header) + "\n\nTRANSCRIPT\n" + "\n".join(body)


AnyIR = LogIR | CodeIR | RouteIR | TranscriptIR
