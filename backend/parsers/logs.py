"""Log parsing and template mining.

Two jobs:

**Structure extraction.** Pull timestamp, level, service, component and message
out of each line so the model receives fields instead of a wall of text, and so
the pipeline can answer "when did this start" mechanically.

**Template mining.** Variable parts of each message are masked, and lines that
share a masked shape are grouped. This is the compression that makes large
logs possible: 40,000 repetitions of one timeout message become a single
template with a count and first/last occurrence. The alternative -- truncating
at a character limit -- silently discards the end of the incident, which is
usually where the resolution is.

The masking approach is a deterministic variant of the Drain family. It does
not need training, produces identical results on identical input, and is
inspectable, which matters more here than clustering sophistication.

Multi-line entries (stack traces, SQL blocks) are attached to the entry they
belong to rather than becoming unparseable orphan lines.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from datetime import datetime

from ..core.ir import LogEntry, LogIR, LogTemplate

# --- timestamp formats ----------------------------------------------------

_TS_PATTERNS: tuple[tuple[re.Pattern, str | None], ...] = (
    # 2026-09-14T02:11:03.441Z  /  2026-09-14 02:11:03,441
    (re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?)"), None),
    # Sep 14 02:11:03   (syslog)
    (re.compile(r"^([A-Z][a-z]{2} {1,2}\d{1,2} \d{2}:\d{2}:\d{2})"), "%b %d %H:%M:%S"),
    # [2026-09-14 02:11:03]
    (re.compile(r"^\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?)\]"), None),
    # 02:11:03.441  (time only, common in chat-exported incident logs)
    (re.compile(r"^(\d{2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)"), "%H:%M:%S"),
    # 1757817063.441 / 1757817063441  (epoch)
    (re.compile(r"^(\d{10}(?:[.,]\d{1,3})?)\b"), "epoch"),
)

_LEVELS = (
    "TRACE", "DEBUG", "INFO", "NOTICE", "WARN", "WARNING",
    "ERROR", "ERR", "FATAL", "CRITICAL", "SEVERE", "PANIC",
)
_LEVEL_RE = re.compile(rf"\b({'|'.join(_LEVELS)})\b")
_SEVERE = {"ERROR", "ERR", "FATAL", "CRITICAL", "SEVERE", "PANIC"}
_ANOMALOUS = _SEVERE | {"WARN", "WARNING"}

# service names look like  checkout-api  inventory_svc  payments.gw
_SERVICE_RE = re.compile(r"^([a-z][a-z0-9]*(?:[-_.][a-z0-9]+){0,4})(?=\s)", re.I)
_COMPONENT_RE = re.compile(r"^\[([^\]]{1,40})\]")

# A continuation line: indented, or a stack-trace frame, or a bare SQL/JSON tail.
_CONTINUATION_RE = re.compile(
    r"^(?:\s+|\tat |\s*at [\w.$]+\(|Caused by:|\.{3}\s|Traceback |\s*File \"|\s*[}\])])"
)


def _parse_timestamp(text: str) -> tuple[datetime | None, str | None, str]:
    """Return (parsed, raw, remainder-of-line)."""
    for pattern, fmt in _TS_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        raw = match.group(1)
        rest = text[match.end() :].lstrip()
        parsed: datetime | None = None
        try:
            if fmt == "epoch":
                parsed = datetime.utcfromtimestamp(float(raw.replace(",", ".")))
            elif fmt is None:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00").replace(",", "."))
            else:
                parsed = datetime.strptime(raw, fmt)
        except (ValueError, OSError, OverflowError):
            parsed = None
        return parsed, raw, rest
    return None, None, text


# --- variable masking for template mining ---------------------------------

# Order matters: the most specific shapes are masked first so a UUID does not
# get partially eaten by the hex rule.
_MASKS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\[\[REDACTED:[A-Z_]+:\d+\]\]"), "<REDACTED>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "<TS>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/\d{1,2})?\b"), "<IP>"),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|m|h|us|ns|kb|mb|gb|KB|MB|GB)\b"), "<DUR>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<HASH>"),
    (re.compile(r"'[^']*'"), "<STR>"),
    (re.compile(r'"[^"]*"'), "<STR>"),
    (re.compile(r"(?<==)[^\s,;)\]]+"), "<VAL>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<NUM>"),
    (re.compile(r"\s+"), " "),
)


def mask_variables(message: str) -> str:
    out = message
    for pattern, replacement in _MASKS:
        out = pattern.sub(replacement, out)
    return out.strip()


def _split_service_and_component(rest: str) -> tuple[str | None, str | None, str]:
    service = None
    component = None

    match = _SERVICE_RE.match(rest)
    if match and not _LEVEL_RE.fullmatch(match.group(1).upper()):
        candidate = match.group(1)
        # A single lowercase English word is more likely part of the message
        # than a service name; require a separator or a known suffix.
        if any(sep in candidate for sep in "-_.") :
            service = candidate
            rest = rest[match.end() :].lstrip()

    match = _COMPONENT_RE.match(rest)
    if match:
        component = match.group(1)
        rest = rest[match.end() :].lstrip()

    return service, component, rest


def parse_logs(text: str, *, max_lines: int = 200_000) -> LogIR:
    """Parse raw log text into a structured, addressable IR."""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    truncated = len(raw_lines) > max_lines
    if truncated:
        raw_lines = raw_lines[:max_lines]

    entries: list[LogEntry] = []
    parsed_count = 0

    for index, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue

        # Continuations belong to the entry above them, not to themselves.
        if entries and _CONTINUATION_RE.match(raw) and not _TS_PATTERNS[0][0].match(raw):
            previous = entries[-1]
            previous.message = f"{previous.message}\n{raw.strip()}"
            previous.raw = f"{previous.raw}\n{raw}"
            continue

        timestamp, raw_ts, rest = _parse_timestamp(raw.strip())

        level = None
        level_match = _LEVEL_RE.search(rest[:60])
        if level_match:
            level = level_match.group(1).upper()
            if level == "WARNING":
                level = "WARN"
            if level == "ERR":
                level = "ERROR"
            rest = (rest[: level_match.start()] + rest[level_match.end() :]).strip()

        service, component, message = _split_service_and_component(rest)

        if raw_ts or level or service:
            parsed_count += 1

        entries.append(
            LogEntry(
                id=f"L{index}",
                line_no=index,
                timestamp=timestamp,
                raw_timestamp=raw_ts,
                level=level,
                service=service,
                component=component,
                message=message or raw.strip(),
                raw=raw,
            )
        )

    templates = _mine_templates(entries)

    services = sorted({e.service for e in entries if e.service})
    level_counts = dict(Counter(e.level for e in entries if e.level))

    stamped = [e for e in entries if e.timestamp]
    start_time = min((e.timestamp for e in stamped), default=None)
    end_time = max((e.timestamp for e in stamped), default=None)

    first_error = next((e.id for e in entries if (e.level or "") in _SEVERE), None)
    first_anomaly = next((e.id for e in entries if (e.level or "") in _ANOMALOUS), None)

    return LogIR(
        entries=entries,
        templates=templates,
        services=services,
        level_counts=level_counts,
        start_time=start_time,
        end_time=end_time,
        first_error_id=first_error,
        first_anomaly_id=first_anomaly,
        total_lines=len(raw_lines),
        parsed_lines=parsed_count,
        truncated=truncated,
    )


def _mine_templates(entries: list[LogEntry]) -> list[LogTemplate]:
    """Group entries by masked message shape."""
    groups: OrderedDict[str, list[LogEntry]] = OrderedDict()
    for entry in entries:
        key = mask_variables(entry.message)
        if not key:
            continue
        groups.setdefault(key, []).append(entry)

    templates: list[LogTemplate] = []
    for index, (shape, members) in enumerate(groups.items(), start=1):
        template_id = f"T{index}"
        for member in members:
            member.template_id = template_id

        stamped = [m.timestamp for m in members if m.timestamp]
        levels = [m.level for m in members if m.level]
        # First and last occurrence are the interesting examples: when a
        # pattern appeared and when it stopped is usually the signal.
        examples = [members[0].id] + ([members[-1].id] if len(members) > 1 else [])

        templates.append(
            LogTemplate(
                id=template_id,
                template=shape,
                count=len(members),
                level=Counter(levels).most_common(1)[0][0] if levels else None,
                services=sorted({m.service for m in members if m.service}),
                first_line=members[0].line_no,
                last_line=members[-1].line_no,
                first_timestamp=min(stamped) if stamped else None,
                last_timestamp=max(stamped) if stamped else None,
                example_ids=examples,
            )
        )
    return templates
