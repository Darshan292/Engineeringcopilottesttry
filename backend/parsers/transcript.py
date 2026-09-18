"""Incident transcript parsing and structural anonymization.

The postmortem tool's central promise is that no human name reaches the output.
Asking a model nicely does not deliver that -- it delivers it most of the time,
which is the same as not delivering it.

Here the guarantee is structural. Names are extracted from speaker prefixes,
mapped to role placeholders, and replaced everywhere they occur, including
inside message bodies where one participant addresses another by name. The
name-to-placeholder table stays in process memory and is never serialized into
a prompt, a response, or a log line. The model cannot leak a name it was never
shown.

Roles are inferred from what people did, using explicit phrases in the
transcript ("taking IC", "I own inventory-svc"), so `[IC]` and `[SERVICE_OWNER]`
carry real meaning rather than being anonymous numbering.

Marker detection -- when the incident was detected, mitigated and resolved --
is done here too, because those are findable with patterns and are exactly the
timeline rows that get argued about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.ir import Speaker, TranscriptIR, Utterance

# --- speaker line formats -------------------------------------------------

_SPEAKER_PATTERNS: tuple[re.Pattern, ...] = (
    # [02:18] Priya Raghavan: text        (and 2026-09-14T02:18:03 variants)
    re.compile(r"^\[(?P<ts>[^\]]{3,30})\]\s*(?P<name>[A-Za-z][\w .'\-]{0,40}?)\s*:\s*(?P<text>.*)$"),
    # 02:18 Priya Raghavan: text
    re.compile(r"^(?P<ts>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s+(?P<name>[A-Za-z][\w .'\-]{0,40}?)\s*:\s*(?P<text>.*)$"),
    # Priya Raghavan [02:18]: text        (Slack export)
    re.compile(r"^(?P<name>[A-Za-z][\w .'\-]{0,40}?)\s*\[(?P<ts>[^\]]{3,30})\]\s*:?\s*(?P<text>.*)$"),
    # Priya Raghavan (02:18) text
    re.compile(r"^(?P<name>[A-Za-z][\w .'\-]{0,40}?)\s*\((?P<ts>\d{1,2}:\d{2}[^)]*)\)\s*:?\s*(?P<text>.*)$"),
    # <priya> text                        (IRC)
    re.compile(r"^<(?P<name>[\w .'\-]{1,40})>\s*(?P<text>.*)$"),
    # Priya Raghavan: text                (no timestamp)
    re.compile(r"^(?P<name>[A-Z][\w .'\-]{0,40}?)\s*:\s{1,4}(?P<text>\S.*)$"),
)

# Words that look like a speaker prefix but are not a person.
_NOT_A_NAME = {
    "note", "update", "edit", "warning", "error", "info", "status", "summary",
    "http", "https", "tldr", "todo", "fyi", "ps", "re", "action", "next steps",
    "root cause", "impact", "timeline", "resolution", "mitigation", "eta",
}

# --- role inference -------------------------------------------------------

# Ordered by specificity: the first rule that matches a speaker's messages wins.
_ROLE_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("INCIDENT_COMMANDER", re.compile(r"\b(?:taking|i'?ll take|i am|i'?m)\s+(?:the\s+)?ic\b|\bincident commander\b|\bcalling it\b", re.I)),
    ("COMMS", re.compile(r"\b(?:own|owning|handle|updating|updated)\s+(?:the\s+)?(?:comms|statuspage|status page)\b|\bstatuspage updated\b", re.I)),
    ("ONCALL", re.compile(r"\b(?:getting|got|i'?m)\s+paged\b|\bi'?m on ?call\b|\bpager\b", re.I)),
    ("SERVICE_OWNER", re.compile(r"\bi own\b|\bmy service\b|\bi maintain\b|\bi'?m the owner\b", re.I)),
    ("DEPLOYER", re.compile(r"\bi (?:pushed|deployed|shipped|released|merged)\b", re.I)),
    ("DBA", re.compile(r"\bi'?ll (?:check|look at) (?:the )?(?:db|database|replica)\b", re.I)),
)

_GENERIC_ROLE = "RESPONDER"

# --- utterance signals ----------------------------------------------------

_SIGNAL_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("detection", re.compile(r"\b(?:paged|alert(?:ed|ing)?|noticed|seeing|spiking|dashboards? (?:are|is) red|reports? of)\b", re.I)),
    ("hypothesis", re.compile(r"\b(?:maybe|might be|could be|i think|suspect|looks like|lines up|probably|theory)\b", re.I)),
    ("ruled_out", re.compile(r"\b(?:no deploys|not (?:it|that|the cause)|ruled out|that'?s not|hours ago|unrelated)\b", re.I)),
    ("action", re.compile(r"\b(?:scaling|scaled|restart(?:ing|ed)?|kill(?:ing|ed)?|cancel(?:ling|led)?|roll(?:ing|ed)? back|failing over|disabled|reverted)\b", re.I)),
    ("mitigation", re.compile(r"\b(?:cancelled|killed|rolled back|reverted|drained|scaled down|failed over|disabled)\b", re.I)),
    ("recovery", re.compile(r"\b(?:recovering|draining|first success|back to normal|closed|circuit breaker closed|pool is draining)\b", re.I)),
    ("resolution", re.compile(r"\b(?:mitigated|resolved|calling it|all clear|incident over|monitoring)\b", re.I)),
    ("gap", re.compile(r"\b(?:we never|no alert|didn'?t (?:catch|know|notice)|nobody caught|the real miss|should have)\b", re.I)),
    ("impact", re.compile(r"\b(?:customers?|users?|orders?|failed|lost|revenue|backlog|\d+\s*(?:%|percent))\b", re.I)),
    ("escalation", re.compile(r"\b(?:joining|paging|escalat(?:e|ing)|looping in|can you)\b", re.I)),
)

_MARKER_PRIORITY = {
    "detection": ("detection", 0),
    "mitigation": ("mitigation", 1),
    "resolution": ("resolution", 2),
}

_CHANNEL_RE = re.compile(r"^#[\w\-]+", re.M)


@dataclass
class _SpeakerBuild:
    name: str
    index: int
    lines: list[str] = field(default_factory=list)
    first_utterance: str | None = None


def _looks_like_name(candidate: str) -> bool:
    text = candidate.strip()
    if not text or len(text) > 40:
        return False
    if text.lower() in _NOT_A_NAME:
        return False
    if text.lower().startswith(("http", "www.")):
        return False
    # A "name" of four or more words is a sentence that happened to contain a
    # colon, not a speaker.
    if len(text.split()) > 3:
        return False
    return bool(re.match(r"^[A-Za-z][\w .'\-]*$", text))


def _match_speaker_line(line: str):
    for pattern in _SPEAKER_PATTERNS:
        match = pattern.match(line.strip())
        if not match:
            continue
        name = (match.groupdict().get("name") or "").strip()
        if not _looks_like_name(name):
            continue
        return name, (match.groupdict().get("ts") or None), (match.groupdict().get("text") or "").strip()
    return None


def _infer_role(messages: list[str]) -> str | None:
    joined = " ".join(messages)
    for role, pattern in _ROLE_RULES:
        if pattern.search(joined):
            return role
    return None


def _name_variants(name: str) -> list[str]:
    """Every spelling of a name that could appear in a message body."""
    parts = [p for p in re.split(r"[ .]+", name) if len(p) > 2]
    variants = {name}
    variants.update(parts)
    # Slack-style mentions.
    variants.add(f"@{name}")
    variants.update(f"@{p}" for p in parts)
    # Longest first so "Priya Raghavan" is replaced before "Priya".
    return sorted({v for v in variants if len(v) > 2}, key=len, reverse=True)


def parse_transcript(text: str) -> TranscriptIR:
    """Parse and anonymize an incident transcript.

    Returns an IR in which every participant is a role placeholder. The
    real-name mapping is intentionally not part of the returned object.
    """
    raw_lines = text.replace("\r\n", "\n").split("\n")
    channel_match = _CHANNEL_RE.search(text)

    # Pass 1: identify speakers and collect their messages so roles can be
    # inferred from the whole conversation rather than one line at a time.
    builds: dict[str, _SpeakerBuild] = {}
    parsed: list[tuple[int, str, str | None, str]] = []

    for line_no, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        match = _match_speaker_line(line)
        if match is None:
            # A continuation of the previous speaker's message.
            if parsed:
                index, name, timestamp, body = parsed[-1]
                parsed[-1] = (index, name, timestamp, f"{body} {line.strip()}")
            continue
        name, timestamp, body = match
        if name not in builds:
            builds[name] = _SpeakerBuild(name=name, index=len(builds) + 1)
        builds[name].lines.append(body)
        parsed.append((line_no, name, timestamp, body))

    # Pass 2: assign placeholders. Roles are deduped with a numeric suffix so
    # two service owners do not collapse into one identity.
    role_counts: dict[str, int] = {}
    placeholders: dict[str, str] = {}
    speakers: list[Speaker] = []

    for build in builds.values():
        role = _infer_role(build.lines) or _GENERIC_ROLE
        role_counts[role] = role_counts.get(role, 0) + 1
        suffix = f"_{role_counts[role]}" if role_counts[role] > 1 or role == _GENERIC_ROLE else ""
        placeholder = f"[{role}{suffix}]"
        placeholders[build.name] = placeholder
        speakers.append(
            Speaker(
                id=f"S{build.index}",
                placeholder=placeholder,
                inferred_role=None if role == _GENERIC_ROLE else role.replace("_", " ").lower(),
                message_count=len(build.lines),
            )
        )

    speaker_ids = {name: f"S{build.index}" for name, build in builds.items()}

    # Pass 3: build utterances with names scrubbed from the bodies too. This is
    # the step that matters -- "Marcus can you own comms?" would otherwise
    # carry a real name into the prompt despite the speaker being anonymized.
    replacements: list[tuple[re.Pattern, str]] = []
    for name, placeholder in placeholders.items():
        for variant in _name_variants(name):
            replacements.append((re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)"), placeholder))
    replacements.sort(key=lambda pair: -len(pair[0].pattern))

    def scrub(body: str) -> str:
        out = body
        for pattern, placeholder in replacements:
            out = pattern.sub(placeholder, out)
        return out

    utterances: list[Utterance] = []
    markers: dict[str, str] = {}

    for index, (line_no, name, timestamp, body) in enumerate(parsed, start=1):
        clean = scrub(body)
        signals = [label for label, pattern in _SIGNAL_RULES if pattern.search(body)]

        utterance = Utterance(
            id=f"U{index}",
            line_no=line_no,
            raw_timestamp=timestamp,
            speaker_id=speaker_ids[name],
            speaker_placeholder=placeholders[name],
            text=clean,
            is_question=clean.rstrip().endswith("?"),
            signals=signals,
        )
        utterances.append(utterance)

        for signal in signals:
            if signal in _MARKER_PRIORITY:
                key = _MARKER_PRIORITY[signal][0]
                # Detection is the first occurrence; mitigation and resolution
                # are the last, because incidents get re-mitigated.
                if key == "detection":
                    markers.setdefault(key, utterance.id)
                else:
                    markers[key] = utterance.id

    for speaker in speakers:
        first = next((u.id for u in utterances if u.speaker_id == speaker.id), None)
        speaker.first_utterance_id = first

    stamps = [u.raw_timestamp for u in utterances if u.raw_timestamp]

    return TranscriptIR(
        channel=channel_match.group(0) if channel_match else None,
        speakers=speakers,
        utterances=utterances,
        start_time=stamps[0] if stamps else None,
        end_time=stamps[-1] if stamps else None,
        detection_id=markers.get("detection"),
        mitigation_id=markers.get("mitigation"),
        resolution_id=markers.get("resolution"),
        total_lines=len([ln for ln in raw_lines if ln.strip()]),
        parsed_lines=len(parsed),
    )


def extract_real_names(text: str) -> set[str]:
    """The names present in the input, for assertion purposes only.

    Tests use this to prove that no name in the source survives into the
    rendered IR or the final document. It is never called on a request path.
    """
    names: set[str] = set()
    for line in text.split("\n"):
        match = _match_speaker_line(line)
        if match:
            names.add(match[0])
    return names
