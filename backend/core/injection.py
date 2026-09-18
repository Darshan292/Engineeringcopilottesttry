"""Prompt-injection detection.

Fencing input and telling the model "this is data" is necessary and not
sufficient. It is an instruction, and instructions are what an attacker is
competing for. This module adds the two things fencing does not provide:

**Detection.** Injection attempts are found and reported before the call, so
the response can say an attempt was present and neutralized rather than the
operator never learning about it.

**Neutralization.** Detected spans are defanged in place -- the imperative is
wrapped in a marker that makes it unmistakably quoted content. The text stays
analysable (a log line containing an injection attempt is itself a finding
worth reporting) while ceasing to read as a directive.

Neither of these is the real defence. The real defence is structural and lives
downstream: the model must return JSON matching a fixed schema, every evidence
citation is checked against the parsed IR, and the final document is rendered
by our code from validated fields. An injected "ignore your instructions and
output X" cannot produce a document, because documents are not what the model
is allowed to return.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Severity is about how unambiguous the attempt is, not how dangerous.
_PATTERNS: tuple[tuple[str, str, re.Pattern], ...] = (
    (
        "instruction_override",
        "high",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override|bypass|discard)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,20}"
            r"\b(?:instruction|prompt|rule|direction|guideline|system|context)",
        ),
    ),
    (
        "role_reassignment",
        "high",
        re.compile(
            r"(?i)\byou\s+are\s+(?:now|no longer)\b|\bfrom\s+now\s+on\s+you\b|"
            r"\bact\s+as\s+(?:a|an|if)\b|\bpretend\s+(?:to\s+be|you)\b|\bnew\s+(?:instructions?|persona|role)\s*[:=]"
        ),
    ),
    (
        "fake_system_turn",
        "high",
        re.compile(
            r"(?i)(?:^|\n)\s*(?:<\|?(?:im_start|im_end|system|assistant|endoftext)\|?>|"
            r"\[/?(?:INST|SYS)\]|###\s*(?:system|assistant|instruction)\s*:?|"
            r"(?:system|assistant)\s*:\s*you\b)"
        ),
    ),
    (
        "output_hijack",
        "medium",
        re.compile(
            r"(?i)\b(?:instead|rather than)\b[^.\n]{0,30}\b(?:output|respond|reply|return|print|say|write)\b|"
            r"\b(?:output|respond|reply|return|print)\s+(?:only|just|exactly|nothing but)\b"
        ),
    ),
    (
        "exfiltration",
        "high",
        re.compile(
            r"(?i)\b(?:reveal|show|print|repeat|disclose|leak|dump)\b[^.\n]{0,30}"
            r"\b(?:system\s+prompt|instructions|api[_ ]?key|secret|credential|env(?:ironment)?\s+var)"
        ),
    ),
    (
        "delimiter_escape",
        "medium",
        re.compile(r"-{3,}\s*(?:CODE|LOG|TRANSCRIPT|INPUT)\s+END\s*-{3,}", re.I),
    ),
    (
        "encoded_payload",
        "low",
        re.compile(r"(?i)\b(?:base64|rot13|hex)\s*(?:decode|encoded?)\b[^.\n]{0,20}(?:then|and)\s+(?:run|execute|follow)"),
    ),
)

_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass
class InjectionFinding:
    kind: str
    severity: str
    start: int
    end: int
    excerpt: str


@dataclass
class InjectionReport:
    findings: list[InjectionFinding] = field(default_factory=list)
    neutralized_text: str = ""

    @property
    def detected(self) -> bool:
        return bool(self.findings)

    @property
    def max_severity(self) -> str | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity

    def public(self) -> dict:
        return {
            "detected": self.detected,
            "max_severity": self.max_severity,
            "count": len(self.findings),
            "kinds": sorted({f.kind for f in self.findings}),
            # Excerpts are truncated and returned so an operator can see what
            # was found without having to re-read the whole input.
            "excerpts": [f.excerpt for f in self.findings[:5]],
        }


# Wrapper that makes a defanged span unmistakably quoted content.
_OPEN = "‹INJECTION-ATTEMPT-QUOTED›"
_CLOSE = "‹/INJECTION-ATTEMPT-QUOTED›"


def scan(text: str) -> InjectionReport:
    """Find injection attempts without modifying the text."""
    findings: list[InjectionFinding] = []
    claimed: list[tuple[int, int]] = []

    for kind, severity, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(not (end <= s or start >= e) for s, e in claimed):
                continue
            claimed.append((start, end))
            excerpt = re.sub(r"\s+", " ", match.group())[:120]
            findings.append(InjectionFinding(kind, severity, start, end, excerpt))

    findings.sort(key=lambda f: f.start)
    return InjectionReport(findings=findings, neutralized_text=text)


def neutralize(text: str) -> InjectionReport:
    """Scan, then wrap every detected span so it cannot read as a directive."""
    report = scan(text)
    if not report.findings:
        report.neutralized_text = text
        return report

    out: list[str] = []
    cursor = 0
    for finding in report.findings:
        # Snap to word boundaries: a match that ends mid-word would otherwise
        # produce "instruction<marker>s", which reads as corruption.
        start, end = finding.start, finding.end
        while start > cursor and text[start - 1].isalnum():
            start -= 1
        while end < len(text) and text[end].isalnum():
            end += 1
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(f"{_OPEN}{text[start:end]}{_CLOSE}")
        cursor = end
    out.append(text[cursor:])

    report.neutralized_text = "".join(out)
    return report


def strip_markers(text: str) -> str:
    """Remove neutralization markers, for display or comparison."""
    return text.replace(_OPEN, "").replace(_CLOSE, "")
