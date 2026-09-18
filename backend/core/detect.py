"""Deterministic input classification.

Cheap, explainable scoring rather than a model call. Every signal that fires is
recorded, so when the classifier is wrong you can see exactly which evidence
misled it instead of re-running a black box.

Two questions are answered here: what programming language is this, and does
the text look like the kind of input the selected tool expects. The second one
matters because pasting a log into the test generator should produce a clear
"this is not code" rather than a confidently hallucinated test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- language signals -----------------------------------------------------

_LANGUAGE_SIGNALS: dict[str, tuple[tuple[str, float], ...]] = {
    "python": (
        (r"^\s*def \w+\s*\(", 3.0),
        (r"^\s*class \w+.*:\s*$", 2.5),
        (r"^\s*(?:from [\w.]+ )?import \w", 2.0),
        (r"^\s*@\w[\w.]*\s*(?:\(|$)", 1.0),
        (r":\s*(?:str|int|float|bool|list|dict|None)\b", 1.5),
        (r"\bself\b", 1.0),
        (r'"""', 1.0),
        (r"\bTrue\b|\bFalse\b|\bNone\b", 0.8),
        (r"\bexcept\s+\w*(?:Error|Exception)", 1.5),
        (r"\bf\"", 1.0),
    ),
    "javascript": (
        (r"\b(?:const|let|var)\s+\w+\s*=", 2.0),
        (r"\bfunction\s+\w+\s*\(", 2.5),
        (r"=>\s*\{", 2.0),
        (r"\brequire\(['\"]", 2.0),
        (r"^\s*(?:export|import)\s+(?:default\s+)?\{?", 1.5),
        (r"\bconsole\.log\(", 1.5),
        (r"\basync\s+function\b", 1.5),
        (r";\s*$", 0.4),
    ),
    "typescript": (
        (r":\s*(?:string|number|boolean|void|any|unknown)\b", 2.5),
        (r"\binterface\s+\w+\s*\{", 3.0),
        (r"\btype\s+\w+\s*=", 2.0),
        (r"\bexport\s+(?:interface|type|enum)\b", 2.5),
        (r"<\w+(?:\[\])?>\s*\(", 1.0),
    ),
    "java": (
        (r"\bpublic\s+(?:static\s+)?(?:final\s+)?(?:class|interface|enum)\b", 3.0),
        (r"\b(?:public|private|protected)\s+\w[\w<>\[\], ]*\s+\w+\s*\(", 2.5),
        (r"^\s*package\s+[\w.]+;", 2.5),
        (r"\bSystem\.out\.print", 2.0),
        (r"@Override\b|@Autowired\b|@RestController\b", 2.0),
    ),
    "go": (
        (r"\bfunc\s+(?:\(\w+\s+\*?\w+\)\s*)?\w+\s*\(", 3.0),
        (r"^\s*package\s+\w+\s*$", 2.5),
        (r"\bif\s+err\s*!=\s*nil\b", 3.0),
        (r":=", 2.0),
        (r"\bfmt\.(?:Print|Sprint|Errorf)", 2.0),
    ),
    "ruby": (
        (r"^\s*def\s+\w+[\w?!]*\s*(?:\(|$)", 2.0),
        (r"^\s*end\s*$", 1.5),
        (r"\brequire\s+['\"]", 1.5),
        (r"\bputs\b", 1.5),
        (r"\bdo\s*\|\w+\|", 2.0),
    ),
    "sql": (
        (r"\bSELECT\b.*\bFROM\b", 3.0),
        (r"\b(?:INSERT INTO|UPDATE|DELETE FROM|CREATE TABLE|ALTER TABLE)\b", 3.0),
    ),
}

_COMPILED_LANGUAGE = {
    lang: tuple((re.compile(p, re.M | re.I if lang == "sql" else re.M), w) for p, w in sigs)
    for lang, sigs in _LANGUAGE_SIGNALS.items()
}


@dataclass
class Detection:
    value: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "runner_up": self._runner_up(),
        }

    def _runner_up(self) -> str | None:
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else None


def detect_language(text: str) -> Detection:
    """Score every language, return the winner with a normalized confidence."""
    if not text.strip():
        return Detection("unknown", 0.0)

    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}

    for language, signals in _COMPILED_LANGUAGE.items():
        total = 0.0
        fired: list[str] = []
        for pattern, weight in signals:
            found = len(pattern.findall(text))
            if found:
                # Diminishing returns: 200 semicolons is not 200x the evidence
                # of one, it is the same single fact about the file.
                total += weight * min(found, 5) ** 0.5
                fired.append(pattern.pattern[:30])
        if total:
            scores[language] = round(total, 2)
            hits[language] = fired

    if not scores:
        return Detection("unknown", 0.0, scores)

    # TypeScript is a superset; its signals only win when they are strong,
    # otherwise a TS-flavoured file reads as JavaScript, which is fine.
    if "typescript" in scores and "javascript" in scores:
        if scores["typescript"] >= 4.0:
            scores["typescript"] += scores["javascript"] * 0.5
        else:
            scores.pop("typescript")

    winner = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = scores[winner] / total if total else 0.0

    return Detection(winner, confidence, scores, hits.get(winner, []))


# --- input kind -----------------------------------------------------------

_TIMESTAMPED_LINE = re.compile(
    r"^\s*(?:\[)?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|^\s*\[?\d{2}:\d{2}(?::\d{2})?\]?\s", re.M
)
_LOG_LEVEL_LINE = re.compile(r"\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b")
_CHAT_LINE = re.compile(r"^\s*(?:\[)?\d{1,2}:\d{2}(?::\d{2})?(?:\])?\s*[A-Z][\w .'-]{1,40}:\s", re.M)
_ROUTE_DECORATOR = re.compile(
    r"@(?:app|router|blueprint|api)\.(?:get|post|put|patch|delete|route)\s*\(|"
    r"\b(?:app|router)\.(?:get|post|put|patch|delete)\s*\(\s*['\"]/"
    r"|@(?:Get|Post|Put|Patch|Delete|RequestMapping|GetMapping|PostMapping)\s*\(",
    re.I,
)


@dataclass
class InputKind:
    kind: str
    confidence: float
    evidence: dict = field(default_factory=dict)


def detect_input_kind(text: str) -> InputKind:
    """Classify as code / routes / logs / transcript."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return InputKind("empty", 1.0)

    total = len(lines)
    timestamped = len(_TIMESTAMPED_LINE.findall(text))
    levelled = len([ln for ln in lines if _LOG_LEVEL_LINE.search(ln)])
    chatty = len(_CHAT_LINE.findall(text))
    routes = len(_ROUTE_DECORATOR.findall(text))
    language = detect_language(text)

    evidence = {
        "lines": total,
        "timestamped_ratio": round(timestamped / total, 3),
        "log_level_ratio": round(levelled / total, 3),
        "chat_line_ratio": round(chatty / total, 3),
        "route_markers": routes,
        "language": language.value,
        "language_confidence": round(language.confidence, 3),
    }

    # A transcript is timestamped AND attributed to speakers; a log is
    # timestamped and carries levels. The speaker prefix is the discriminator.
    if chatty / total > 0.4:
        return InputKind("transcript", min(1.0, chatty / total + 0.2), evidence)

    if (timestamped / total > 0.5 and levelled / total > 0.25) or levelled / total > 0.6:
        return InputKind("logs", min(1.0, (timestamped + levelled) / (2 * total) + 0.2), evidence)

    if routes >= 1 and language.value != "unknown":
        return InputKind("routes", min(1.0, 0.6 + routes * 0.1), evidence)

    if language.value != "unknown" and language.confidence > 0.3:
        return InputKind("code", language.confidence, evidence)

    if timestamped / total > 0.4:
        return InputKind("logs", 0.5, evidence)

    return InputKind("unknown", 0.0, evidence)


# Which detected kinds each tool is prepared to handle. A mismatch is surfaced
# as a warning rather than a hard failure -- the operator may know better.
TOOL_EXPECTED_KINDS: dict[str, set[str]] = {
    "unit-tests": {"code", "routes"},
    "api-docs": {"routes", "code"},
    "log-rca": {"logs"},
    "postmortem": {"transcript", "logs"},
}


def mismatch_warning(tool: str, kind: InputKind) -> str | None:
    expected = TOOL_EXPECTED_KINDS.get(tool)
    if not expected or kind.kind in expected or kind.kind == "unknown":
        return None
    return (
        f"This input looks like {kind.kind} (confidence {kind.confidence:.0%}), but the "
        f"{tool} tool expects {' or '.join(sorted(expected))}. The result may be poor. "
        f"Signals: {kind.evidence}"
    )
