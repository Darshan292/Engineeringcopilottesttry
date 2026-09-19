"""Secret and PII redaction. Runs before anything leaves this machine.

Design constraints that shaped this:

1. **Stable placeholders.** The same secret always maps to the same token, so
   downstream reasoning survives: "the same credential appears in lines 4 and
   91" is still derivable after redaction. A random mask per occurrence would
   destroy that.
2. **Structure-preserving.** Redacting everything that *looks* sensitive makes
   RCA impossible -- private IPs, hostnames and service names are the analysis.
   Policy decides per category, and the defaults keep what an SRE needs.
3. **Entropy as a backstop, not the primary mechanism.** Pattern matches are
   precise; entropy catches the unknown-format key at the cost of false
   positives, so it only fires on assignment-like contexts.
4. **Verifiable.** Every redaction is recorded with its span and kind, so the
   API can report exactly what was withheld, and tests can assert that a known
   secret never reaches the wire.

Placeholders use `[[REDACTED:KIND:N]]`: ASCII, safe inside YAML/JSON/Markdown
and code fences, and trivially greppable when asserting on outbound payloads.
"""

from __future__ import annotations

import math
import re
import bisect
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

PLACEHOLDER_RE = re.compile(r"\[\[REDACTED:([A-Z_]+):(\d+)\]\]")


class Sensitivity(str, Enum):
    """What a match is, which decides whether policy drops or keeps it."""

    CREDENTIAL = "credential"  # always redacted; never optional
    PII = "pii"  # person-identifying
    NETWORK = "network"  # hosts/IPs; often load-bearing for RCA
    FINANCIAL = "financial"


@dataclass(frozen=True)
class Rule:
    kind: str
    sensitivity: Sensitivity
    pattern: re.Pattern
    # Which capture group holds the secret itself. Group 0 means the whole
    # match. Using a subgroup lets a rule anchor on context (`password=`)
    # while only masking the value.
    group: int = 0


def _c(pattern: str, flags: int = 0) -> re.Pattern:
    return re.compile(pattern, flags)


# Ordered most-specific first. The scanner honours this order and will not
# redact inside an already-redacted span, so a broad rule cannot clobber a
# precise one.
RULES: tuple[Rule, ...] = (
    # --- private keys and certificates ---
    Rule(
        "PRIVATE_KEY",
        Sensitivity.CREDENTIAL,
        _c(r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----", re.S),
    ),
    # --- provider-specific tokens (high precision) ---
    Rule("AWS_ACCESS_KEY", Sensitivity.CREDENTIAL, _c(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b")),
    Rule("GITHUB_TOKEN", Sensitivity.CREDENTIAL, _c(r"\b gh[pousr]_[A-Za-z0-9]{20,255}\b".replace(" ", ""))),
    Rule("GITHUB_PAT", Sensitivity.CREDENTIAL, _c(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    Rule("SLACK_TOKEN", Sensitivity.CREDENTIAL, _c(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    Rule("STRIPE_KEY", Sensitivity.CREDENTIAL, _c(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    Rule("GOOGLE_API_KEY", Sensitivity.CREDENTIAL, _c(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    Rule("OPENAI_KEY", Sensitivity.CREDENTIAL, _c(r"\bsk-[A-Za-z0-9]{20,}\b")),
    Rule("OPENROUTER_KEY", Sensitivity.CREDENTIAL, _c(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    Rule("SENDGRID_KEY", Sensitivity.CREDENTIAL, _c(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b")),
    Rule("NPM_TOKEN", Sensitivity.CREDENTIAL, _c(r"\bnpm_[A-Za-z0-9]{36}\b")),
    # --- structured credentials ---
    Rule("JWT", Sensitivity.CREDENTIAL, _c(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    Rule(
        "CONNECTION_STRING",
        Sensitivity.CREDENTIAL,
        # Mask only the credential portion; the host stays visible because
        # "which database" is usually the point of the log line.
        _c(r"(?i)\b[a-z][a-z0-9+.\-]*://([^\s/@:]+:[^\s/@]+)@"),
        group=1,
    ),
    Rule(
        "AUTH_HEADER",
        Sensitivity.CREDENTIAL,
        _c(r"(?i)\b(?:authorization|proxy-authorization)\s*[:=]\s*(?:bearer|basic|token)\s+([A-Za-z0-9._\-+/=]{8,})"),
        group=1,
    ),
    Rule(
        "ASSIGNED_SECRET",
        Sensitivity.CREDENTIAL,
        _c(
            r"(?i)\b(?:api[_\-]?key|apikey|secret[_\-]?key|secret|password|passwd|pwd|token|"
            r"access[_\-]?key|private[_\-]?key|client[_\-]?secret|auth[_\-]?token)\b"
            r"\s*[:=]\s*[\"']?([^\s\"',;}\)]{6,})[\"']?"
        ),
        group=1,
    ),
    # --- PII ---
    Rule("EMAIL", Sensitivity.PII, _c(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    Rule("CREDIT_CARD", Sensitivity.FINANCIAL, _c(r"\b\d(?:[ \-]?\d){12,18}\b")),
    Rule("PHONE", Sensitivity.PII, _c(r"(?<![\w.])\+\d{1,3}[ \-]?\(?\d{2,4}\)?[ \-]?\d{3,4}[ \-]?\d{3,4}(?![\w.])")),
    # --- network ---
    Rule("PUBLIC_IP", Sensitivity.NETWORK, _c(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)

# Categories a policy may keep. Credentials are deliberately absent -- there is
# no supported configuration in which a credential is forwarded.
_OPTIONAL = {Sensitivity.PII, Sensitivity.NETWORK, Sensitivity.FINANCIAL}


@dataclass(frozen=True)
class RedactionPolicy:
    """Per-tool tuning. Credentials are never negotiable."""

    redact_pii: bool = True
    redact_financial: bool = True
    # Off by default: in an incident log, addresses and CIDRs are the evidence.
    # Private ranges are never redacted regardless of this flag.
    redact_public_ips: bool = False
    # Entropy backstop for keys in formats we do not have a rule for.
    entropy_scan: bool = True
    entropy_threshold: float = 4.0
    entropy_min_length: int = 24

    def allows(self, sensitivity: Sensitivity) -> bool:
        if sensitivity is Sensitivity.CREDENTIAL:
            return False
        if sensitivity is Sensitivity.PII:
            return not self.redact_pii
        if sensitivity is Sensitivity.FINANCIAL:
            return not self.redact_financial
        if sensitivity is Sensitivity.NETWORK:
            return not self.redact_public_ips
        return False


@lru_cache(maxsize=1)
def _custom_rules_cached(spec: str) -> tuple[Rule, ...]:
    """Compile once per distinct spec.

    `redact()` runs on every request over inputs up to megabytes; recompiling
    the operator's patterns each time would be pure waste. Keyed on the raw
    spec so a changed environment variable still takes effect on reload.
    """
    return _parse_custom_rules(spec)


def load_custom_rules() -> tuple[Rule, ...]:
    import os

    return _custom_rules_cached(os.getenv("REDACTION_PATTERNS", ""))


def _parse_custom_rules(raw: str) -> tuple[Rule, ...]:
    """Operator-supplied patterns from REDACTION_PATTERNS.

    Format: `NAME=<regex>` entries separated by `;;`. Every organisation has
    identifier shapes this module has never heard of -- internal ticket
    formats, employee IDs, customer account numbers, proprietary project
    codenames. A detector that only knows public credential formats is not a
    DLP solution, and this is the seam that lets it get closer to one without
    a code change.

    A malformed pattern is skipped with a warning rather than crashing startup:
    failing to boot because of a bad regex is worse than running with one fewer
    rule, and the startup log names it.
    """
    import logging

    if not raw.strip():
        return ()

    log = logging.getLogger("copilot.redaction")
    rules: list[Rule] = []
    for entry in raw.split(";;"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        name, _, pattern = entry.partition("=")
        name = name.strip().upper().replace(" ", "_") or "CUSTOM"
        try:
            compiled = re.compile(pattern.strip())
        except re.error as exc:
            log.warning("Ignoring invalid REDACTION_PATTERNS entry %r: %s", name, exc)
            continue
        rules.append(Rule(f"CUSTOM_{name}", Sensitivity.CREDENTIAL, compiled))
    if rules:
        log.info("Loaded %d custom redaction pattern(s): %s", len(rules), [r.kind for r in rules])
    return tuple(rules)


def fail_closed() -> bool:
    """When true, a request carrying credentials is refused rather than redacted.

    Redaction is best-effort pattern matching, and best-effort is the wrong
    posture for some material. An operator who would rather lose the request
    than risk an unrecognised secret shape reaching a third party sets
    REDACTION_FAIL_CLOSED=true, and any detected credential becomes a refusal.
    """
    import os

    return (os.getenv("REDACTION_FAIL_CLOSED", "false") or "").strip().lower() in {"1", "true", "yes", "on"}


# Policy presets. Logs and transcripts need different things kept.
POLICY_STRICT = RedactionPolicy()
POLICY_LOGS = RedactionPolicy(redact_pii=True, redact_public_ips=False)
POLICY_CODE = RedactionPolicy(redact_pii=False, redact_public_ips=False, entropy_threshold=4.2)


@dataclass
class Finding:
    kind: str
    sensitivity: Sensitivity
    start: int
    end: int
    placeholder: str
    # The raw value is kept in-process only so callers can assert on it in
    # tests and so identical values collapse to one placeholder. It is never
    # serialized -- `RedactionReport.public()` is what leaves this module.
    value: str = field(repr=False, default="")


@dataclass
class RedactionReport:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def redacted_count(self) -> int:
        return len(self.findings)

    @property
    def has_credentials(self) -> bool:
        return any(f.sensitivity is Sensitivity.CREDENTIAL for f in self.findings)

    def counts_by_kind(self) -> dict[str, int]:
        return dict(Counter(f.kind for f in self.findings))

    def public(self) -> dict:
        """Safe to return over the API: what was removed, never the values."""
        return {
            "redacted_count": self.redacted_count,
            "by_kind": self.counts_by_kind(),
            "contained_credentials": self.has_credentials,
        }


# --- private network ranges, which stay visible ---------------------------


def _is_private_ip(text: str) -> bool:
    parts = text.split(".")
    if len(parts) != 4:
        return True  # not a real dotted quad; leave it alone
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return True
    if any(o > 255 for o in octets):
        return True  # e.g. a version string like 10.300.1.1
    a, b = octets[0], octets[1]
    return (
        a == 10
        or a == 127
        or (a == 172 and 16 <= b <= 31)
        or (a == 192 and b == 168)
        or (a == 169 and b == 254)
        or a == 0
    )


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


_ENTROPY_CANDIDATE = re.compile(r"[A-Za-z0-9+/_\-=]{24,}")
# Only treat a high-entropy blob as a secret when it sits where a secret sits.
_ENTROPY_CONTEXT = re.compile(
    r"(?i)(?:key|token|secret|password|auth|credential|bearer|signature)"
    r"[\w \t:=\-'\"]{0,24}$"
)


def _luhn_ok(digits: str) -> bool:
    """Keeps long IDs (trace IDs, order numbers) from being called card numbers."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _accept(rule: Rule, value: str, text: str, start: int) -> bool:
    """Per-rule false-positive suppression."""
    if rule.kind == "PUBLIC_IP":
        return not _is_private_ip(value)
    if rule.kind == "CREDIT_CARD":
        return _luhn_ok(value)
    if rule.kind == "ASSIGNED_SECRET":
        # Config templates and docs are not leaks.
        low = value.lower().strip("\"'")
        placeholders = {
            "none", "null", "true", "false", "changeme", "your_key_here", "xxx",
            "replace_me", "redacted", "example", "todo", "placeholder", "secret",
        }
        if low in placeholders or low.startswith(("<", "${", "{{", "$(", "os.environ", "process.env")):
            return False
        return len(value) >= 6
    return True


def redact(text: str, policy: RedactionPolicy = POLICY_STRICT) -> RedactionReport:
    """Replace secrets and PII with stable placeholders.

    Identical values collapse to the same placeholder so that relationships in
    the source survive into whatever reads the redacted text.
    """
    if not text:
        return RedactionReport(text=text or "")

    spans: list[tuple[int, int, Rule, str]] = []

    # `claimed` is kept sorted by start and holds mutually non-overlapping
    # intervals, so an overlap test is two bisect lookups rather than a scan of
    # everything claimed so far. The scan version was O(matches^2): an input
    # with thousands of hits -- a log full of tokens, or an operator pattern
    # that matches every line -- took quadratic time inside the security layer,
    # which is a denial-of-service vector in the component meant to prevent one.
    claimed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        index = bisect.bisect_right(claimed, (start, end))
        if index > 0 and claimed[index - 1][1] > start:
            return True
        return index < len(claimed) and claimed[index][0] < end

    def commit(accepted: list[tuple[int, int]]) -> None:
        if accepted:
            claimed.extend(accepted)
            claimed.sort()

    # Operator patterns run first: an organisation's own identifier shapes are
    # more specific than the generic detectors and should win the span.
    for rule in (*load_custom_rules(), *RULES):
        if policy.allows(rule.sensitivity):
            continue
        accepted: list[tuple[int, int]] = []
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            value = match.group(rule.group)
            if not value or overlaps(start, end):
                continue
            if not _accept(rule, value, text, start):
                continue
            spans.append((start, end, rule, value))
            accepted.append((start, end))
        commit(accepted)

    if policy.entropy_scan:
        entropy_rule = Rule("HIGH_ENTROPY_SECRET", Sensitivity.CREDENTIAL, _ENTROPY_CANDIDATE)
        accepted = []
        for match in _ENTROPY_CANDIDATE.finditer(text):
            start, end = match.span()
            value = match.group()
            if overlaps(start, end) or len(value) < policy.entropy_min_length:
                continue
            if shannon_entropy(value) < policy.entropy_threshold:
                continue
            if not _ENTROPY_CONTEXT.search(text[max(0, start - 40) : start]):
                continue
            spans.append((start, end, entropy_rule, value))
            accepted.append((start, end))
        commit(accepted)

    if not spans:
        return RedactionReport(text=text)

    spans.sort(key=lambda s: s[0])

    # One placeholder per distinct (kind, value) so repeats stay linkable.
    assigned: dict[tuple[str, str], str] = {}
    counters: Counter = Counter()
    findings: list[Finding] = []
    out: list[str] = []
    cursor = 0

    for start, end, rule, value in spans:
        key = (rule.kind, value)
        placeholder = assigned.get(key)
        if placeholder is None:
            counters[rule.kind] += 1
            placeholder = f"[[REDACTED:{rule.kind}:{counters[rule.kind]}]]"
            assigned[key] = placeholder
        out.append(text[cursor:start])
        out.append(placeholder)
        cursor = end
        findings.append(
            Finding(
                kind=rule.kind,
                sensitivity=rule.sensitivity,
                start=start,
                end=end,
                placeholder=placeholder,
                value=value,
            )
        )

    out.append(text[cursor:])
    return RedactionReport(text="".join(out), findings=findings)


def assert_no_secrets(text: str, report: RedactionReport) -> list[str]:
    """Belt-and-braces check that no redacted value survived into `text`.

    Used on the outbound payload right before the network call. Returns the
    kinds that leaked; an empty list is the pass condition.
    """
    leaked = []
    for finding in report.findings:
        if finding.value and finding.value in text:
            leaked.append(finding.kind)
    return sorted(set(leaked))
