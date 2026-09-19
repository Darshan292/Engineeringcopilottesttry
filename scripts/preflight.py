#!/usr/bin/env python3
"""Prove, against your own key, that this deployment works and costs nothing.

`list_models.py` reads the catalogue. This goes further and actually *calls* the
model, because the things most worth checking cannot be checked any other way:
whether the provider accepts the zero-price ceiling, whether a completion really
is billed at zero, and whether the rate-limit headers are the shape this
application assumes.

It runs the application's own client, not a parallel implementation. A preflight
that re-implements the request proves something about the preflight.

    .venv/bin/python scripts/preflight.py
    .venv/bin/python scripts/preflight.py --no-call   # catalogue checks only

Exit code is 0 only if every check passed. One real request is spent, out of
OpenRouter's 50-per-day free allowance, unless --no-call is given.

Why this exists as a script rather than a test: openrouter.ai was unreachable
from the environment this application was built in, so nothing here was ever
verified against the live API. These checks are the ones a human has to run
once, on a machine that can reach it.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _load_dotenv() -> None:
    """Minimal .env reader, so this sees exactly what the server will."""
    import os

    path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Report:
    """Collects results so every check runs before anything is reported."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        marker = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
        print(f"[{marker}] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"          {line}")

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)


async def run(make_call: bool) -> int:
    _load_dotenv()
    report = Report()

    from backend import billing, llm_client
    from backend.config import settings
    from backend.providers import api_key_env_name, detect_provider

    provider = detect_provider()
    print(f"\nProvider : {provider.label} ({provider.base_url})")
    print(f"Model    : {settings.model or '(unset)'}")
    print(f"Guard    : FREE_TIER_ONLY={'on' if billing.free_tier_only() else 'OFF'}\n")

    # --- 1. credentials ----------------------------------------------------
    if settings.api_key:
        report.add(PASS, "API key is set", f"from {api_key_env_name(provider)}")
    else:
        report.add(
            FAIL,
            "API key is set",
            f"Set {api_key_env_name(provider)} in .env. Free key, no card: "
            f"{provider.console_keys_url}",
        )
        return 1

    if not billing.free_tier_only():
        report.add(
            WARN,
            "Billing guard is enabled",
            "FREE_TIER_ONLY is off, so paid calls are permitted. Remove it from .env "
            "to restore the guarantee.",
        )
    else:
        report.add(PASS, "Billing guard is enabled")

    # --- 2. catalogue ------------------------------------------------------
    try:
        models = await llm_client.list_models()
    except Exception as exc:  # noqa: BLE001 - report anything, do not crash
        report.add(
            FAIL,
            "Provider catalogue loads",
            f"{type(exc).__name__}: {exc}\n"
            f"Without it the guard cannot confirm any model is free, so every call "
            f"is refused with 503.",
        )
        return 1

    usable = [m for m in models if m.get("usable")]
    free = [m for m in models if m.get("free")]
    report.add(
        PASS,
        "Provider catalogue loads",
        f"{len(models)} models, {len(usable)} usable here, {len(free)} free",
    )

    if not free:
        report.add(
            FAIL,
            "At least one free model is available",
            "No model in the catalogue prices every field at zero. Either the "
            "provider has withdrawn all free capacity, or the pricing fields are "
            "not being read correctly.",
        )
        return 1
    report.add(PASS, "At least one free model is available")

    # --- 3. the configured model -------------------------------------------
    suggestions = ", ".join(m["id"] for m in free[:3])
    if not settings.model:
        report.add(
            FAIL,
            "LLM_MODEL names a free model",
            f"LLM_MODEL is unset -- no default ships, because free ids get "
            f"withdrawn.\nPick one, e.g. {suggestions}",
        )
        return 1

    entry = next((m for m in models if m.get("id") == settings.model), None)
    if entry is None:
        report.add(
            FAIL,
            "LLM_MODEL names a free model",
            f"'{settings.model}' is not in the catalogue. Free ones now include: "
            f"{suggestions}",
        )
        return 1

    if settings.model in billing.ALWAYS_PAID_MODELS:
        report.add(
            FAIL,
            "LLM_MODEL names a free model",
            f"'{settings.model}' routes across the whole catalogue including paid "
            f"models, so its cost cannot be checked. Use 'openrouter/free' instead.",
        )
        return 1

    # The guard's own verdict, not a second opinion computed here. `list_models`
    # fed the raw price list to the guard as it fetched it, and the `free` flag
    # on each row is that verdict -- so this reports exactly what the app will
    # enforce rather than something that can disagree with it.
    if entry.get("free"):
        report.add(PASS, "LLM_MODEL names a free model", f"'{settings.model}' is priced at zero")
    else:
        report.add(
            FAIL,
            "LLM_MODEL names a free model",
            f"'{settings.model}' is not priced at zero by the provider, so the guard "
            f"would refuse before calling.\nFree ones now include: {suggestions}",
        )
        return 1

    window = entry.get("context_window") or 0
    if window and window >= 6_000:
        report.add(PASS, "Context window is large enough", f"{window:,} tokens")
    else:
        report.add(
            WARN,
            "Context window is large enough",
            f"{window:,} tokens. Below ~6,000 the instructions alone will not fit.",
        )

    if not make_call:
        print("\n--no-call given, so the live request was skipped.")
        return 1 if report.failed else 0

    # --- 4. a real call ----------------------------------------------------
    # The only way to know the provider accepts the zero-price ceiling, and the
    # only way to see what a completion is actually billed at.
    # `list_models` above already fed the guard the raw price list as it fetched
    # it, so the verdicts are loaded; this just confirms it agrees before paying
    # for a request.
    guard = billing.guard
    try:
        guard.assert_free(settings.model)
    except billing.BillingRefused as exc:
        report.add(FAIL, "Guard permits the configured model", f"{exc.message}")
        return 1

    before = guard.public()["observed_cost"]
    try:
        result = await llm_client.complete(
            "You are a calculator. Reply with digits only.",
            "What is 2 + 2?",
            max_tokens=16,
            max_wait_seconds=90,
        )
    except Exception as exc:  # noqa: BLE001
        hint = getattr(exc, "hint", "")
        report.add(
            FAIL,
            "A live completion succeeds",
            f"{type(exc).__name__}: {exc}" + (f"\n{hint}" if hint else ""),
        )
        return 1

    report.add(
        PASS,
        "A live completion succeeds",
        f"answered {result['text'][:40]!r} in {result['elapsed_ms']} ms "
        f"({result['usage'].get('total_tokens')} tokens)",
    )

    waited = result.get("rate_limited_seconds") or 0
    if waited:
        report.add(
            WARN,
            "No rate limit was hit",
            f"waited {waited:.1f}s across {result.get('rate_limit_retries')} retries. "
            f"That is the retry loop working, but you are close to the limit.",
        )
    else:
        report.add(PASS, "No rate limit was hit")

    # --- 5. the invoice ----------------------------------------------------
    after = guard.public()
    charged = after["observed_cost"] - before
    if after["blocked"]:
        report.add(
            FAIL,
            "The completion was billed at zero",
            f"The guard tripped: {after['blocked_reason']}\n"
            f"A free model charged money. Check your OpenRouter activity page.",
        )
    elif charged > 0:
        report.add(FAIL, "The completion was billed at zero", f"charged {charged}")
    else:
        report.add(PASS, "The completion was billed at zero", f"mode: {after['mode']}")

    print()
    if report.failed:
        print("PREFLIGHT FAILED -- fix the items marked FAIL above before using the app.")
        return 1
    print("Preflight passed. The app is configured correctly and this call cost nothing.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-call",
        action="store_true",
        help="check configuration and catalogue only; spend no request",
    )
    args = parser.parse_args()
    return asyncio.run(run(make_call=not args.no_call))


if __name__ == "__main__":
    raise SystemExit(main())
