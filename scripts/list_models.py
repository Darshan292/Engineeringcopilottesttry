#!/usr/bin/env python3
"""Ask the configured provider what this key can actually call.

Run it before setting `LLM_MODEL`, and again whenever a model starts returning
404. No list compiled into this application can stay true -- providers retire chat
models every few months, and OpenRouter's free variants appear and disappear as
providers donate and withdraw capacity -- so the only trustworthy answer comes
from your own key.

    .venv/bin/python scripts/list_models.py           # usable models
    .venv/bin/python scripts/list_models.py --free    # only the free ones
    .venv/bin/python scripts/list_models.py --all     # including unusable ones

Reads the same `.env` the server does, so what it prints is what the app will
see.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _load_dotenv() -> None:
    """Minimal .env reader, so this works without importing the whole app first."""
    path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    import os

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free", action="store_true", help="only models that cost nothing")
    parser.add_argument("--all", action="store_true", help="include models this app cannot use")
    parser.add_argument("--json", action="store_true", help="raw JSON instead of a table")
    args = parser.parse_args()

    _load_dotenv()

    from backend.config import settings
    from backend.llm_client import LLMError, list_models
    from backend.providers import api_key_env_name, detect_provider

    provider = detect_provider(settings.base_url)
    print(f"provider : {provider.label} ({settings.base_url})")
    print(f"model    : {settings.model or '(not set)'}")

    if not settings.has_api_key:
        print(
            f"\nNo API key. Put {api_key_env_name(provider)}=... in .env"
            + (f" (get one at {provider.console_keys_url})" if provider.console_keys_url else "")
        )
        return 2

    try:
        models = await list_models()
    except LLMError as exc:
        print(f"\nCould not list models: {exc.message}")
        if exc.hint:
            print(f"  {exc.hint}")
        return 1

    if args.json:
        import json

        print(json.dumps(models, indent=2))
        return 0

    rows = [m for m in models if args.all or m.get("usable")]
    if args.free:
        rows = [m for m in rows if m.get("free")]

    if not rows:
        print("\nNothing matched. Try --all to see everything the key can call.")
        return 1

    width = max(len(str(m["id"])) for m in rows)
    print(f"\n{len(rows)} model(s):\n")
    for m in rows:
        window = m.get("context_window")
        window_text = f"{window:>9,}" if isinstance(window, int) else "        ?"
        flags = []
        if m.get("free"):
            flags.append("FREE")
        if not m.get("usable"):
            flags.append(m.get("unusable_reason") or "unusable")
        suffix = f"  [{'; '.join(flags)}]" if flags else ""
        print(f"  {str(m['id']):<{width}}  {window_text} ctx{suffix}")

    usable_free = [m for m in rows if m.get("free") and m.get("usable")]
    if usable_free:
        print(f"\nSet one of these in .env, e.g.:\n  LLM_MODEL={usable_free[0]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
