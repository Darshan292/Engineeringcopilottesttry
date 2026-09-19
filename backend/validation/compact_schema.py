"""Render a pydantic model as a compact type sketch instead of JSON Schema.

Full JSON Schema is the obvious thing to put in a prompt and the wrong thing.
Measured on this project's own models it was 43-64% of the entire system
prompt: `{"title": "Summary", "type": "string", "maxLength": 4000}` spends a
dozen tokens saying what `"summary": string` says in three, and `$defs` with
`$ref` indirection forces the model to resolve pointers before it can see the
shape it must produce.

That matters because the free tier's binding constraint is tokens per minute,
not requests per minute. On an 8,000 TPM ceiling a 3,600-token system
prompt means a single repair attempt can exhaust a minute's budget, and the
user sees a 429 that looks like the app is broken.

This renders the same contract as annotated JSON-ish pseudo-syntax: nested
objects are inlined, enums are shown as literal unions, and only descriptions
that carry information a name cannot are kept. Precision is not lost where it
counts, because the real contract is enforced by pydantic after the response
arrives, not by the prompt.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Keys that constrain a value in a way worth telling the model about. Anything
# not listed is enforced on validation and does not need prompt space.
_USEFUL_CONSTRAINTS = ("minLength", "maxLength", "minimum", "maximum", "pattern")


def _resolve(node: dict, defs: dict) -> dict:
    """Follow a single `$ref` into `$defs`."""
    ref = node.get("$ref")
    if not ref:
        return node
    name = ref.rsplit("/", 1)[-1]
    resolved = dict(defs.get(name, {}))
    # Preserve sibling keys such as `description` that sit next to the $ref.
    for key, value in node.items():
        if key != "$ref":
            resolved.setdefault(key, value)
    return resolved


def _render(node: dict, defs: dict, indent: int, required: bool) -> str:
    node = _resolve(node, defs)
    pad = "  " * indent

    # anyOf is how pydantic expresses `X | None` and unions.
    if "anyOf" in node:
        options = [o for o in node["anyOf"] if o.get("type") != "null"]
        nullable = len(options) != len(node["anyOf"])
        if len(options) == 1:
            rendered = _render(options[0], defs, indent, required)
            return f"{rendered} | null" if nullable else rendered
        parts = [_render(o, defs, indent, required) for o in options]
        return " | ".join(parts) + (" | null" if nullable else "")

    if "const" in node:
        return repr(node["const"])

    if "enum" in node:
        return " | ".join(f'"{v}"' for v in node["enum"])

    kind = node.get("type")

    if kind == "object" and "properties" in node:
        return _render_object(node, defs, indent)

    if kind == "array":
        item = node.get("items") or {}
        return f"[{_render(item, defs, indent, True)}, ...]"

    if kind == "object":
        return "object"

    mapping = {"string": "string", "integer": "int", "number": "number", "boolean": "bool", "null": "null"}
    return mapping.get(kind, "any")


def _annotation(node: dict, defs: dict, name: str, required: bool) -> str:
    """The trailing comment for one field, or empty."""
    node = _resolve(node, defs)
    bits: list[str] = []

    if not required:
        default = node.get("default")
        bits.append("optional" if default is None else f"optional, default {default!r}")

    for key in _USEFUL_CONSTRAINTS:
        if key in node:
            bits.append(f"{key} {node[key]}")

    description = (node.get("description") or "").strip()
    if description:
        # A description that merely restates the field name is noise.
        condensed = description.replace("\n", " ")
        if condensed.lower().rstrip(".").replace(" ", "_") != name.lower():
            bits.append(condensed)

    return ("  // " + "; ".join(bits)) if bits else ""


def _render_object(node: dict, defs: dict, indent: int) -> str:
    pad = "  " * indent
    inner = "  " * (indent + 1)
    required = set(node.get("required") or [])

    lines = ["{"]
    for name, prop in (node.get("properties") or {}).items():
        is_required = name in required
        rendered = _render(prop, defs, indent + 1, is_required)
        comment = _annotation(prop, defs, name, is_required)
        lines.append(f'{inner}"{name}": {rendered},{comment}')
    lines.append(pad + "}")
    return "\n".join(lines)


def compact_schema(model: type[BaseModel]) -> str:
    """A prompt-sized rendering of `model`'s contract."""
    schema: dict[str, Any] = model.model_json_schema()
    defs = schema.get("$defs", {})
    return _render_object(schema, defs, 0)
