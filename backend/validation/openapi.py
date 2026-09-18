"""OpenAPI validation.

"Valid-looking YAML" and "a valid OpenAPI document" are different claims, and
only the second one is useful to someone generating a client from it. This
parses the YAML and validates the result against the OpenAPI 3.0/3.1 schema
with `openapi-spec-validator` (Apache-2.0, bundles its schemas, works offline).

It also cross-checks the spec against the routes the parser found, which
catches the failure the schema validator cannot see: a syntactically perfect
document that documents endpoints the code does not have, or omits ones it
does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml
from openapi_spec_validator import validate
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError


@dataclass
class OpenAPIReport:
    parsed: bool = False
    schema_valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    documented_operations: list[str] = field(default_factory=list)
    missing_routes: list[str] = field(default_factory=list)
    extra_routes: list[str] = field(default_factory=list)
    version: str | None = None

    @property
    def ok(self) -> bool:
        return self.parsed and self.schema_valid and not self.missing_routes and not self.extra_routes

    def repair_feedback(self) -> str:
        """A precise instruction for the repair pass."""
        parts: list[str] = []
        if not self.parsed:
            parts.append(f"The openapi_yaml field is not parseable YAML: {'; '.join(self.errors)}")
        elif not self.schema_valid:
            parts.append(f"The YAML parses but is not a valid OpenAPI document: {'; '.join(self.errors[:5])}")
        if self.missing_routes:
            parts.append(
                f"These routes exist in the source but are missing from paths: {', '.join(self.missing_routes)}"
            )
        if self.extra_routes:
            parts.append(
                f"These operations are in the spec but do not exist in the source, so remove them: "
                f"{', '.join(self.extra_routes)}"
            )
        return " ".join(parts)

    def public(self) -> dict:
        return {
            "parsed": self.parsed,
            "schema_valid": self.schema_valid,
            "openapi_version": self.version,
            "operations_documented": len(self.documented_operations),
            "errors": self.errors[:10],
            "warnings": self.warnings[:10],
            "routes_missing_from_spec": self.missing_routes,
            "routes_in_spec_but_not_in_code": self.extra_routes,
        }


def _normalize(method: str, path: str) -> str:
    return f"{method.upper()} {path.rstrip('/') or '/'}"


def validate_openapi(text: str, route_ir=None) -> OpenAPIReport:
    """Parse, schema-validate, and cross-check against extracted routes."""
    report = OpenAPIReport()

    body = (text or "").strip()
    # Models frequently wrap the document in a fence despite being asked not to.
    if body.startswith("```"):
        lines = body.split("\n")
        body = "\n".join(lines[1 : -1 if lines[-1].strip().startswith("```") else None])

    try:
        spec: Any = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        report.errors.append(str(exc).replace("\n", " ")[:300])
        return report

    if not isinstance(spec, dict):
        report.errors.append(f"Top level of the document is {type(spec).__name__}, expected a mapping.")
        return report

    report.parsed = True
    report.version = str(spec.get("openapi") or spec.get("swagger") or "")

    try:
        validate(spec)
        report.schema_valid = True
    except OpenAPIValidationError as exc:
        report.errors.append(str(exc).split("\n")[0][:300])
    except Exception as exc:  # the validator raises several unrelated types
        report.errors.append(f"{type(exc).__name__}: {str(exc).split(chr(10))[0][:300]}")

    paths = spec.get("paths") or {}
    if isinstance(paths, dict):
        for path, operations in paths.items():
            if not isinstance(operations, dict):
                continue
            for method in operations:
                if method.lower() in {"get", "post", "put", "patch", "delete", "head", "options", "trace"}:
                    report.documented_operations.append(_normalize(method, str(path)))

    if not report.documented_operations:
        report.warnings.append("The spec documents no operations.")

    # Cross-check against what the parser actually found in the source.
    if route_ir is not None and getattr(route_ir, "routes", None):
        actual = {_normalize(r.method, r.path) for r in route_ir.routes}
        documented = set(report.documented_operations)
        report.missing_routes = sorted(actual - documented)
        report.extra_routes = sorted(documented - actual)

    return report
