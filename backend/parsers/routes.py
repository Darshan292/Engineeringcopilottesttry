"""API route extraction.

The part that earns its place is error discovery. A handler's real failure
modes live in `raise HTTPException(status_code=409, ...)` statements buried in
the body, not in the response model. Reading them out of the AST means the
generated documentation lists the 409 a caller will actually hit, with the
condition that triggers it, rather than the 200 the signature advertises.

Pydantic `Field(...)` constraints are extracted the same way, so `ge=1, le=50`
reaches the OpenAPI output as real `minimum`/`maximum` values instead of being
re-invented by the model.

FastAPI and Flask parse through `ast`. Express and other JS frameworks use a
regex pass, flagged with a lower-confidence parser name so the prompt knows to
qualify its claims.
"""

from __future__ import annotations

import ast
import re

from ..core.detect import detect_language
from ..core.ir import FieldSpec, ModelSpec, ResponseSpec, RouteIR, RouteSpec

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}

# Pydantic Field(...) keywords mapped onto their OpenAPI equivalents.
_CONSTRAINT_MAP = {
    "gt": "exclusiveMinimum",
    "ge": "minimum",
    "lt": "exclusiveMaximum",
    "le": "maximum",
    "min_length": "minLength",
    "max_length": "maxLength",
    "pattern": "pattern",
    "regex": "pattern",
    "multiple_of": "multipleOf",
    "max_items": "maxItems",
    "min_items": "minItems",
}

_PY_TO_OPENAPI = {
    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
    "bytes": "string", "dict": "object", "list": "array", "None": "null",
    "datetime": "string", "date": "string", "UUID": "string", "Decimal": "number",
}


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return None


def _openapi_type(annotation: str | None) -> tuple[str, bool]:
    """Map a Python annotation onto an OpenAPI type. Returns (type, inferred)."""
    if not annotation:
        return "string", True
    text = annotation.replace(" ", "")
    optional = "None" in text or text.startswith("Optional[")
    base = text.replace("Optional[", "").replace("|None", "").replace("None|", "").rstrip("]")

    if base.startswith("Literal["):
        return "string", False
    if base.startswith(("list[", "List[", "Sequence[", "tuple[")):
        return "array", False
    if base.startswith(("dict[", "Dict[", "Mapping[")):
        return "object", False

    mapped = _PY_TO_OPENAPI.get(base)
    if mapped:
        return mapped, False
    # An unrecognised annotation is almost always another model; say so rather
    # than silently calling it a string.
    return ("object", False) if base and base[0].isupper() else ("string", True)


def _extract_constraints(call: ast.Call) -> tuple[dict, str | None, bool, str | None]:
    """Pull constraints, default, requiredness and description out of Field(...)."""
    constraints: dict = {}
    default: str | None = None
    required = True
    description: str | None = None

    if call.args:
        first = _unparse(call.args[0])
        if first == "...":
            required = True
        elif first is not None:
            required = False
            default = first

    for keyword in call.keywords:
        if keyword.arg == "default":
            value = _unparse(keyword.value)
            if value == "...":
                required = True
            else:
                required = False
                default = value
        elif keyword.arg == "default_factory":
            required = False
            default = f"{_unparse(keyword.value)}()"
        elif keyword.arg == "description":
            description = _unparse(keyword.value)
            if description:
                description = description.strip("\"'")
        elif keyword.arg in _CONSTRAINT_MAP:
            value = _unparse(keyword.value)
            if value is not None:
                constraints[_CONSTRAINT_MAP[keyword.arg]] = value.strip("\"'")

    return constraints, default, required, description


def _parse_model(node: ast.ClassDef, model_id: str) -> ModelSpec:
    fields: list[FieldSpec] = []

    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        annotation = _unparse(item.annotation)
        field_type, inferred = _openapi_type(annotation)

        constraints: dict = {}
        default: str | None = None
        required = "None" not in (annotation or "")
        description: str | None = None

        if isinstance(item.value, ast.Call) and (_unparse(item.value.func) or "").endswith("Field"):
            constraints, default, required, description = _extract_constraints(item.value)
        elif item.value is not None:
            default = _unparse(item.value)
            required = False

        # Literal[...] enumerates the allowed values; that belongs in the spec.
        if annotation and "Literal[" in annotation:
            inner = annotation[annotation.index("Literal[") + 8 :].rstrip("]")
            constraints["enum"] = [v.strip().strip("\"'") for v in inner.split(",") if v.strip()]

        fields.append(
            FieldSpec(
                name=item.target.id,
                type=field_type,
                required=required,
                default=default,
                constraints=constraints,
                description=description,
                type_inferred=inferred,
            )
        )

    return ModelSpec(id=model_id, name=node.name, fields=fields, docstring=ast.get_docstring(node))


def _route_from_decorator(decorator: ast.Call) -> tuple[str, str, dict] | None:
    """Return (METHOD, path, decorator kwargs) for a routing decorator."""
    func = decorator.func
    if not isinstance(func, ast.Attribute) or func.attr.lower() not in _HTTP_METHODS | {"route"}:
        return None

    path = ""
    if decorator.args:
        value = _unparse(decorator.args[0])
        if value:
            path = value.strip("\"'")

    kwargs = {kw.arg: _unparse(kw.value) for kw in decorator.keywords if kw.arg}

    method = func.attr.upper()
    if method == "ROUTE":
        # Flask: @app.route("/x", methods=["POST"])
        methods = kwargs.get("methods") or "['GET']"
        found = re.findall(r"['\"](\w+)['\"]", methods)
        method = (found[0] if found else "GET").upper()

    return method, path, kwargs


def _responses_from_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ResponseSpec]:
    """Find every `raise HTTPException(status_code=..., detail=...)` in a handler.

    These are the failure modes a consumer actually hits and that hand-written
    docs consistently omit.
    """
    found: dict[int, ResponseSpec] = {}

    for child in ast.walk(node):
        if not isinstance(child, ast.Raise) or not isinstance(child.exc, ast.Call):
            continue
        name = _unparse(child.exc.func) or ""
        if "HTTPException" not in name and "abort" not in name:
            continue

        status: int | None = None
        detail = ""

        for arg in child.exc.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                status = arg.value
        for keyword in child.exc.keywords:
            value = _unparse(keyword.value)
            if keyword.arg in {"status_code", "code", "status"} and value and value.isdigit():
                status = int(value)
            elif keyword.arg == "detail" and value:
                # `detail=f"..."` unparses with the f prefix attached.
                detail = re.sub(r"^[frbu]+(?=[\"'])", "", value).strip("\"'")

        if status is not None:
            found.setdefault(status, ResponseSpec(status=status, description=detail, from_raise=True))

    return sorted(found.values(), key=lambda r: r.status)


def _params_from_handler(node: ast.FunctionDef | ast.AsyncFunctionDef, path: str):
    """Split handler arguments into path / query / header / body / auth."""
    path_names = set(re.findall(r"\{(\w+)\}", path)) | set(re.findall(r"<(?:\w+:)?(\w+)>", path))

    path_params: list[FieldSpec] = []
    query_params: list[FieldSpec] = []
    header_params: list[FieldSpec] = []
    auth: list[str] = []
    body_model: str | None = None

    args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    defaults: list[ast.expr | None] = (
        [None] * (len(node.args.posonlyargs) + len(node.args.args) - len(node.args.defaults))
        + list(node.args.defaults)
        + list(node.args.kw_defaults)
    )

    for arg, default in zip(args, defaults + [None] * (len(args) - len(defaults))):
        annotation = _unparse(arg.annotation)
        field_type, inferred = _openapi_type(annotation)
        default_text = _unparse(default)
        marker = (default_text or "")

        spec = FieldSpec(
            name=arg.arg,
            type=field_type,
            required=True,
            default=None,
            type_inferred=inferred,
        )

        if marker.startswith("Depends("):
            inner = marker[len("Depends(") : -1]
            auth.append(inner or arg.arg)
            continue

        if isinstance(default, ast.Call):
            call_name = (_unparse(default.func) or "").split(".")[-1]
            constraints, call_default, required, description = _extract_constraints(default)
            spec.constraints = constraints
            spec.required = required
            spec.default = call_default
            spec.description = description

            if call_name == "Header":
                header_params.append(spec)
                continue
            if call_name == "Path":
                path_params.append(spec)
                continue
            if call_name in {"Query", "Cookie", "Form"}:
                query_params.append(spec)
                continue
            if call_name == "Body":
                body_model = annotation
                continue

        if arg.arg in path_names:
            spec.required = True
            path_params.append(spec)
        elif annotation and annotation[0].isupper() and field_type == "object":
            # An un-defaulted model annotation is the request body in FastAPI.
            body_model = annotation
        else:
            spec.required = default is None
            spec.default = default_text
            query_params.append(spec)

    return path_params, query_params, header_params, body_model, auth


def parse_python_routes(source: str) -> RouteIR:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return RouteIR(
            framework="unknown",
            parser="ast (failed)",
            syntax_error=f"line {exc.lineno}: {exc.msg}",
            source=source,
        )

    framework = "unknown"
    lowered = source.lower()
    if "fastapi" in lowered:
        framework = "FastAPI"
    elif "flask" in lowered:
        framework = "Flask"
    elif "django" in lowered:
        framework = "Django"

    # Router prefix, e.g. APIRouter(prefix="/v1/deployments")
    base_prefix = ""
    prefix_match = re.search(r"APIRouter\([^)]*prefix\s*=\s*['\"]([^'\"]+)['\"]", source)
    if prefix_match:
        base_prefix = prefix_match.group(1)
    else:
        bp = re.search(r"Blueprint\([^)]*url_prefix\s*=\s*['\"]([^'\"]+)['\"]", source)
        if bp:
            base_prefix = bp.group(1)

    models: list[ModelSpec] = []
    routes: list[RouteSpec] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = {_unparse(b) or "" for b in node.bases}
            if any("BaseModel" in b or "Schema" in b for b in bases):
                models.append(_parse_model(node, f"M{len(models) + 1}"))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            parsed = _route_from_decorator(decorator)
            if parsed is None:
                continue
            method, path, kwargs = parsed

            path_params, query_params, header_params, body_model, auth = _params_from_handler(node, path)
            responses = _responses_from_body(node)

            # FastAPI and Flask both default to 200 for every method; only an
            # explicit status_code changes that. Guessing 201 for POST put a
            # status into the docs that the service does not actually return.
            success = 200
            declared = (kwargs.get("status_code") or "").strip()
            if declared.isdigit():
                success = int(declared)

            response_model = (kwargs.get("response_model") or "").strip() or None

            full_path = f"{base_prefix}{path}" if base_prefix and not path.startswith(base_prefix) else (path or base_prefix)

            routes.append(
                RouteSpec(
                    id=f"R{len(routes) + 1}",
                    method=method,
                    path=full_path or "/",
                    handler=node.name,
                    summary=(kwargs.get("summary") or "").strip("\"'") or None,
                    docstring=ast.get_docstring(node),
                    path_params=path_params,
                    query_params=query_params,
                    header_params=header_params,
                    body_model=body_model,
                    response_model=response_model,
                    success_status=success,
                    responses=responses,
                    auth_dependencies=auth,
                    line_start=node.lineno,
                )
            )

    return RouteIR(
        framework=framework,
        parser="python-ast",
        base_prefix=base_prefix,
        routes=routes,
        models=models,
        source=source,
    )


# --- JavaScript / Express fallback ----------------------------------------

_EXPRESS_ROUTE = re.compile(
    r"\b(?:app|router)\.(?P<method>get|post|put|patch|delete)\s*\(\s*['\"`](?P<path>[^'\"`]+)['\"`]",
    re.I,
)
_EXPRESS_STATUS = re.compile(r"\.status\(\s*(\d{3})\s*\)")


def parse_js_routes(source: str) -> RouteIR:
    routes: list[RouteSpec] = []
    matches = list(_EXPRESS_ROUTE.finditer(source))

    for index, match in enumerate(matches, start=1):
        path = match.group("path")
        end = matches[index].start() if index < len(matches) else len(source)
        body = source[match.end() : end]

        statuses = sorted({int(s) for s in _EXPRESS_STATUS.findall(body)})
        responses = [
            ResponseSpec(status=s, description="status set in handler", from_raise=True)
            for s in statuses
            if s >= 400
        ]
        success = next((s for s in statuses if s < 400), 200)

        routes.append(
            RouteSpec(
                id=f"R{index}",
                method=match.group("method").upper(),
                path=path,
                handler=f"handler_{index}",
                path_params=[
                    FieldSpec(name=name, type="string", required=True, type_inferred=True)
                    for name in re.findall(r":(\w+)", path)
                ],
                success_status=success,
                responses=responses,
                auth_dependencies=re.findall(r"\b(authenticate|requireAuth|verifyToken|passport\.\w+)", body)[:3],
                line_start=source[: match.start()].count("\n") + 1,
            )
        )

    return RouteIR(
        framework="Express" if routes else "unknown",
        parser="regex-heuristic (javascript)",
        routes=routes,
        source=source,
    )


def parse_routes(source: str, language: str | None = None) -> RouteIR:
    resolved = language or detect_language(source).value
    if resolved == "python":
        return parse_python_routes(source)
    if resolved in {"javascript", "typescript"}:
        return parse_js_routes(source)
    return RouteIR(
        framework="unknown",
        parser="none",
        source=source,
        syntax_error=f"No route parser for '{resolved}'; the model receives raw source.",
    )
