"""Code structure extraction.

Python goes through the real `ast` module, so what comes out is fact, not
pattern-matching: the exact boundary conditions, the exceptions the code can
actually raise, which calls cross a module boundary and therefore need mocking,
and how many independent paths exist.

That last point is the whole argument for this layer. Asked to "think of edge
cases", a model produces plausible ones. Handed `comparisons: subtotal > 10000;
customer_tier != 'platinum'; rate > 0.30`, it is being asked a much easier
question -- write a case for each stated boundary -- and the test suite's
coverage becomes a property of the extractor rather than of the model's mood.

Other languages fall back to a signature-level regex extractor. It is honestly
weaker, `parser` and `confidence` say so, and the prompt is told not to claim
precision it does not have. Adding tree-sitter would fix this, and is the right
next step, but it is a build-time dependency this project does not currently
take.
"""

from __future__ import annotations

import ast
import re

from ..core.detect import detect_language
from ..core.ir import ClassSpec, CodeIR, FunctionSpec, ParamSpec

# Modules that ship with Python; calls into anything else cross a dependency
# boundary and are what a test should mock.
_STDLIB_PREFIXES = {
    "abc", "argparse", "array", "ast", "asyncio", "base64", "bisect", "calendar",
    "collections", "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
    "enum", "functools", "glob", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json", "logging", "math",
    "mimetypes", "operator", "os", "pathlib", "pickle", "queue", "random", "re",
    "secrets", "shutil", "signal", "socket", "sqlite3", "statistics", "string",
    "struct", "subprocess", "sys", "tempfile", "textwrap", "threading", "time",
    "types", "typing", "unicodedata", "urllib", "uuid", "warnings", "weakref", "zipfile",
}

# Calls that reach outside the process: worth flagging as side effects.
_SIDE_EFFECT_CALLS = {
    "open", "print", "input", "exec", "eval", "exit", "quit",
    "requests", "httpx", "urlopen", "subprocess", "system", "popen",
    "connect", "execute", "commit", "send", "publish", "write",
}

_MUTATING_METHODS = {"append", "extend", "insert", "pop", "remove", "clear", "update", "sort", "add", "setdefault"}


def _safe_unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - unparse is total on valid trees
        return None


class _FunctionAnalyzer(ast.NodeVisitor):
    """Walks one function body, without descending into nested functions."""

    def __init__(self, param_names: set[str], import_roots: dict[str, str]):
        self.param_names = param_names
        self.import_roots = import_roots
        self.raises: list[str] = []
        self.branches = 0
        self.loops = 0
        self.returns = 0
        self.complexity = 1
        self.calls: list[str] = []
        self.external_calls: list[str] = []
        self.comparisons: list[str] = []
        self.magic: list[str] = []
        self.mutates = False
        self.side_effects = False
        self._depth = 0

    # Nested defs are their own units; counting their branches here would
    # misattribute complexity to the parent.
    def visit_FunctionDef(self, node):
        if self._depth:
            return
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        self.branches += 1
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.branches += 1
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops += 1
        self.complexity += 1
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.loops += 1
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        if node.type is not None:
            name = _safe_unparse(node.type)
            if name:
                self.raises.append(f"catches {name}")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # `a and b and c` is two extra paths, not one.
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Match(self, node):
        self.complexity += max(0, len(node.cases) - 1)
        self.branches += len(node.cases)
        self.generic_visit(node)

    def visit_Return(self, node):
        self.returns += 1
        self.generic_visit(node)

    def visit_Raise(self, node):
        exc = node.exc
        if exc is None:
            self.raises.append("re-raise")
        else:
            target = exc.func if isinstance(exc, ast.Call) else exc
            name = _safe_unparse(target)
            if name:
                self.raises.append(name)
        self.generic_visit(node)

    def visit_Compare(self, node):
        rendered = _safe_unparse(node)
        if rendered and len(rendered) <= 90:
            self.comparisons.append(rendered)
        for operand in node.comparators:
            if isinstance(operand, ast.Constant) and not isinstance(operand.value, bool):
                literal = _safe_unparse(operand)
                if literal:
                    self.magic.append(literal)
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _safe_unparse(node.func)
        if name:
            self.calls.append(name)
            root = name.split(".")[0].split("(")[0]
            leaf = name.split(".")[-1]

            source = self.import_roots.get(root)
            if source == "third_party":
                self.external_calls.append(name)
            if root in _SIDE_EFFECT_CALLS or leaf in _SIDE_EFFECT_CALLS:
                self.side_effects = True
                if source != "stdlib":
                    self.external_calls.append(name)

            # obj.append(...) where obj is a parameter mutates the caller's data.
            if isinstance(node.func, ast.Attribute) and leaf in _MUTATING_METHODS:
                if isinstance(node.func.value, ast.Name) and node.func.value.id in self.param_names:
                    self.mutates = True
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            base = target
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id in self.param_names and base is not target:
                self.mutates = True
        self.generic_visit(node)


def _params_from_args(args: ast.arguments) -> list[ParamSpec]:
    params: list[ParamSpec] = []

    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for arg, default in zip(positional, defaults):
        params.append(
            ParamSpec(
                name=arg.arg,
                annotation=_safe_unparse(arg.annotation),
                default=_safe_unparse(default),
                kind="positional",
            )
        )
    if args.vararg:
        params.append(ParamSpec(name=f"*{args.vararg.arg}", annotation=_safe_unparse(args.vararg.annotation), kind="vararg"))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            ParamSpec(
                name=arg.arg,
                annotation=_safe_unparse(arg.annotation),
                default=_safe_unparse(default),
                kind="keyword-only",
            )
        )
    if args.kwarg:
        params.append(ParamSpec(name=f"**{args.kwarg.arg}", annotation=_safe_unparse(args.kwarg.annotation), kind="kwargs"))
    return params


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = _safe_unparse(node.args) or ""
    returns = _safe_unparse(node.returns)
    suffix = f" -> {returns}" if returns else ""
    return f"{prefix} {node.name}({args}){suffix}"


def _build_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    func_id: str,
    qualname: str,
    import_roots: dict[str, str],
    source_lines: list[str],
) -> FunctionSpec:
    params = _params_from_args(node.args)
    param_names = {p.name.lstrip("*") for p in params}

    analyzer = _FunctionAnalyzer(param_names, import_roots)
    for child in node.body:
        analyzer.visit(child)

    start, end = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
    source = "\n".join(source_lines[start - 1 : end])

    return FunctionSpec(
        id=func_id,
        name=node.name,
        qualname=qualname,
        signature=_signature(node),
        params=params,
        returns=_safe_unparse(node.returns),
        docstring=ast.get_docstring(node),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=[d for d in (_safe_unparse(d) for d in node.decorator_list) if d],
        line_start=start,
        line_end=end,
        raises=analyzer.raises,
        branch_count=analyzer.branches,
        loop_count=analyzer.loops,
        return_count=analyzer.returns,
        cyclomatic_complexity=analyzer.complexity,
        calls=sorted(set(analyzer.calls))[:40],
        external_calls=sorted(set(analyzer.external_calls))[:20],
        comparisons=analyzer.comparisons[:20],
        magic_numbers=sorted(set(analyzer.magic))[:20],
        mutates_arguments=analyzer.mutates,
        has_side_effects=analyzer.side_effects,
        source=source,
    )


def _collect_imports(tree: ast.Module) -> tuple[list[str], list[str], dict[str, str]]:
    imports: list[str] = []
    third_party: list[str] = []
    roots: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                root = alias.name.split(".")[0]
                bound = (alias.asname or alias.name).split(".")[0]
                kind = "stdlib" if root in _STDLIB_PREFIXES else "third_party"
                roots[bound] = kind
                if kind == "third_party":
                    third_party.append(root)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            # A relative import is first-party by definition.
            kind = "local" if node.level else ("stdlib" if root in _STDLIB_PREFIXES else "third_party")
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
                roots[alias.asname or alias.name] = kind
            if kind == "third_party" and root:
                third_party.append(root)

    return sorted(set(imports)), sorted(set(third_party)), roots


def parse_python(source: str) -> CodeIR:
    lines = source.split("\n")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CodeIR(
            language="python",
            parser="ast (failed)",
            confidence=0.0,
            syntax_error=f"line {exc.lineno}: {exc.msg}",
            total_lines=len(lines),
            source=source,
        )

    imports, third_party, roots = _collect_imports(tree)

    functions: list[FunctionSpec] = []
    classes: list[ClassSpec] = []
    counter = 0

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counter += 1
            functions.append(_build_function(node, f"F{counter}", node.name, roots, lines))
        elif isinstance(node, ast.ClassDef):
            methods: list[FunctionSpec] = []
            attributes: list[ParamSpec] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    counter += 1
                    methods.append(
                        _build_function(item, f"F{counter}", f"{node.name}.{item.name}", roots, lines)
                    )
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attributes.append(
                        ParamSpec(
                            name=item.target.id,
                            annotation=_safe_unparse(item.annotation),
                            default=_safe_unparse(item.value),
                        )
                    )
            classes.append(
                ClassSpec(
                    id=f"C{len(classes) + 1}",
                    name=node.name,
                    bases=[b for b in (_safe_unparse(b) for b in node.bases) if b],
                    docstring=ast.get_docstring(node),
                    methods=methods,
                    attributes=attributes,
                    decorators=[d for d in (_safe_unparse(d) for d in node.decorator_list) if d],
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
                )
            )

    return CodeIR(
        language="python",
        parser="python-ast",
        confidence=1.0,
        imports=imports,
        third_party_imports=third_party,
        functions=functions,
        classes=classes,
        module_docstring=ast.get_docstring(tree),
        total_lines=len(lines),
        source=source,
    )


# --- non-Python fallback --------------------------------------------------

_GENERIC_FUNCTION_PATTERNS: dict[str, re.Pattern] = {
    "javascript": re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)"
        r"|^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name2>\w+)\s*=\s*(?:async\s*)?\((?P<args2>[^)]*)\)\s*=>",
        re.M,
    ),
    "typescript": re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*(?::\s*(?P<ret>[\w<>\[\]|, ]+))?"
        r"|^\s*(?:export\s+)?(?:const|let)\s+(?P<name2>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?\((?P<args2>[^)]*)\)\s*(?::\s*(?P<ret2>[\w<>\[\]|, ]+))?\s*=>",
        re.M,
    ),
    "go": re.compile(
        r"^func\s+(?:\(\w+\s+\*?\w+\)\s*)?(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*(?P<ret>\([^)]*\)|[\w.*\[\]]+)?",
        re.M,
    ),
    "java": re.compile(
        r"^\s*(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?"
        r"(?P<ret>[\w<>\[\], ]+?)\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)",
        re.M,
    ),
    "ruby": re.compile(r"^\s*def\s+(?P<name>[\w?!]+)\s*(?:\((?P<args>[^)]*)\))?", re.M),
}

_GENERIC_THROW = {
    "javascript": re.compile(r"\bthrow\s+new\s+(\w+)"),
    "typescript": re.compile(r"\bthrow\s+new\s+(\w+)"),
    "java": re.compile(r"\bthrow\s+new\s+(\w+)"),
    "go": re.compile(r"\berrors\.New\(|fmt\.Errorf\("),
    "ruby": re.compile(r"\braise\s+(\w+)"),
}


def parse_generic(source: str, language: str) -> CodeIR:
    """Signature-level extraction for languages without a bundled parser."""
    lines = source.split("\n")
    pattern = _GENERIC_FUNCTION_PATTERNS.get(language)
    functions: list[FunctionSpec] = []

    if pattern:
        for index, match in enumerate(pattern.finditer(source), start=1):
            name = match.group("name") or (match.groupdict().get("name2"))
            if not name:
                continue
            raw_args = match.group("args") or match.groupdict().get("args2") or ""
            line_no = source[: match.start()].count("\n") + 1

            params = []
            for piece in (p.strip() for p in raw_args.split(",")):
                if not piece:
                    continue
                # `name: type` (TS), `name type` (Go), `Type name` (Java)
                if ":" in piece:
                    pname, _, ptype = piece.partition(":")
                    params.append(ParamSpec(name=pname.strip(), annotation=ptype.strip() or None))
                elif " " in piece:
                    left, _, right = piece.rpartition(" ")
                    if language == "go":
                        params.append(ParamSpec(name=left.strip(), annotation=right.strip()))
                    else:
                        params.append(ParamSpec(name=right.strip(), annotation=left.strip()))
                else:
                    params.append(ParamSpec(name=piece))

            # Approximate the body as the lines until the next signature.
            body = source[match.end() : match.end() + 2000]
            throw_re = _GENERIC_THROW.get(language)
            raises = sorted(set(throw_re.findall(body))) if throw_re else []

            functions.append(
                FunctionSpec(
                    id=f"F{index}",
                    name=name,
                    qualname=name,
                    signature=match.group(0).strip(),
                    params=params,
                    returns=(match.groupdict().get("ret") or match.groupdict().get("ret2") or None),
                    line_start=line_no,
                    line_end=line_no,
                    raises=[r for r in raises if r],
                    branch_count=len(re.findall(r"\bif\b", body)),
                    loop_count=len(re.findall(r"\b(?:for|while)\b", body)),
                    return_count=len(re.findall(r"\breturn\b", body)),
                    cyclomatic_complexity=1 + len(re.findall(r"\b(?:if|for|while|case|catch|&&|\|\|)\b", body)),
                )
            )

    return CodeIR(
        language=language,
        parser=f"regex-heuristic ({language})",
        # Deliberately low: the prompt reads this and is told to state that the
        # extraction is approximate rather than assert facts it cannot support.
        confidence=0.45 if functions else 0.15,
        functions=functions,
        total_lines=len(lines),
        source=source,
    )


def parse_code(source: str, language: str | None = None) -> CodeIR:
    """Entry point: detect the language if not given, then extract."""
    resolved = language or detect_language(source).value
    if resolved == "python":
        ir = parse_python(source)
        # A confident Python detection that will not parse is worth saying out
        # loud; falling back silently would hide a real syntax error.
        if ir.syntax_error:
            return ir
        return ir
    if resolved in _GENERIC_FUNCTION_PATTERNS:
        return parse_generic(source, resolved)
    return CodeIR(
        language=resolved,
        parser="none",
        confidence=0.0,
        total_lines=len(source.split("\n")),
        source=source,
        syntax_error=f"No structural parser available for '{resolved}'; the model receives raw source.",
    )
