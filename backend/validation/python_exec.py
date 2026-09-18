"""Actually run the generated tests.

"Runnable tests" is a claim, and until something runs them it is the model's
claim rather than a fact. This writes the original source and the generated
test file into a temporary directory and executes pytest against them in a
subprocess.

Three outcomes, all useful and all different:

- **collection error** -- the file does not import. The test suite is not
  merely wrong, it is not a test suite. This is the single most common failure
  in generated tests and is invisible without execution.
- **failures** -- the tests run and disagree with the code. Sometimes the test
  is wrong; sometimes it found a real bug. Both are reported, with the failure
  text, so the repair pass can act and a human can judge.
- **pass** -- the suite runs green. Now "runnable" is established rather than
  asserted.

Safety. This executes model-generated code, which is a real risk and is treated
as one. It runs in a subprocess under a wall-clock timeout, in an isolated
temp directory that is deleted afterwards, with `-I` so the ambient environment
does not leak in, with `-p no:cacheprovider` so nothing is written back, and
under CPU and address-space limits via `setrlimit` where the platform supports
it. That is meaningful containment, not a sandbox: it does not block network
access or filesystem writes outside the temp directory. It is therefore
**opt-out** via `ENABLE_TEST_EXECUTION=false`, and anyone running untrusted
input through this should turn it off or run the service in a container.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 30
_CPU_SECONDS = 20
_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


@dataclass
class ExecutionReport:
    ran: bool = False
    skipped_reason: str | None = None
    syntax_ok: bool = False
    collected: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    failure_details: list[str] = field(default_factory=list)
    collection_error: str | None = None
    stdout_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.ran and self.syntax_ok and self.collected > 0 and self.failed == 0 and self.errors == 0

    def repair_feedback(self) -> str:
        if not self.syntax_ok:
            return f"The generated test file has a syntax error and cannot run: {self.collection_error}"
        if self.collection_error:
            return (
                f"pytest could not collect the generated tests: {self.collection_error} "
                f"Fix the imports and module references so the file runs against the source as given."
            )
        if self.failed or self.errors:
            details = "\n".join(self.failure_details[:6])
            return (
                f"{self.failed} test(s) failed and {self.errors} errored when actually executed:\n{details}\n"
                f"Either the test's expectation is wrong, or it found a real defect. Correct the tests "
                f"that encode a wrong expectation, and for any that reveal a genuine bug in the source, "
                f"keep the test and note it under 'untestable' as a suspected defect."
            )
        if self.collected == 0:
            return "The file contains no tests pytest can collect. Test functions must start with 'test_'."
        return ""

    def public(self) -> dict:
        return {
            "executed": self.ran,
            "skipped_reason": self.skipped_reason,
            "syntax_ok": self.syntax_ok,
            "collected": self.collected,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
            "collection_error": self.collection_error,
            "failures": self.failure_details[:5],
        }


def execution_enabled() -> bool:
    return os.getenv("ENABLE_TEST_EXECUTION", "true").strip().lower() not in {"false", "0", "no", "off"}


def _limit_resources() -> None:  # pragma: no cover - child process only
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (_ADDRESS_SPACE_BYTES, _ADDRESS_SPACE_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        # Not available on every platform; the timeout remains the backstop.
        pass


_SUMMARY_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")
# `-q --no-header` suppresses the "collected N items" line, so the count has to
# be reconstructed from the summary; without this every run reported collected=0
# and the repair feedback said "no tests" even when tests had failed.
_COLLECTED_RE = re.compile(r"collected (\d+) item")


def _guess_module_name(source: str) -> str:
    """Name the source file after whatever the tests try to import from it.

    Generated tests import from a module name the model invented (`pricing`,
    `calculator`, `main`). Writing the source under that name is what turns an
    otherwise-guaranteed ImportError into a real run.
    """
    return "source_under_test"


def _rewrite_imports(test_code: str, defined_names: list[str], module_name: str) -> tuple[str, list[str]]:
    """Point the generated test's imports at the module we actually wrote.

    Returns the rewritten code and a note of what was changed, so the report
    can say the run required a rewrite rather than implying the file was
    runnable as produced.
    """
    notes: list[str] = []
    rewritten = test_code

    # from <anything> import X, Y  ->  from source_under_test import X, Y
    # only when every imported name is something the source actually defines.
    def replace(match: re.Match) -> str:
        module, names = match.group(1), match.group(2)
        if module == module_name:
            return match.group(0)
        imported = [n.strip().split(" as ")[0].strip() for n in names.split(",")]
        if imported and all(n in defined_names for n in imported if n and n != "*"):
            notes.append(f"rewrote 'from {module} import ...' to '{module_name}'")
            return f"from {module_name} import {names}"
        return match.group(0)

    rewritten = re.sub(r"^from ([\w.]+) import ([^\n]+)$", replace, rewritten, flags=re.M)

    # import <module>  ->  import source_under_test as <module>
    def replace_plain(match: re.Match) -> str:
        module = match.group(1)
        if module in {"pytest", "unittest", "sys", "os", "re", "json", "math", "datetime", "typing", module_name}:
            return match.group(0)
        if module.split(".")[0] in {"mock", "unittest"}:
            return match.group(0)
        notes.append(f"rewrote 'import {module}' to '{module_name}'")
        return f"import {module_name} as {module}"

    rewritten = re.sub(r"^import ([\w.]+)$", replace_plain, rewritten, flags=re.M)
    return rewritten, sorted(set(notes))


def run_python_tests(
    source_code: str,
    test_code: str,
    defined_names: list[str] | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[ExecutionReport, list[str]]:
    """Execute generated tests against the source. Returns (report, rewrite notes)."""
    report = ExecutionReport()

    if not execution_enabled():
        report.skipped_reason = "ENABLE_TEST_EXECUTION is false."
        return report, []

    # Refuse to run something that will not even compile; report it as the
    # hard failure it is rather than paying for a subprocess to find out.
    try:
        compile(test_code, "<generated_test>", "exec")
        report.syntax_ok = True
    except SyntaxError as exc:
        report.collection_error = f"line {exc.lineno}: {exc.msg}"
        return report, []

    module_name = _guess_module_name(source_code)
    rewritten, notes = _rewrite_imports(test_code, defined_names or [], module_name)

    workdir = Path(tempfile.mkdtemp(prefix="copilot-testrun-"))
    try:
        (workdir / f"{module_name}.py").write_text(source_code, encoding="utf-8")
        (workdir / "test_generated.py").write_text(rewritten, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(workdir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(workdir),
            # Generated tests should not reach the network; this does not
            # enforce it, but it stops well-behaved libraries from trying.
            "no_proxy": "*",
            "NO_PROXY": "*",
        }

        import time

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    sys.executable, "-I", "-m", "pytest",
                    "test_generated.py",
                    "-p", "no:cacheprovider",
                    "-q", "--tb=short", "--no-header",
                    "-W", "ignore::DeprecationWarning",
                ],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
            output = f"{completed.stdout}\n{completed.stderr}"
        except subprocess.TimeoutExpired:
            report.ran = True
            report.duration_seconds = timeout
            report.collection_error = (
                f"Execution exceeded {timeout}s and was killed. The generated tests likely "
                f"contain an infinite loop or a blocking call."
            )
            return report, notes
        except (OSError, ValueError) as exc:
            report.skipped_reason = f"Could not start the test subprocess: {exc}"
            return report, notes

        report.ran = True
        report.duration_seconds = time.perf_counter() - started
        report.stdout_tail = output[-4000:]

        collected = _COLLECTED_RE.search(output)
        if collected:
            report.collected = int(collected.group(1))

        skipped = 0
        for count, label in _SUMMARY_RE.findall(output):
            number = int(count)
            if label == "passed":
                report.passed = number
            elif label == "failed":
                report.failed = number
            elif label.startswith("error"):
                report.errors = number
            elif label == "skipped":
                skipped = number

        if not report.collected:
            report.collected = report.passed + report.failed + report.errors + skipped

        if "ERROR collecting" in output or "ImportError" in output or "ModuleNotFoundError" in output:
            for line in output.split("\n"):
                if any(marker in line for marker in ("ImportError", "ModuleNotFoundError", "ERROR collecting")):
                    report.collection_error = line.strip()[:400]
                    break

        # Pull out the assertion lines, which are what a repair pass needs.
        details: list[str] = []
        current: list[str] = []
        for line in output.split("\n"):
            if line.startswith("____") or line.startswith("===="):
                if current:
                    details.append("\n".join(current[:8]))
                    current = []
                if line.startswith("____"):
                    current = [line.strip("_ ")]
            elif current:
                current.append(line)
        if current:
            details.append("\n".join(current[:8]))
        report.failure_details = [d for d in details if d.strip()][:8]

        if report.collected == 0 and not report.collection_error and report.passed == 0:
            report.collection_error = "pytest collected no tests from the generated file."

        return report, notes
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
