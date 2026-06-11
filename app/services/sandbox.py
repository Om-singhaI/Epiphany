"""Step 4 — Validate: statistical sandbox + secure code execution.

Runs a *real* statistical test — chosen adaptively from the data (Chi-Square,
Welch's t-test, ANOVA, or Pearson correlation; see
:mod:`app.services.statistics`) — using SciPy on the actual rows of the dataset.
There is no synthetic data: the p-value is always computed from the values the
agent pulled from Elastic or the local file.

* :func:`run_statistical_test_isolated` — spawns
  :mod:`app.services.sandbox_worker` as a **separate, hardened process** (no
  network egress, CPU/memory limits) so validation can never take down or
  exfiltrate from the API process, then falls back to in-process execution only
  if the worker cannot be spawned.

Security layer (Phase 4)
========================

Because the ADK agent now *dynamically generates* hypothesis/model code, that
code is untrusted. Before any of it is executed (or deployed), it is screened
by an :class:`ast`-based scanner that:

* **blocks** imports of system-level modules (``os``, ``sys``, ``subprocess``,
  ``shutil``, ``socket``, ...), and dangerous builtins (``eval``, ``exec``,
  ``__import__``, ``open``, ...);
* **allows** only vetted data-science libraries (``pandas``, ``numpy``,
  ``scipy`` — plus ``sklearn``/``xgboost`` for generated model scripts).

:func:`run_sandboxed_code` then executes vetted snippets under a *severely
restricted* ``globals()`` (no app state, env, or service clients) wrapped in an
``asyncio.wait_for`` timeout, so a malicious payload or infinite loop yields a
safe error instead of crashing the API.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger("epiphany.sandbox")


# ─────────────────────────────────────────────────────────────────────────
# Phase 4 — AST-based security layer for dynamically generated code
# ─────────────────────────────────────────────────────────────────────────


class SecurityError(Exception):
    """Raised when scanned code violates the sandbox security policy."""


# System-level modules that must NEVER be importable from generated code.
# This is a hard denylist: a match fails the scan regardless of the allowlist.
BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "os", "sys", "subprocess", "shutil", "socket", "ctypes", "signal",
        "resource", "importlib", "builtins", "pickle", "marshal", "shelve",
        "threading", "multiprocessing", "asyncio", "pathlib", "glob",
        "tempfile", "fcntl", "pty", "mmap", "http", "urllib", "urllib2",
        "requests", "ftplib", "smtplib", "telnetlib", "webbrowser", "ssl",
        "platform", "getpass", "pwd", "grp", "code", "codeop", "runpy",
    }
)

# Builtins / names that grant code-execution or sandbox-escape capability.
BLOCKED_NAMES: frozenset[str] = frozenset(
    {
        "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
        "exit", "quit", "help", "globals", "locals", "vars", "getattr",
        "setattr", "delattr", "memoryview", "super", "type",
    }
)

# Vetted data-science libraries permitted in executable validation snippets.
DEFAULT_ALLOWED_MODULES: frozenset[str] = frozenset(
    {"pandas", "numpy", "scipy", "math", "statistics"}
)

# Broader allowlist used when *statically screening* a generated model script
# (which legitimately imports ML libraries) before it is deployed.
MODEL_ALLOWED_MODULES: frozenset[str] = DEFAULT_ALLOWED_MODULES | frozenset(
    {"sklearn", "xgboost", "lightgbm", "joblib"}
)

# The only builtins exposed to executed code. Notably excludes eval/exec/open/
# __import__/getattr/etc. — those are supplied (if at all) under tight control.
_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(__builtins__, name, None)
    if not isinstance(__builtins__, dict)
    else __builtins__.get(name)
    for name in (
        "abs", "all", "any", "bool", "bytes", "complex", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hasattr",
        "int", "isinstance", "issubclass", "len", "list", "map", "max", "min",
        "pow", "print", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip",
    )
}


@dataclass
class SecurityReport:
    """Outcome of an AST security scan."""

    safe: bool
    imports: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "imports": self.imports,
            "violations": self.violations,
        }


class _SecurityScanner(ast.NodeVisitor):
    """Walk an AST and collect security-policy violations."""

    def __init__(self, allowed_modules: frozenset[str]) -> None:
        self.allowed = allowed_modules
        self.imports: list[str] = []
        self.violations: list[str] = []

    # ── imports ──────────────────────────────────────────────────────
    def _check_module(self, module: str | None) -> None:
        if not module:
            return
        root = module.split(".")[0]
        self.imports.append(module)
        if root in BLOCKED_MODULES:
            self.violations.append(f"blocked import of system module '{root}'")
        elif root not in self.allowed:
            self.violations.append(
                f"import of '{root}' is not in the allowed library set"
            )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._check_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self._check_module(node.module)
        self.generic_visit(node)

    # ── dangerous calls / names / attributes ─────────────────────────
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Name) and func.id in BLOCKED_NAMES:
            self.violations.append(f"call to forbidden builtin '{func.id}()'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in BLOCKED_NAMES:
            self.violations.append(f"reference to forbidden name '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Block dunder attribute access (e.g. __globals__, __subclasses__,
        # __class__, __builtins__) used for sandbox escapes.
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.violations.append(
                f"access to dunder attribute '{node.attr}' is forbidden"
            )
        self.generic_visit(node)


def scan_code(
    code: str, allowed_modules: frozenset[str] | None = None
) -> SecurityReport:
    """Statically scan ``code`` and raise :class:`SecurityError` if unsafe.

    Args:
        code: Python source to vet.
        allowed_modules: Import allowlist. Defaults to the data-science set.

    Returns:
        A :class:`SecurityReport` (``safe=True``) when the code passes.

    Raises:
        SecurityError: when the code fails to parse or violates the policy.
    """
    allowed = allowed_modules or DEFAULT_ALLOWED_MODULES
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SecurityError(f"code failed to parse: {exc}") from exc

    scanner = _SecurityScanner(allowed)
    scanner.visit(tree)
    if scanner.violations:
        raise SecurityError("; ".join(dict.fromkeys(scanner.violations)))
    return SecurityReport(safe=True, imports=scanner.imports)


def _make_safe_import(allowed: frozenset[str]):
    """Return an ``__import__`` replacement that honours the allowlist."""

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".")[0]
        if root in BLOCKED_MODULES or root not in allowed:
            raise SecurityError(f"import of '{root}' is blocked in the sandbox")
        return importlib.import_module(name)

    return _safe_import


def safe_globals(
    allowed_modules: frozenset[str] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a severely restricted ``globals()`` for sandboxed execution.

    The returned namespace exposes only a curated set of builtins, a guarded
    ``__import__`` limited to the allowlist, and pre-loaded ``pd``/``np``/
    ``stats`` handles. It contains **no** application state, environment
    variables, or service clients.
    """
    allowed = allowed_modules or DEFAULT_ALLOWED_MODULES
    builtins_ns = dict(_SAFE_BUILTINS)
    builtins_ns["__import__"] = _make_safe_import(allowed)
    g: dict[str, Any] = {
        "__builtins__": builtins_ns,
        "__name__": "__epiphany_sandbox__",
        "np": np,
        "pd": None,  # lazily importable via the guarded __import__ if allowed
        "stats": stats,
    }
    if context:
        g.update(context)
    return g


@dataclass
class SandboxExecution:
    """Result of attempting to execute sandboxed code."""

    ok: bool
    error: str | None = None
    namespace: dict[str, Any] | None = None
    security: SecurityReport | None = None


async def run_sandboxed_code(
    code: str,
    context: dict[str, Any] | None = None,
    allowed_modules: frozenset[str] | None = None,
    timeout: float = 5.0,
) -> SandboxExecution:
    """Securely execute dynamically generated code.

    Pipeline: AST scan (return a safe error on violation) → execute the vetted
    code inside a **hardened, killable subprocess** (network blocked, CPU/memory
    limited, restricted ``globals()``) wrapped in :func:`asyncio.wait_for`. On
    timeout the subprocess is killed, so a runaway loop or hostile payload yields
    a safe error instead of crashing or blocking the API process.

    Args:
        code: Python source to execute. A top-level ``result`` variable, if set,
            is returned (JSON-safe) in ``namespace['result']``.
        context: Optional JSON-serialisable names injected into the namespace.
        allowed_modules: Import allowlist (defaults to the data-science set).
        timeout: Hard wall-clock limit in seconds (default 5).

    Returns:
        A :class:`SandboxExecution`; ``ok=False`` with ``error`` set when the
        code is blocked, times out, or raises.
    """
    # 1) Static security screen — never spawn a worker for code that fails.
    try:
        report = scan_code(code, allowed_modules)
    except SecurityError as exc:
        logger.warning("Sandbox blocked unsafe code: %s", exc)
        return SandboxExecution(
            ok=False,
            error=f"SecurityError: {exc}",
            security=SecurityReport(safe=False, violations=[str(exc)]),
        )

    job = {
        "code": code,
        "allowed_modules": sorted(allowed_modules) if allowed_modules else None,
        "context": context or {},
    }

    # 2) Time-boxed execution in a separate, hardened process we can kill.
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "app.services.code_exec_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(job).encode()), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning("Sandbox execution exceeded %.1fs timeout; killing.", timeout)
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:  # pragma: no cover - already gone
                pass
        return SandboxExecution(
            ok=False, error=f"TimeoutError: execution exceeded {timeout:.0f}s"
        )
    except Exception as exc:  # noqa: BLE001 - contain all spawn/IO failures
        logger.warning("Sandbox worker failed to run: %s", exc)
        return SandboxExecution(ok=False, error=f"{type(exc).__name__}: {exc}")

    if proc.returncode not in (0, 1):
        detail = stderr.decode()[:300] if stderr else "sandbox worker crashed"
        return SandboxExecution(ok=False, error=f"worker exited {proc.returncode}: {detail}")

    try:
        payload = json.loads(stdout.decode() or "{}")
    except json.JSONDecodeError:
        return SandboxExecution(ok=False, error="sandbox produced no parsable result")

    if not payload.get("ok"):
        return SandboxExecution(ok=False, error=payload.get("error", "execution failed"))

    return SandboxExecution(
        ok=True, namespace={"result": payload.get("result")}, security=report
    )


@dataclass
class TestResult:
    """Outcome of a statistical validation run (any adaptive test)."""

    test: str
    statistic: float
    p_value: float
    is_significant: bool
    sample_size: int
    alpha: float
    effect_size: float | None = None
    effect_name: str | None = None
    statistic_name: str | None = None
    summary: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "statistic": round(self.statistic, 4),
            "statistic_name": self.statistic_name,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "sample_size": self.sample_size,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "effect_name": self.effect_name,
            "summary": self.summary,
            "detail": self.detail,
        }


# Keys that are promoted to first-class TestResult fields; the rest land in
# ``detail`` so test-specific evidence (contingency tables, group means, ...) is
# preserved without bloating the core schema.
_PROMOTED_KEYS = frozenset(
    {
        "test", "statistic", "statistic_name", "p_value", "is_significant",
        "sample_size", "alpha", "effect_size", "effect_name", "summary",
    }
)


def _result_from_payload(payload: dict[str, Any]) -> TestResult:
    """Build a :class:`TestResult` from a worker JSON payload."""
    detail = {k: v for k, v in payload.items() if k not in _PROMOTED_KEYS}
    return TestResult(
        test=payload["test"],
        statistic=float(payload["statistic"]),
        statistic_name=payload.get("statistic_name"),
        p_value=float(payload["p_value"]),
        is_significant=bool(payload["is_significant"]),
        sample_size=int(payload["sample_size"]),
        alpha=float(payload["alpha"]),
        effect_size=payload.get("effect_size"),
        effect_name=payload.get("effect_name"),
        summary=payload.get("summary", ""),
        detail=detail,
    )


async def run_statistical_test_isolated(
    test: str,
    feature_values: list[Any],
    target_values: list[Any],
    alpha: float = 0.05,
    threshold: float | None = None,
    timeout: float = 45.0,
) -> TestResult:
    """Run an adaptive statistical test in a hardened, isolated subprocess.

    Spawns :mod:`app.services.sandbox_worker` (no network, resource-limited),
    feeds it the chosen ``test`` plus the *real* row-aligned values, and parses
    the JSON result. Falls back to running the test in-process if the worker
    cannot be spawned (e.g. a restricted platform), so validation never silently
    stalls.

    Args:
        test: One of ``chi_square`` | ``t_test`` | ``anova`` | ``correlation``.
        feature_values: Real feature column values from the dataset.
        target_values: Real target column values, row-aligned with the feature.
        alpha: Significance threshold.
        threshold: Optional split point for a numeric feature in a Chi-Square.
        timeout: Hard wall-clock limit for the worker.
    """
    job = {
        "test": test,
        "alpha": alpha,
        "threshold": threshold,
        "feature_values": list(feature_values),
        "target_values": list(target_values),
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "app.services.sandbox_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(job).encode()), timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode()[:500] or "sandbox worker failed")
        payload = json.loads(stdout.decode())
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return _result_from_payload(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Isolated sandbox failed (%s); running test in-process.", exc)
        from app.services.statistics import run_statistical_test

        payload = run_statistical_test(
            test, feature_values, target_values, alpha=alpha, threshold=threshold
        )
        return _result_from_payload(payload)
