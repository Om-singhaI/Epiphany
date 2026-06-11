"""Isolated code-execution worker — runs untrusted generated code in a process.

Run via ``python -m app.services.code_exec_worker`` with a JSON job on stdin:

    {"code": "...", "allowed_modules": ["pandas", ...], "context": {...}}

It writes a JSON result to stdout. The process is hardened (network blocked,
CPU/memory limited) before any code runs, and the code is re-scanned with the
AST security policy as defense-in-depth. Because this runs as a *separate*
process, the parent (:func:`app.services.sandbox.run_sandboxed_code`) can
``kill()`` it on timeout — so an infinite loop or runaway aggregation is
terminated instead of blocking the API.
"""

from __future__ import annotations

import json
import sys
from typing import Any

# Import heavy/stdlib modules (asyncio, ssl, numpy, scipy via ``sandbox``)
# BEFORE hardening. ``_harden`` swaps out ``socket.socket``, which breaks later
# ``class SSLSocket(socket)`` definitions during a lazy ``ssl`` import.
from app.services.sandbox import SecurityError, safe_globals, scan_code
from app.services.sandbox_worker import _harden


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of an execution result to a JSON-safe value."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _run(job: dict) -> dict:
    code = job.get("code", "")
    allowed = job.get("allowed_modules")
    allowed_set = frozenset(allowed) if allowed else None
    context = job.get("context") or {}

    try:
        scan_code(code, allowed_set)
    except SecurityError as exc:
        return {"ok": False, "error": f"SecurityError: {exc}"}

    g = safe_globals(allowed_set, context)
    try:
        compiled = compile(code, "<epiphany-sandbox>", "exec")
        exec(compiled, g)  # noqa: S102 - guarded execution in a hardened process
    except SecurityError as exc:
        return {"ok": False, "error": f"SecurityError: {exc}"}
    except Exception as exc:  # noqa: BLE001 - contain all execution failures
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    result = g.get("result")
    return {"ok": True, "result": _jsonable(result)}


def main() -> int:
    _harden()
    try:
        job = json.loads(sys.stdin.read() or "{}")
        sys.stdout.write(json.dumps(_run(job)))
        return 0
    except Exception as exc:  # noqa: BLE001 - surface failure to the parent
        sys.stdout.write(json.dumps({"ok": False, "error": repr(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
