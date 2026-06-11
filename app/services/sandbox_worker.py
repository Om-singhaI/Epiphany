"""Isolated sandbox worker — executed as a separate process.

Run via ``python -m app.services.sandbox_worker`` with a JSON job on stdin; it
writes a JSON result to stdout. This process is hardened before any statistical
code runs:

* **No network egress** — :func:`socket.socket` is replaced with a stub that
  raises, so the validation step cannot exfiltrate data or call out.
* **Resource limits** — CPU-seconds and address-space caps (POSIX
  ``resource`` limits) bound runaway computation.

The job carries a chosen statistical ``test`` plus the *real* row-aligned
``feature_values`` / ``target_values`` pulled from the dataset. The worker runs
that exact test via :mod:`app.services.statistics` — there is no synthetic data
and no fabricated effect. The parent (:mod:`app.services.sandbox`) spawns this
worker, so a crash, hang, or memory blow-up in validation can never take down
the API process.
"""

from __future__ import annotations

import json
import sys


def _harden() -> None:
    """Lock the process down before running the numeric validation code."""
    # 1) Block all outbound network access.
    import socket

    def _blocked(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise OSError("Network access is disabled inside the Epiphany sandbox.")

    socket.socket = _blocked  # type: ignore[assignment]

    # 2) Apply CPU and memory limits where supported (POSIX only).
    try:
        import resource

        # Max 30 CPU-seconds.
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        # Cap address space at ~2 GiB (skip if the platform rejects it).
        two_gib = 2 * 1024 * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (two_gib, two_gib))
        except (ValueError, OSError):
            pass
    except ImportError:
        # `resource` is unavailable on some platforms (e.g. Windows); the
        # network block and process isolation still apply.
        pass


def _run(job: dict) -> dict:
    # Imported here so the hardening above is in force first.
    from app.services.statistics import run_statistical_test

    test = job.get("test", "chi_square")
    alpha = float(job.get("alpha", 0.05))
    threshold = job.get("threshold")
    feature_values = job.get("feature_values") or []
    target_values = job.get("target_values") or []

    if not feature_values or not target_values:
        return {"error": "no data supplied to the sandbox (empty feature/target)"}

    return run_statistical_test(
        test=test,
        feature_values=feature_values,
        target_values=target_values,
        alpha=alpha,
        threshold=float(threshold) if threshold is not None else None,
    )


def main() -> int:
    _harden()
    try:
        job = json.loads(sys.stdin.read() or "{}")
        result = _run(job)
        sys.stdout.write(json.dumps(result))
        return 0
    except Exception as exc:  # noqa: BLE001 - surface failure to the parent
        sys.stdout.write(json.dumps({"error": repr(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
