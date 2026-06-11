"""Ad-hoc Phase 4 security checks for the validation sandbox."""

from __future__ import annotations

import asyncio

from app.services.model_generator import render_model_script
from app.services.sandbox import (
    MODEL_ALLOWED_MODULES,
    SecurityError,
    run_sandboxed_code,
    scan_code,
)

MALICIOUS = [
    ("os.system", "import os\nos.system('rm -rf /')"),
    ("subprocess", "import subprocess\nsubprocess.run(['curl', 'evil'])"),
    ("shutil", "import shutil\nshutil.rmtree('/')"),
    ("sys.exit", "import sys\nsys.exit(1)"),
    ("eval", "eval('2+2')"),
    ("exec", "exec('x=1')"),
    ("open file", "open('/etc/passwd').read()"),
    ("dunder escape", "().__class__.__bases__[0].__subclasses__()"),
    ("builtins import", "__import__('os').system('id')"),
]

SAFE = (
    "import numpy as np\n"
    "from scipy import stats\n"
    "arr = np.array([1, 2, 3, 4])\n"
    "result = float(arr.mean())\n"
)


def main() -> None:
    print("--- STATIC SCAN (should all BLOCK) ---")
    for name, code in MALICIOUS:
        try:
            scan_code(code)
            print(f"  FAIL-OPEN: {name} passed!")
        except SecurityError as exc:
            print(f"  BLOCKED {name}: {exc}")

    print("--- SAFE CODE (should PASS) ---")
    try:
        report = scan_code(SAFE)
        print("  scan OK, imports:", report.imports)
    except SecurityError as exc:
        print("  FALSE-POSITIVE:", exc)

    async def run_exec() -> None:
        print("--- run_sandboxed_code ---")
        r = await run_sandboxed_code(SAFE)
        print("  safe exec ok=", r.ok, "| result=", (r.namespace or {}).get("result"))
        r = await run_sandboxed_code("import os\nos.system('id')")
        print("  malicious exec ok=", r.ok, "| error=", r.error)
        r = await run_sandboxed_code("x = 0\nwhile True:\n    x += 1\n", timeout=2.0)
        print("  infinite loop ok=", r.ok, "| error=", r.error)

        print("--- generated model script screen (MODEL allowlist) ---")
        script = render_model_script(
            {
                "feature": "avg_latency_ms",
                "target": "churn_30_days",
                "threshold": 450,
                "validation": {
                    "statistic": 9.0,
                    "p_value": 0.004,
                    "sample_size": 10000,
                    "alpha": 0.05,
                },
            }
        )
        try:
            rep = scan_code(script, allowed_modules=MODEL_ALLOWED_MODULES)
            print("  model script SAFE, imports:", rep.imports)
        except SecurityError as exc:
            print("  model script BLOCKED (unexpected):", exc)

    asyncio.run(run_exec())


if __name__ == "__main__":
    main()
