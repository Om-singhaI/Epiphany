"""Phase 5 end-to-end verification: dynamic model + GitLab deploy + final log.

Run: PYTHONPATH=. FORCE_SIMULATION=true python scripts/test_phase5_deploy.py
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.services.clients.gitlab_client import GitLabClient
from app.services.events import event_bus
from app.services.model_generator import model_filename, render_model_script
from app.services.sandbox import MODEL_ALLOWED_MODULES, SecurityError, scan_code


SAMPLE_HYPOTHESIS = {
    "statement": "Enterprise users with onboarding_step < 3 churn more.",
    "feature": "onboarding_step",
    "target": "churn_30_days",
    "threshold": 3,
    "validation": {
        "statistic": 18.4,
        "p_value": 0.000034,
        "sample_size": 10000,
        "alpha": 0.05,
        "is_significant": True,
    },
}


def test_model_screen() -> None:
    print("--- generated model script AST screen ---")
    script = render_model_script(SAMPLE_HYPOTHESIS)
    print("  filename:", model_filename(SAMPLE_HYPOTHESIS))
    try:
        rep = scan_code(script, allowed_modules=MODEL_ALLOWED_MODULES)
        print("  SAFE, imports:", rep.imports)
    except SecurityError as exc:
        print("  BLOCKED (unexpected):", exc)


async def test_gitlab_sim() -> None:
    print("--- GitLabClient.deploy_model (simulation) ---")
    gl = GitLabClient(Settings())
    mr = await gl.deploy_model(SAMPLE_HYPOTHESIS)
    print("  title:        ", mr["title"])
    print("  source_branch:", mr["source_branch"])
    print("  target_branch:", mr["target_branch"])
    print("  file_path:    ", mr["file_path"])
    print("  web_url:      ", mr["web_url"])
    print("  mode:         ", mr["mode"])


async def test_full_cycle() -> None:
    print("--- full orchestrator cycle (FORCE_SIMULATION) ---")
    from app.services.agent_orchestrator import AgentOrchestrator

    orch = AgentOrchestrator()
    gitlab_logs: list[dict] = []

    async def collect() -> None:
        async with event_bus.subscribe() as q:
            while True:
                evt = await q.get()
                if evt.get("source") == "GITLAB MCP":
                    gitlab_logs.append(evt)

    task = asyncio.create_task(collect())
    result = await orch.run_cycle(user_goal="Find the strongest driver of churn.")
    await asyncio.sleep(0.2)
    task.cancel()

    print("  cycle status:", result.get("status"))
    print("  significant: ", result.get("significant"))
    mr = result.get("merge_request")
    print("  merge_request:", (mr or {}).get("web_url") if mr else None)
    print("  GITLAB MCP log events:")
    for evt in gitlab_logs:
        print(f"    [{evt['level']:8}] ({evt['mode']}) {evt['message']}")


async def main() -> None:
    test_model_screen()
    await test_gitlab_sim()
    await test_full_cycle()


if __name__ == "__main__":
    asyncio.run(main())
