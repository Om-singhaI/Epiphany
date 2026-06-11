"""Live provider connectivity check (no secrets printed, no writes).

Run with: python scripts/check_providers.py
Reports only the resulting mode/status for each provider so credentials are
never echoed. GitLab is auth-only here — no Merge Request is created.
"""

from __future__ import annotations

import asyncio

from app.config import get_settings
from app.services.clients.elastic_client import ElasticClient
from app.services.clients.fivetran_client import FivetranClient
from app.services.clients.gemini_client import GeminiClient
from app.services.clients.gitlab_client import GitLabClient
from app.services.data_port import DataPort


async def main() -> None:
    s = get_settings()

    # ── Data source (Elastic live, else local file) ──
    ec = ElasticClient(s)
    port = DataPort(s, ec)
    try:
        profile = await port.profile()
        print(
            f"[Data    ] mode={profile.mode} source={profile.source} "
            f"rows={profile.row_count} cols={len(profile.columns)}"
        )
        assoc = []
        targets = profile.candidate_targets()
        if targets:
            assoc = await port.rank_associations(targets[0].name)
            top = assoc[0] if assoc else {}
            print(
                f"[Explore ] target={targets[0].name} "
                f"top_driver={top.get('feature')} strength={top.get('strength')}"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[Data    ] ERROR: {e!r}")
        profile, assoc = None, []
    finally:
        await ec.aclose()

    # ── Fivetran ──
    try:
        m = await FivetranClient(s).get_sync_manifest()
        print(f"[Fivetran] mode={m['mode']} table={m.get('table')} rows={m.get('rows_synced')}")
    except Exception as e:  # noqa: BLE001
        print(f"[Fivetran] ERROR: {e!r}")

    # ── Gemini ──
    try:
        gc = GeminiClient(s)
        if profile is not None:
            h = await gc.generate_hypothesis(profile.as_dict(), assoc, "Smoke test goal")
            print(
                f"[Gemini  ] mode={h['mode']} feature={h.get('feature')} "
                f"target={h.get('target')}"
            )
        else:
            print(f"[Gemini  ] enabled={gc.enabled} (no data profile to reason over)")
    except Exception as e:  # noqa: BLE001
        print(f"[Gemini  ] ERROR: {e!r}")

    # ── GitLab (auth only, no MR created) ──
    try:
        gl = GitLabClient(s)
        if gl.enabled:
            print("[GitLab  ] enabled=True (auth succeeded)")
        else:
            print("[GitLab  ] enabled=False (auth failed -> simulation)")
    except Exception as e:  # noqa: BLE001
        print(f"[GitLab  ] ERROR: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
