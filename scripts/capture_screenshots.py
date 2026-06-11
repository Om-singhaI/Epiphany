"""Capture dashboard screenshots for the README (run against the demo server).

Usage:  python scripts/capture_screenshots.py [base_url]
Writes PNGs to docs/images/. Requires a running demo instance with Clerk
disabled (so the local demo login works headlessly).
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)


def run() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900},
                                device_scale_factor=2)
        # 1) Landing page
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(1800)
        page.screenshot(path=str(OUT / "landing.png"))
        print("saved landing.png")

        # 2) Sign in via the local demo flow → dashboard
        page.click("#landing-start")
        page.wait_for_selector("#auth-name", state="visible")
        page.fill("#auth-name", "Ada Lovelace")
        page.fill("#auth-email", "ada@epiphany.ai")
        page.fill("#auth-password", "demo12345")
        page.click("#auth-submit")
        page.wait_for_selector("#workspace", state="visible")
        page.wait_for_timeout(3500)  # let it hydrate + render the chart/code
        page.screenshot(path=str(OUT / "dashboard.png"))
        print("saved dashboard.png")

        # 3) Connections view
        try:
            page.click('button[data-view="connections"]')
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "connections.png"))
            print("saved connections.png")
        except Exception as exc:  # noqa: BLE001
            print("connections screenshot skipped:", exc)

        browser.close()


if __name__ == "__main__":
    run()
