"""Keep the Streamlit Cloud app awake.

Streamlit Community Cloud puts an app to sleep after 12 hours without visitors. A plain HTTP
request does not count as a visit, so this opens the app in a headless browser like a
visitor would (waking it first if it is already asleep) and keeps the page open briefly.
It asks no question, so nothing is logged and no model call is made.

Run by .github/workflows/keep-awake.yml every few hours.
"""

from __future__ import annotations

import os
import time

from live_check import APP_URL, WAKE_SECONDS, app_frame
from playwright.sync_api import sync_playwright

STAY_SECONDS = 20  # keep the session open long enough to register as a visit


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        started = time.time()
        app_frame(page, started + WAKE_SECONDS)  # clicks the wake-up button if asleep
        loaded = time.time() - started
        page.wait_for_timeout(STAY_SECONDS * 1000)
        page.screenshot(path="keep-awake.png")
        browser.close()
    print(f"{APP_URL} is up (loaded in {loaded:.0f} s)")


if __name__ == "__main__":
    main()
