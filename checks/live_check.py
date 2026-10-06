"""Check the live app the way a visitor uses it: open it, ask a question, read the answer.

Run by .github/workflows/live-check.yml (on demand and weekly).
  APP_URL   the app to check (default https://filings.streamlit.app)
  QUESTION  what to ask (default: Tesla supply chain risk)
Exits non-zero, with what the page shows, if no cited answer appears.
"""

from __future__ import annotations

import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL") or "https://filings.streamlit.app"
QUESTION = os.environ.get("QUESTION") or "What does Tesla say about supply chain risk?"
WAKE_SECONDS = 300  # a sleeping Streamlit Cloud app takes a while to start
ANSWER_SECONDS = 180
DONE = re.compile(
    r"Answered in|Not found in the filings|Setup incomplete|not reachable|limit|quota"
)


def app_frame(page, deadline: float):
    """The frame that renders the app (Streamlit Cloud wraps the app in an iframe)."""
    while time.time() < deadline:
        wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
        if wake.count():
            print("App was asleep; waking it up", flush=True)
            wake.first.click()
        for frame in page.frames:
            try:
                if frame.get_by_text("Filings Q&A").count():
                    return frame
            except Exception:  # frame navigated away while we looked
                continue
        page.wait_for_timeout(3000)
    raise SystemExit(
        f"The app did not load within {WAKE_SECONDS} s:\n{page.inner_text('body')[:2000]}"
    )


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 1100})
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        frame = app_frame(page, time.time() + WAKE_SECONDS)
        print(f"Loaded {APP_URL}", flush=True)

        frame.get_by_label("Your question").fill(QUESTION)
        frame.get_by_role("button", name="Ask", exact=True).click()
        print(f"Asked: {QUESTION}", flush=True)

        deadline = time.time() + ANSWER_SECONDS
        text = ""
        while time.time() < deadline:
            text = frame.locator("body").inner_text()
            if "Answered in" in text or DONE.search(text.split("Your question", 1)[-1]):
                break
            page.wait_for_timeout(2000)
        page.wait_for_timeout(1000)
        page.screenshot(path="live-check.png", full_page=True)
        browser.close()

    shown = text.split("Your question", 1)[-1].strip()
    print("---- what the page shows ----\n" + shown[:3000] + "\n-----------------------------")
    if "Answered in" not in text or "open the 10-K" not in text:
        sys.exit("No cited answer on the live app (see the page text above).")
    print("Live app answered with citations.")


if __name__ == "__main__":
    main()
