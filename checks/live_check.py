"""Check the live app the way a visitor uses it: open it, ask a question, read the answer.

Run by .github/workflows/live-check.yml (on demand and weekly).
  APP_URL    the app to check (default https://filings.streamlit.app)
  QUESTION   what to ask (default: Tesla supply chain risk); must get a cited answer
  QUESTIONS  optional batch, separated by "|": each must get a cited answer or a clear
             "Not found in the filings." (both are correct behaviour); errors fail the run
Prints what the page shows for each question and a summary table.
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
ANSWER_SECONDS = 300  # a first question after the app wakes can take minutes
WAKE_BUTTON = re.compile(r"get this app back up|wake .*up", re.I)
DONE = re.compile(
    r"Answered in|Not found in the filings|Setup incomplete|not reachable|limit|quota"
)


def app_frame(page, deadline: float):
    """The frame that renders the app (Streamlit Cloud wraps the app in an iframe).

    Counts as loaded only once the question box is on screen: the sleep page and the
    Streamlit Cloud shell can show the app's name before the app itself is running. The
    wake-up button is searched for in every frame and clicked whenever it appears.
    """
    woke = False
    while time.time() < deadline:
        for frame in page.frames:
            try:
                wake = frame.get_by_role("button", name=WAKE_BUTTON)
                if wake.count():
                    if not woke:
                        print("App was asleep; waking it up", flush=True)
                    woke = True
                    wake.first.click()
                    break
                if frame.get_by_label("Your question").count():
                    return frame
            except Exception:  # frame navigated away while we looked
                continue
        page.wait_for_timeout(3000)
    raise SystemExit(
        f"The app did not load within {WAKE_SECONDS} s:\n{page.inner_text('body')[:2000]}"
    )


def ask(page, question: str) -> tuple[str, str]:
    """Ask one question in a fresh session; return (outcome, text the app shows)."""
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
    frame = app_frame(page, time.time() + WAKE_SECONDS)
    frame.get_by_label("Your question").fill(question)
    frame.get_by_role("button", name="Ask", exact=True).click()
    deadline = time.time() + ANSWER_SECONDS
    shown = ""
    while time.time() < deadline:
        shown = frame.locator("body").inner_text().split("Your question", 1)[-1]
        if "Answered in" in shown or DONE.search(shown):
            break
        page.wait_for_timeout(2000)
    page.wait_for_timeout(1000)
    if "Answered in" in shown and "open the 10-K" in shown:
        return "answered", shown.strip()
    if "Not found in the filings" in shown:
        return "not found", shown.strip()
    return "problem", shown.strip()


def main() -> None:
    batch = [q.strip() for q in os.environ.get("QUESTIONS", "").split("|") if q.strip()]
    questions = batch or [QUESTION]
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 1100})
        for question in questions:
            outcome, shown = ask(page, question)
            results.append((question, outcome))
            print(f"---- {question} -> {outcome} ----\n{shown[:2500]}\n", flush=True)
        page.screenshot(path="live-check.png", full_page=True)
        browser.close()

    print("| outcome | question |\n|---|---|")
    for question, outcome in results:
        print(f"| {outcome} | {question} |")
    allowed = {"answered", "not found"} if batch else {"answered"}
    failed = [q for q, outcome in results if outcome not in allowed]
    if failed:
        sys.exit(f"{len(failed)} question(s) did not get the expected response: {failed}")
    print(f"Live app handled all {len(results)} question(s).")


if __name__ == "__main__":
    main()
