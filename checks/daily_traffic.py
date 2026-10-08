"""Plan the daily traffic check: which questions to ask the live app, and when.

Each day gets a random total between MIN_PER_DAY and MAX_PER_DAY questions, drawn without
repeats from checks/questions.txt and spread over hourly slots from 9:00 to 16:59 New York
time. The plan is seeded by the date, so every run of a day agrees on the same total and on
which questions belong to each slot.

GitHub may start scheduled runs late or skip them, so each run catches up: it asks the
questions of every slot that has started since the previous run of the day (found through
the GitHub API), not only the current one. A late run still asks everything due, and no
question is asked twice.

Run:  python checks/daily_traffic.py        # prints the questions due now, separated by "|"
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

POOL_PATH = Path(__file__).with_name("questions.txt")
TIMEZONE = ZoneInfo("America/New_York")
FIRST_HOUR = 9  # slots start 9:00 local time
SLOTS = 8  # 9:xx, 10:xx, ... 16:xx
MIN_PER_DAY = 1
MAX_PER_DAY = 67
WORKFLOW = "daily-traffic.yml"


def load_pool(path: Path = POOL_PATH) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _rng(day: date) -> random.Random:
    return random.Random(f"filings-qa-daily-{day.isoformat()}")


def daily_total(day: date) -> int:
    return _rng(day).randint(MIN_PER_DAY, MAX_PER_DAY)


def questions_for_slot(pool: list[str], day: date, slot: int) -> list[str]:
    """This slot's share of the day's questions (each question asked once per day)."""
    rng = _rng(day)
    total = rng.randint(MIN_PER_DAY, MAX_PER_DAY)
    picked = rng.sample(pool, min(total, len(pool)))
    slots = [rng.randrange(SLOTS) for _ in picked]
    return [q for q, s in zip(picked, slots, strict=True) if s == slot]


def last_started_slot(local: datetime) -> int:
    """The latest slot that has started by this local time (-1 before 9:00)."""
    return min(local.hour - FIRST_HOUR, SLOTS - 1)


def due_slots(now: datetime, previous: datetime | None) -> list[int]:
    """Slots started since the previous run of the same day (all started ones if none)."""
    local = now.astimezone(TIMEZONE)
    done = -1
    if previous is not None and previous.astimezone(TIMEZONE).date() == local.date():
        done = last_started_slot(previous.astimezone(TIMEZONE))
    return list(range(done + 1, last_started_slot(local) + 1))


def plan(
    now: datetime, pool: list[str], previous: datetime | None = None
) -> tuple[date, list[int], list[str]]:
    day = now.astimezone(TIMEZONE).date()
    slots = due_slots(now, previous)
    return day, slots, [q for s in slots for q in questions_for_slot(pool, day, s)]


def previous_run_start(now: datetime) -> datetime | None:
    """Start time of the latest earlier run of this workflow today (New York date).

    Uses the GitHub API with the job's token. Runs that finished (passed or failed) count:
    they asked their questions, and a failure has already alerted the owner. Returns None
    outside GitHub Actions; raises if the API cannot be read.
    """
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        return None
    this_run = int(os.environ.get("GITHUB_RUN_ID", "0"))
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/runs?per_page=50"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        runs = json.load(response)["workflow_runs"]
    today = now.astimezone(TIMEZONE).date()
    starts = []
    for run in runs:
        if run["id"] == this_run or run.get("conclusion") not in ("success", "failure"):
            continue
        started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
        if started < now and started.astimezone(TIMEZONE).date() == today:
            starts.append(started)
    return max(starts, default=None)


def main() -> None:
    now = datetime.now(UTC)
    try:
        previous = previous_run_start(now)
    except Exception as e:  # API unavailable: ask this hour's questions only
        print(f"Could not read earlier runs ({e}); asking this hour's only", file=sys.stderr)
        previous = now - timedelta(hours=1)
    day, slots, questions = plan(now, load_pool(), previous)
    total = daily_total(day)
    where = f"slots {[s + 1 for s in slots]} of {SLOTS} due" if slots else "no slots due"
    print(f"{day}: {total} questions today; {where}; {len(questions)} now", file=sys.stderr)
    print("|".join(questions))
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"**{day}**: {total} questions today, {where}, {len(questions)} asked now.\n")


if __name__ == "__main__":
    main()
