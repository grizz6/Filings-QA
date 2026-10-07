"""Plan the daily traffic check: which questions to ask the live app, and when.

Each day gets a random total between MIN_PER_DAY and MAX_PER_DAY questions, drawn without
repeats from checks/questions.txt and spread over hourly slots from 9:00 to 16:59 New York
time. The plan is seeded by the date, so every hourly run of a day agrees on the same total
and on which questions belong to its slot.

Run:  python checks/daily_traffic.py        # prints the questions for the current slot,
                                            # separated by "|" (empty outside the window)
"""

from __future__ import annotations

import os
import random
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

POOL_PATH = Path(__file__).with_name("questions.txt")
TIMEZONE = ZoneInfo("America/New_York")
FIRST_HOUR = 9  # slots start 9:00 local time
SLOTS = 8  # 9:xx, 10:xx, ... 16:xx
MIN_PER_DAY = 1
MAX_PER_DAY = 67


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


def slot_for(local: datetime) -> int | None:
    """Slot index for a local time, or None outside 9:00-16:59."""
    slot = local.hour - FIRST_HOUR
    return slot if 0 <= slot < SLOTS else None


def plan(now: datetime, pool: list[str]) -> tuple[date, int | None, list[str]]:
    local = now.astimezone(TIMEZONE)
    slot = slot_for(local)
    questions = [] if slot is None else questions_for_slot(pool, local.date(), slot)
    return local.date(), slot, questions


def main() -> None:
    day, slot, questions = plan(datetime.now(UTC), load_pool())
    total = daily_total(day)
    where = "outside the 9am-5pm window" if slot is None else f"slot {slot + 1} of {SLOTS}"
    print(f"{day}: {total} questions today; {where}; {len(questions)} now", file=sys.stderr)
    print("|".join(questions))
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"**{day}**: {total} questions today, {where}, {len(questions)} asked now.\n")


if __name__ == "__main__":
    main()
