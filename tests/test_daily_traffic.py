from datetime import date, datetime
from zoneinfo import ZoneInfo

from checks import daily_traffic as dt
from src.config import load_config
from src.retrieve import detect_companies

NY = ZoneInfo("America/New_York")
POOL = dt.load_pool()


def test_pool_is_large_unique_and_names_one_company_each():
    assert len(POOL) >= dt.MAX_PER_DAY  # a full day never repeats a question
    assert len(set(POOL)) == len(POOL)
    cfg = load_config()
    for q in POOL:
        assert len(detect_companies(q, cfg)) == 1, q  # retrieval searches the right filing


def test_daily_total_is_random_within_bounds_and_stable_per_day():
    totals = [dt.daily_total(date(2026, 10, d)) for d in range(1, 31)]
    assert all(dt.MIN_PER_DAY <= t <= dt.MAX_PER_DAY for t in totals)
    assert len(set(totals)) > 10  # varies from day to day
    assert dt.daily_total(date(2026, 10, 7)) == dt.daily_total(date(2026, 10, 7))


def test_slots_split_the_day_without_repeats():
    day = date(2026, 10, 7)
    per_slot = [dt.questions_for_slot(POOL, day, s) for s in range(dt.SLOTS)]
    asked = [q for qs in per_slot for q in qs]
    assert len(asked) == dt.daily_total(day)
    assert len(set(asked)) == len(asked)
    assert per_slot == [dt.questions_for_slot(POOL, day, s) for s in range(dt.SLOTS)]  # stable


def test_slot_for_local_time_only_inside_the_window():
    assert dt.slot_for(datetime(2026, 10, 7, 8, 59, tzinfo=NY)) is None
    assert dt.slot_for(datetime(2026, 10, 7, 9, 23, tzinfo=NY)) == 0
    assert dt.slot_for(datetime(2026, 10, 7, 16, 23, tzinfo=NY)) == dt.SLOTS - 1
    assert dt.slot_for(datetime(2026, 10, 7, 17, 0, tzinfo=NY)) is None


def test_plan_uses_new_york_time(capsys):
    # 13:23 UTC is 9:23 in New York during daylight saving (EDT, UTC-4).
    utc = datetime(2026, 10, 7, 13, 23, tzinfo=ZoneInfo("UTC"))
    day, slot, questions = dt.plan(utc, POOL)
    assert (day, slot) == (date(2026, 10, 7), 0)
    assert questions == dt.questions_for_slot(POOL, day, 0)
    # 13:23 UTC in January is 8:23 in New York (EST, UTC-5): outside the window.
    assert dt.plan(datetime(2026, 1, 7, 13, 23, tzinfo=ZoneInfo("UTC")), POOL)[1] is None
