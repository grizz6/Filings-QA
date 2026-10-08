import io
import json
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


UTC = ZoneInfo("UTC")


def ny(hour, minute=23, day=7):
    return datetime(2026, 10, day, hour, minute, tzinfo=NY)


def test_last_started_slot_covers_the_whole_day():
    assert dt.last_started_slot(ny(8, 59)) == -1  # before 9am nothing is due
    assert dt.last_started_slot(ny(9, 0)) == 0
    assert dt.last_started_slot(ny(16, 59)) == dt.SLOTS - 1
    assert dt.last_started_slot(ny(21, 5)) == dt.SLOTS - 1  # a late evening run still counts


def test_on_time_runs_ask_one_slot_each():
    assert dt.due_slots(ny(9), previous=None) == [0]
    assert dt.due_slots(ny(10), previous=ny(9)) == [1]
    assert dt.due_slots(ny(8), previous=None) == []


def test_late_or_skipped_runs_catch_up_without_repeats():
    # Runs at 9:23 and then nothing until 13:40: slots 10, 11, 12 and 13 are all asked now.
    assert dt.due_slots(ny(13, 40), previous=ny(9)) == [1, 2, 3, 4]
    # The first run of the day starts at 16:40 (or after 5pm): the whole day is caught up.
    assert dt.due_slots(ny(16, 40), previous=None) == list(range(dt.SLOTS))
    assert dt.due_slots(ny(21, 5), previous=None) == list(range(dt.SLOTS))
    # Two runs in the same hour: the second asks nothing.
    assert dt.due_slots(ny(10, 50), previous=ny(10, 23)) == []
    # Runs after the last slot was caught up ask nothing.
    assert dt.due_slots(ny(18), previous=ny(17)) == []
    # A run from yesterday does not cover today.
    assert dt.due_slots(ny(9, day=8), previous=ny(16)) == [0]


def test_every_question_is_asked_exactly_once_whatever_the_run_times():
    day = date(2026, 10, 7)
    starts = [ny(9, 40), ny(12, 5), ny(12, 50), ny(20, 30)]  # late, skipped and doubled runs
    asked, previous = [], None
    for now in starts:
        asked += dt.plan(now, POOL, previous)[2]
        previous = now
    expected = [q for s in range(dt.SLOTS) for q in dt.questions_for_slot(POOL, day, s)]
    assert asked == expected


def test_plan_uses_new_york_time():
    # 13:23 UTC is 9:23 in New York during daylight saving (EDT, UTC-4).
    day, slots, questions = dt.plan(datetime(2026, 10, 7, 13, 23, tzinfo=UTC), POOL)
    assert (day, slots) == (date(2026, 10, 7), [0])
    assert questions == dt.questions_for_slot(POOL, day, 0)
    # 13:23 UTC in January is 8:23 in New York (EST, UTC-5): nothing due yet.
    assert dt.plan(datetime(2026, 1, 7, 13, 23, tzinfo=UTC), POOL)[1] == []


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run(run_id, started, conclusion="success"):
    return {"id": run_id, "run_started_at": started, "conclusion": conclusion}


def test_previous_run_start_reads_todays_finished_runs(monkeypatch):
    runs = [
        run(5, "2026-10-07T17:40:00Z", None),  # this run (in progress)
        run(4, "2026-10-07T15:10:00Z", "cancelled"),  # stopped: its questions may be unasked
        run(3, "2026-10-07T14:05:00Z", "failure"),  # failed: already alerted, not repeated
        run(2, "2026-10-07T13:30:00Z"),
        run(1, "2026-10-07T01:05:00Z"),  # 21:05 on October 6 in New York
    ]
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["auth"] = request.full_url, request.get_header("Authorization")
        return FakeResponse(json.dumps({"workflow_runs": runs}).encode())

    monkeypatch.setattr(dt.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_RUN_ID", "5")
    now = datetime(2026, 10, 7, 17, 41, tzinfo=UTC)
    assert dt.previous_run_start(now) == datetime(2026, 10, 7, 14, 5, tzinfo=UTC)
    assert "owner/repo/actions/workflows/daily-traffic.yml/runs" in seen["url"]
    assert seen["auth"] == "Bearer t"
    runs[:] = [run(1, "2026-10-07T01:05:00Z")]
    assert dt.previous_run_start(now) is None  # nothing earlier today


def test_previous_run_start_outside_github_actions(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert dt.previous_run_start(datetime.now(UTC)) is None


def test_main_falls_back_to_this_hour_when_the_api_fails(monkeypatch, capsys, tmp_path):
    def broken(now):
        raise OSError("API down")

    monkeypatch.setattr(dt, "previous_run_start", broken)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    dt.main()
    out = capsys.readouterr()
    assert "API down" in out.err
    local = datetime.now(UTC).astimezone(dt.TIMEZONE)
    slot = dt.last_started_slot(local)
    expected = dt.questions_for_slot(POOL, local.date(), slot) if 0 <= local.hour - 9 < 8 else []
    assert out.out.strip() == "|".join(expected)
    assert "questions today" in (tmp_path / "summary.md").read_text()
