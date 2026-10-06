from datetime import UTC, datetime, timedelta

from src import monitor

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def row(hours_ago, status="answered", latency_ms=1000, top_score=0.6, question="Q?"):
    return {
        "ts": NOW - timedelta(hours=hours_ago),
        "question": question,
        "tickers": ["TSLA"],
        "status": status,
        "n_citations": 0 if status != "answered" else 2,
        "top_score": top_score,
        "latency_ms": latency_ms,
        "model": "m",
        "error": None,
    }


def test_percentile_nearest_rank():
    assert monitor.percentile([], 50) is None
    assert monitor.percentile([5], 95) == 5
    assert monitor.percentile([1, 2, 3, 4, 100], 50) == 3
    assert monitor.percentile([1, 2, 3, 4, 100], 95) == 100


def test_summarize_counts_rates_and_latency():
    rows = [
        row(1, "answered", 1000, 0.7),
        row(2, "answered", 2000, 0.6),
        row(3, "refused", 3000, 0.3),
        row(4, "error", 50, None),
    ]
    s = monitor.summarize(rows, days=7, now=NOW, low_score=0.4)
    assert s["total"] == 4
    assert (s["answered"], s["refused"], s["errors"]) == (2, 1, 1)
    assert s["refusal_rate"] == 1 / 3  # errors excluded: they got no answer at all
    assert s["latency_p50_ms"] == 2000 and s["latency_p95_ms"] == 3000  # errors excluded
    assert s["low_confidence_rate"] == 1 / 3  # top score 0.3 < 0.4; errors have no score


def test_summarize_fills_every_day_and_lists_recent_first():
    rows = [row(1, question="new"), row(50, "refused", question="old")]
    s = monitor.summarize(rows, days=3, now=NOW, low_score=0.4)
    assert [d["date"] for d in s["daily"]] == ["2026-10-04", "2026-10-05", "2026-10-06"]
    assert [d["questions"] for d in s["daily"]] == [1, 0, 1]
    assert s["daily"][0]["refused"] == 1
    assert s["daily"][1]["latency_p50_ms"] is None
    assert [r["question"] for r in s["recent"]] == ["new", "old"]


def test_summarize_empty_period():
    s = monitor.summarize([], days=2, now=NOW, low_score=0.4)
    assert s["total"] == 0 and s["refusal_rate"] is None and s["latency_p95_ms"] is None
    assert len(s["daily"]) == 2


def test_memory_log_counts_and_filters_by_time():
    log = monitor.MemoryLog()
    log.add(row(30))
    log.add(row(1))
    assert log.count_since(NOW - timedelta(hours=2)) == 1
    assert len(log.rows_since(NOW - timedelta(days=2))) == 2


def test_report_markdown_mentions_the_key_numbers():
    s = monitor.summarize([row(1), row(2, "refused", top_score=0.2)], 7, NOW, 0.4)
    text = monitor.report_markdown(s)
    assert "| Questions | 2 |" in text
    assert "| Refusal rate | 50.0% |" in text
    assert "| Low-confidence retrievals | 50.0% |" in text


class FakeConn:
    """Stands in for a psycopg connection: records every statement and its parameters."""

    def __init__(self, calls, result):
        self.calls, self.result = calls, result

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        return self

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_psycopg(monkeypatch, result=None):
    import sys
    import types

    calls = []
    psycopg = types.SimpleNamespace(connect=lambda url, row_factory=None: FakeConn(calls, result))
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", types.SimpleNamespace(dict_row=object()))
    return calls


def test_pglog_insert_matches_columns_and_creates_schema_once(monkeypatch):
    # A mismatch here would be silent in production: the API swallows logging errors.
    calls = fake_psycopg(monkeypatch)
    log = monitor.PgLog("postgresql://x", source="app")
    record = {c: f"v_{c}" for c in monitor.COLUMNS}
    log.add(record)
    log.add(record)
    sqls = [sql for sql, _ in calls]
    assert sum(s.startswith("CREATE TABLE") for s in sqls) == 1
    inserts = [(sql, p) for sql, p in calls if sql.startswith("INSERT")]
    assert len(inserts) == 2
    sql, params = inserts[0]
    assert sql.count("%s") == len(params) == len(monitor.COLUMNS) + 1
    assert params == ("app", *(f"v_{c}" for c in monitor.COLUMNS))
    assert sql.startswith("INSERT INTO qa_log (source, " + ", ".join(monitor.COLUMNS) + ")")


def test_pglog_reads_only_its_own_source(monkeypatch):
    since = NOW - timedelta(days=1)
    calls = fake_psycopg(monkeypatch, result={"n": 7})
    assert monitor.PgLog("postgresql://x", source="app").count_since(since) == 7
    assert calls[-1][1] == ("app", since)
    calls = fake_psycopg(monkeypatch, result=[row(1)])
    assert monitor.PgLog("postgresql://x", source="ci").rows_since(since) == [row(1)]
    assert "WHERE source = %s AND ts >= %s" in calls[-1][0] and calls[-1][1] == ("ci", since)


def test_main_writes_the_report_to_the_job_summary(monkeypatch, tmp_path, capsys):
    store = monitor.MemoryLog()
    store.add({**row(0), "ts": datetime.now(UTC)})  # real clock: main() reads the last 7 days
    monkeypatch.setattr(monitor, "PgLog", lambda url: store)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://x")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setattr("sys.argv", ["monitor", "--days", "7"])
    monitor.main()
    assert "| Questions | 1 |" in capsys.readouterr().out
    assert "| Questions | 1 |" in (tmp_path / "summary.md").read_text()
