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
