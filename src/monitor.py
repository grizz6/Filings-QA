"""Question log and monitoring (Day 10).

Every question the app answers is logged to Supabase (table qa_log): status (answered,
refused, error), latency, number of citations and the top retrieval score. `summarize`
turns those rows into what the dashboard and the weekly report show:

- questions per day, refusal rate, latency p50/p95, errors;
- low-confidence rate: share of questions whose best chunk scored below
  `app.low_confidence_score`. A rising rate means people ask about things the indexed
  Risk Factors sections do not cover (or retrieval got worse): a drift signal.

Run:  python -m src.monitor --days 7     # markdown report (needs SUPABASE_DB_URL)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import UTC, datetime, timedelta

from src.config import load_config

SOURCE = os.environ.get("LOG_SOURCE", "app")  # CI smoke tests log as "ci", not counted

SCHEMA_SQL = """CREATE TABLE IF NOT EXISTS qa_log (
    id          bigserial   PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    source      text        NOT NULL,
    question    text        NOT NULL,
    tickers     text[]      NOT NULL,
    status      text        NOT NULL,
    n_citations integer     NOT NULL,
    top_score   real,
    latency_ms  integer     NOT NULL,
    model       text        NOT NULL,
    error       text
)"""
INDEX_SQL = "CREATE INDEX IF NOT EXISTS qa_log_source_ts ON qa_log (source, ts)"
# Supabase serves tables in the public schema through its Data API. Row-level security with
# no policies closes that path; the app connects as the table owner, which RLS does not limit.
RLS_SQL = "ALTER TABLE qa_log ENABLE ROW LEVEL SECURITY"
COLUMNS = ("question", "tickers", "status", "n_citations", "top_score", "latency_ms", "model",
           "error")  # fmt: skip
RECENT_ROWS = 20


class MemoryLog:
    """In-memory stand-in for PgLog (tests, local runs without a database)."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, record: dict) -> None:
        self.rows.append({"ts": record.get("ts") or datetime.now(UTC), **record})

    def count_since(self, since: datetime) -> int:
        return len(self.rows_since(since))

    def rows_since(self, since: datetime) -> list[dict]:
        return [r for r in self.rows if r["ts"] >= since]


class PgLog:
    """qa_log table in Supabase Postgres; one short connection per call (low traffic)."""

    def __init__(self, url: str, source: str = SOURCE):
        self.url = url
        self.source = source
        self._schema_ready = False

    def _connect(self):
        import psycopg
        from psycopg.rows import dict_row

        conn = psycopg.connect(self.url, row_factory=dict_row)
        if not self._schema_ready:
            conn.execute(SCHEMA_SQL)
            conn.execute(INDEX_SQL)
            conn.execute(RLS_SQL)
            conn.commit()
            self._schema_ready = True
        return conn

    def add(self, record: dict) -> None:
        cols = ", ".join(COLUMNS)
        marks = ", ".join(["%s"] * (len(COLUMNS) + 1))
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO qa_log (source, {cols}) VALUES ({marks})",
                (self.source, *(record[c] for c in COLUMNS)),
            )

    def count_since(self, since: datetime) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT count(*) AS n FROM qa_log WHERE source = %s AND ts >= %s",
                (self.source, since),
            ).fetchone()["n"]

    def rows_since(self, since: datetime) -> list[dict]:
        with self._connect() as conn:
            return conn.execute(
                f"SELECT ts, {', '.join(COLUMNS)} FROM qa_log "
                "WHERE source = %s AND ts >= %s ORDER BY ts",
                (self.source, since),
            ).fetchall()


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation): always a value that was observed."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def summarize(rows: list[dict], days: int, now: datetime, low_score: float) -> dict:
    """Totals, rates, latency percentiles, per-day series and the most recent questions."""
    answered = [r for r in rows if r["status"] == "answered"]
    refused = [r for r in rows if r["status"] == "refused"]
    replied = answered + refused  # errors got no answer, so they are kept out of the rates
    scored = [r for r in replied if r["top_score"] is not None]

    first_day = (now - timedelta(days=days - 1)).date()
    daily = []
    for i in range(days):
        day = first_day + timedelta(days=i)
        on_day = [r for r in rows if r["ts"].date() == day]
        daily.append(
            {
                "date": day.isoformat(),
                "questions": len(on_day),
                "refused": sum(r["status"] == "refused" for r in on_day),
                "errors": sum(r["status"] == "error" for r in on_day),
                "latency_p50_ms": percentile(
                    [r["latency_ms"] for r in on_day if r["status"] != "error"], 50
                ),
            }
        )

    recent = sorted(rows, key=lambda r: r["ts"], reverse=True)[:RECENT_ROWS]
    return {
        "days": days,
        "total": len(rows),
        "answered": len(answered),
        "refused": len(refused),
        "errors": len(rows) - len(replied),
        "refusal_rate": _rate(len(refused), len(replied)),
        "low_confidence_rate": _rate(sum(r["top_score"] < low_score for r in scored), len(scored)),
        "latency_p50_ms": percentile([r["latency_ms"] for r in replied], 50),
        "latency_p95_ms": percentile([r["latency_ms"] for r in replied], 95),
        "daily": daily,
        # Question text stays in the database only: these stats are shown publicly.
        "recent": [
            {
                "ts": r["ts"].isoformat(timespec="seconds"),
                "status": r["status"],
                "top_score": r["top_score"],
                "latency_ms": r["latency_ms"],
            }
            for r in recent
        ],
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x / 1000:.1f} s"


def report_markdown(s: dict) -> str:
    lines = [
        f"### Usage, last {s['days']} days",
        "",
        "| Signal | Value |",
        "|---|---|",
        f"| Questions | {s['total']} |",
        f"| Answered / refused / errors | {s['answered']} / {s['refused']} / {s['errors']} |",
        f"| Refusal rate | {_pct(s['refusal_rate'])} |",
        f"| Low-confidence retrievals | {_pct(s['low_confidence_rate'])} |",
        f"| Latency p50 / p95 | {_ms(s['latency_p50_ms'])} / {_ms(s['latency_p95_ms'])} |",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.index import db_url

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()
    cfg = load_config()["app"]
    now = datetime.now(UTC)
    since = datetime.combine((now - timedelta(days=args.days - 1)).date(), datetime.min.time(), UTC)
    rows = PgLog(db_url()).rows_since(since)
    text = report_markdown(summarize(rows, args.days, now, cfg["low_confidence_score"]))
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
