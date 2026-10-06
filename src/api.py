"""HTTP API for the Q&A system (Day 9).

  GET  /health   settings and whether the secrets are configured (never their values)
  POST /ask      {"question": "..."} -> answer, citations, retrieved sources, latency
  GET  /stats    monitoring summary of the question log (?days=14)

Every question is logged (src/monitor.py). A daily cap protects the free Gemini quota.

Run:  uvicorn src.api:app --port 8000      (then open http://localhost:8000/docs)
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import monitor
from src.config import load_config
from src.llm import LLMError

SNIPPET_CHARS = 300
STATS_MAX_DAYS = 90


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"


def _default_log():
    from src.index import db_url

    try:
        return monitor.PgLog(db_url())
    except Exception as exc:  # no database configured: keep answering, log in memory
        print(f"WARNING: question log is in memory only ({exc})", file=sys.stderr)
        return monitor.MemoryLog()


def create_app(answer_fn=None, log=None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app_cfg = cfg["app"]
    if answer_fn is None:
        from src.answer import answer

        def answer_fn(question):
            return answer(question, cfg)

    log = log if log is not None else _default_log()
    app = FastAPI(title="Filings Q&A", description=__doc__.split("\n")[0])

    def record(question, status, latency_ms, result=None, error=None):
        chunks = (result or {}).get("chunks", [])
        try:
            log.add(
                {
                    "question": question,
                    "tickers": list(dict.fromkeys(c["ticker"] for c in chunks)),
                    "status": status,
                    "n_citations": len((result or {}).get("citations", [])),
                    "top_score": max((c["score"] for c in chunks), default=None),
                    "latency_ms": latency_ms,
                    "model": cfg["llm"]["model"],
                    "error": error,
                }
            )
        except Exception as exc:  # monitoring must never break answering
            print(f"WARNING: could not log question: {exc}", file=sys.stderr)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": cfg["llm"]["model"],
            "top_k": cfg["retrieval"]["top_k"],
            "llm_key_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
            "db_configured": bool(os.environ.get("SUPABASE_DB_URL", "").strip()),
        }

    @app.post("/ask")
    def ask(req: AskRequest):
        question = req.question.strip()
        if len(question) > app_cfg["max_question_chars"]:
            raise HTTPException(
                422, f"Questions are limited to {app_cfg['max_question_chars']} characters."
            )
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            asked_today = log.count_since(today)
        except Exception as exc:
            print(f"WARNING: could not read the question log: {exc}", file=sys.stderr)
            asked_today = 0
        if asked_today >= app_cfg["max_questions_per_day"]:
            raise HTTPException(
                429, "The demo's daily limit of questions is reached. Try again tomorrow (UTC)."
            )

        start = time.perf_counter()
        elapsed = lambda: int((time.perf_counter() - start) * 1000)  # noqa: E731
        try:
            result = answer_fn(question)
        except LLMError as exc:
            record(question, "error", elapsed(), error=str(exc)[:300])
            if "quota" in str(exc).lower():
                raise HTTPException(
                    503, "The free Gemini quota is used up for today. Try again tomorrow."
                ) from None
            raise HTTPException(
                503, "The language model is unavailable right now. Try again in a minute."
            ) from None
        except Exception as exc:
            record(question, "error", elapsed(), error=f"{type(exc).__name__}: {str(exc)[:300]}")
            raise HTTPException(
                500, "Something went wrong while answering. Please try again."
            ) from None

        latency_ms = elapsed()
        record(question, "refused" if result["refused"] else "answered", latency_ms, result)
        tickers = {c["id"]: c["ticker"] for c in result["chunks"]}
        return {
            "question": question,
            "answer": result["answer"],
            "refused": result["refused"],
            "citations": [
                {**c, "ticker": tickers.get(c["id"], c["id"].split("_")[0])}
                for c in result["citations"]
            ],  # fmt: skip
            "sources": [
                {
                    "id": c["id"],
                    "ticker": c["ticker"],
                    "score": round(c["score"], 4),
                    "source_url": c["source_url"],
                    "snippet": _snippet(c["text"]),
                }
                for c in result["chunks"]
            ],  # fmt: skip
            "latency_ms": latency_ms,
        }

    @app.get("/stats")
    def stats(days: int = Query(14, ge=1, le=STATS_MAX_DAYS)):
        now = datetime.now(UTC)
        since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return monitor.summarize(log.rows_since(since), days, now, app_cfg["low_confidence_score"])

    return app


def __getattr__(name):
    """`uvicorn src.api:app` builds the real app on first access (keeps imports light)."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
