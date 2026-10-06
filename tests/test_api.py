from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from src import api, monitor
from src.llm import LLMError

CFG = {
    "llm": {"model": "test-model"},
    "retrieval": {"top_k": 8},
    "app": {"max_questions_per_day": 3, "max_question_chars": 100, "low_confidence_score": 0.4},
}
CHUNK = {
    "id": "TSLA_1A_0002",
    "ticker": "TSLA",
    "text": "word " * 200,
    "source_url": "https://www.sec.gov/tsla.htm",
    "score": 0.62,
}


def fake_answer(question, refused=False):
    return {
        "question": question,
        "answer": "Not found in the filings." if refused else "Supply risk [1].",
        "refused": refused,
        "citations": []
        if refused
        else [{"n": 1, "id": CHUNK["id"], "source_url": CHUNK["source_url"]}],
        "chunks": [CHUNK],
    }


@pytest.fixture
def log():
    return monitor.MemoryLog()


def client(log, answer_fn=fake_answer):
    return TestClient(api.create_app(answer_fn=answer_fn, log=log, cfg=CFG))


def test_health_reports_settings_without_secrets(log, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    body = client(log).get("/health").json()
    assert body == {
        "status": "ok",
        "model": "test-model",
        "top_k": 8,
        "llm_key_configured": True,
        "db_configured": False,
    }
    assert "secret-value" not in str(body)


def test_ask_returns_answer_with_citations_and_short_sources(log):
    r = client(log).post("/ask", json={"question": "What does Tesla say about supply?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Supply risk [1]." and not body["refused"]
    assert body["citations"] == [
        {
            "n": 1,
            "id": "TSLA_1A_0002",
            "ticker": "TSLA",
            "source_url": "https://www.sec.gov/tsla.htm",
        }
    ]
    src = body["sources"][0]
    assert src["id"] == "TSLA_1A_0002" and src["score"] == 0.62
    assert len(src["snippet"]) <= api.SNIPPET_CHARS + 1  # trimmed, with an ellipsis
    assert isinstance(body["latency_ms"], int)


def test_ask_logs_each_question(log):
    c = client(log, answer_fn=lambda q: fake_answer(q, refused="CEO" in q))
    c.post("/ask", json={"question": "What does Tesla say about supply?"})
    c.post("/ask", json={"question": "Who is the CEO of Tesla?"})
    assert [r["status"] for r in log.rows] == ["answered", "refused"]
    first = log.rows[0]
    assert first["tickers"] == ["TSLA"] and first["n_citations"] == 1
    assert first["top_score"] == 0.62 and first["model"] == "test-model"


def test_ask_validates_question_length(log):
    c = client(log)
    assert c.post("/ask", json={"question": "hi"}).status_code == 422
    assert c.post("/ask", json={"question": "x" * 101}).status_code == 422
    assert log.rows == []  # rejected before any work or logging


def test_daily_cap_returns_429(log):
    for _ in range(3):
        log.add({"status": "answered", "ts": datetime.now(UTC)})
    log.add({"status": "answered", "ts": datetime.now(UTC) - timedelta(days=2)})  # not today
    r = client(log).post("/ask", json={"question": "What does Tesla say about supply?"})
    assert r.status_code == 429
    assert "daily limit" in r.json()["detail"]


def test_llm_error_returns_503_and_is_logged(log):
    def broken(q):
        raise LLMError("Gemini daily free-tier quota is used up for this model")

    r = client(log, answer_fn=broken).post("/ask", json={"question": "Tesla supply risk?"})
    assert r.status_code == 503
    assert "quota" in r.json()["detail"]
    assert log.rows[-1]["status"] == "error" and "quota" in log.rows[-1]["error"]


def test_unexpected_error_hides_details(log):
    def broken(q):
        raise RuntimeError("password=hunter2 in connection string")

    r = client(log, answer_fn=broken).post("/ask", json={"question": "Tesla supply risk?"})
    assert r.status_code == 500
    assert "hunter2" not in r.text
    assert log.rows[-1]["status"] == "error"


def test_logging_failure_does_not_break_answers():
    class BrokenLog(monitor.MemoryLog):
        def add(self, record):
            raise ConnectionError("database down")

    r = client(BrokenLog()).post("/ask", json={"question": "Tesla supply risk?"})
    assert r.status_code == 200


def test_stats_summarizes_the_log(log):
    c = client(log, answer_fn=lambda q: fake_answer(q, refused="CEO" in q))
    c.post("/ask", json={"question": "What does Tesla say about supply?"})
    c.post("/ask", json={"question": "Who is the CEO of Tesla?"})
    body = c.get("/stats", params={"days": 7}).json()
    assert body["total"] == 2 and body["refusal_rate"] == 0.5
    assert len(body["daily"]) == 7
    assert c.get("/stats", params={"days": 0}).status_code == 422


def test_real_answer_pipeline_is_wired_end_to_end(monkeypatch):
    # create_app() without answer_fn uses the real answer() -> retrieve() -> chat() chain;
    # only the vector search and the Gemini HTTP call are faked.
    from src import index, llm
    from src.config import load_config

    cfg = load_config()
    searched = {}

    def fake_query(text, index_name, tickers, k):
        searched.update(index_name=index_name, tickers=tickers, k=k)
        return [("TSLA_1A_0002", "TSLA", "Tesla depends on suppliers.", "https://sec/t", 0.61)]

    class GeminiReply:
        status_code = 200
        text = ""
        headers = {}

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Supplier risk [1]."}]}}]}

    monkeypatch.setattr(index, "query", fake_query)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: GeminiReply())
    c = TestClient(api.create_app(log=monitor.MemoryLog(), cfg=cfg))
    body = c.post("/ask", json={"question": "What does Tesla say about suppliers?"}).json()
    assert body["answer"] == "Supplier risk [1]."
    assert body["citations"][0]["id"] == "TSLA_1A_0002"
    assert searched == {"index_name": "main", "tickers": ["TSLA"], "k": cfg["retrieval"]["top_k"]}


def test_without_database_the_log_falls_back_to_memory(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    assert isinstance(api._default_log(), monitor.MemoryLog)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@host:5432/db")
    log = api._default_log()  # no connection is made until the first question
    assert isinstance(log, monitor.PgLog) and "sslmode=require" in log.url


def test_unreadable_log_does_not_block_questions():
    class CountFails(monitor.MemoryLog):
        def count_since(self, since):
            raise ConnectionError("database down")

    r = client(CountFails()).post("/ask", json={"question": "Tesla supply risk?"})
    assert r.status_code == 200


def test_llm_outage_that_is_not_quota_returns_503_try_again(log):
    def overloaded(q):
        raise LLMError("Gemini returned 503: high demand")

    r = client(log, answer_fn=overloaded).post("/ask", json={"question": "Tesla supply risk?"})
    assert r.status_code == 503
    assert "unavailable right now" in r.json()["detail"]


def test_module_level_app_is_built_lazily(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delitem(api.__dict__, "app", raising=False)
    assert TestClient(api.app).get("/health").status_code == 200
    with pytest.raises(AttributeError):
        api.not_a_thing  # noqa: B018
