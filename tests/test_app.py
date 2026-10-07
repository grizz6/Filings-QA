"""UI tests: run the Streamlit pages headless against fake API responses."""

import sys

import pytest
import requests

from src.config import ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))  # the pages import `client` from their own folder
import client  # noqa: E402

ANSWER = {
    "question": "What does Tesla say about supply chain risk?",
    "answer": "Component shortages could delay production [1].",
    "refused": False,
    "citations": [{"n": 1, "id": "TSLA_1A_0002", "ticker": "TSLA", "source_url": "https://sec/t"}],
    "sources": [{"id": "TSLA_1A_0002", "ticker": "TSLA", "score": 0.61,
                 "source_url": "https://sec/t", "snippet": "Tesla depends on suppliers…"}],
    "latency_ms": 2300,
}  # fmt: skip
STATS = {
    "days": 14, "total": 3, "answered": 2, "refused": 1, "errors": 0,
    "refusal_rate": 1 / 3, "low_confidence_rate": 0.25,
    "latency_p50_ms": 2100, "latency_p95_ms": 4800,
    "daily": [{"date": f"2026-10-{d:02d}", "questions": d % 3, "refused": 0, "errors": 0,
               "latency_p50_ms": 2000 if d % 3 else None} for d in range(1, 15)],
    "recent": [{"ts": "2026-10-06T12:00:00+00:00", "status": "answered",
                "top_score": 0.6, "latency_ms": 2100}],
}  # fmt: skip


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture
def api(monkeypatch):
    """Route the pages' API calls to canned responses: api[path] = (status, body)."""
    routes, seen = {}, []

    def fake_request(method, url, timeout, **kwargs):
        path = url.removeprefix(client.API_URL)
        seen.append((method, path, kwargs))
        result = routes[path]
        if isinstance(result, Exception):
            raise result
        return Resp(*result)

    monkeypatch.setattr(client, "API_URL", "http://api.test")  # HTTP mode, as in Docker
    routes["/health"] = (200, {"status": "ok", "model": "m", "top_k": 8,
                               "llm_key_configured": True, "db_configured": True})  # fmt: skip
    monkeypatch.setattr(requests, "request", fake_request)
    routes["_seen"] = seen
    return routes


def page(name):
    at = AppTest.from_file(str(ROOT / "app" / name), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def ask(at, question):
    at.text_input[0].input(question)
    next(b for b in at.button if b.label == "Ask").click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_client_turns_api_errors_into_readable_messages(api):
    api["/a"] = (503, {"detail": "The free Gemini quota is used up for today."})
    api["/b"] = (422, {"detail": [{"msg": "String too short"}, {"msg": "bad"}]})
    api["/c"] = (500, ValueError("not json"))
    api["/d"] = requests.ConnectionError("refused")
    for path, message in [("/a", "quota is used up"), ("/b", "String too short; bad"),
                          ("/c", "returned 500"), ("/d", "not reachable")]:  # fmt: skip
        with pytest.raises(client.ApiError, match=message):
            client.call("GET", path)
    api["/ok"] = (200, {"fine": True})
    assert client.call("GET", "/ok") == {"fine": True}


def test_ask_page_renders_examples_and_companies(api):
    at = page("Ask.py")
    assert at.title[0].value == "Filings Q&A"
    assert sum("?" in b.label for b in at.button) == 4  # example questions
    assert "Tesla (TSLA)" in at.caption[0].value


def test_example_button_fills_the_question(api):
    at = page("Ask.py")
    next(b for b in at.button if "Tesla" in b.label).click()
    at.run()
    assert at.text_input[0].value == "What does Tesla say about supply chain risk?"


def test_ask_shows_cited_answer_and_sources(api):
    api["/ask"] = (200, ANSWER)
    at = ask(page("Ask.py"), "What does Tesla say about supply chain risk?")
    text = " ".join(m.value for m in at.markdown)
    assert "Component shortages could delay production [1]." in text
    assert "[open the 10-K](https://sec/t)" in text
    assert "Retrieved excerpts (1)" in at.expander[0].label
    assert any("2.3 s" in c.value for c in at.caption)
    method, path, kwargs = api["_seen"][-1]
    assert (method, path) == ("POST", "/ask")
    assert kwargs["json"] == {"question": "What does Tesla say about supply chain risk?"}


def test_ask_shows_refusal_as_info(api):
    api["/ask"] = (200, {**ANSWER, "answer": "Not found in the filings.", "refused": True,
                         "citations": []})  # fmt: skip
    at = ask(page("Ask.py"), "Who is the CEO of Microsoft?")
    assert "Not found in the filings" in at.info[0].value


def test_ask_shows_api_errors_as_warning(api):
    api["/ask"] = (429, {"detail": "The demo's daily limit of questions is reached."})
    at = ask(page("Ask.py"), "What does Tesla say about supply chain risk?")
    assert "daily limit" in at.warning[0].value


def test_monitoring_page_shows_kpis_charts_and_recent_questions(api):
    api["/stats"] = (200, STATS)
    at = page("pages/1_Monitoring.py")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {
        "Questions": "3",
        "Refusal rate": "33%",
        "Latency p50 / p95": "2.1 s / 4.8 s",
        "Low-confidence retrievals": "25%",
    }
    assert len(at.get("vega_lite_chart")) == 2
    assert len(at.dataframe) >= 1
    assert api["_seen"][-1][2]["params"] == {"days": 14}


def test_monitoring_page_with_no_questions_yet(api):
    empty = {"total": 0, "answered": 0, "refused": 0, "refusal_rate": None,
             "low_confidence_rate": None, "latency_p50_ms": None, "latency_p95_ms": None,
             "recent": []}  # fmt: skip
    api["/stats"] = (200, {**STATS, **empty})
    at = page("pages/1_Monitoring.py")
    assert "No questions in this period yet" in at.info[0].value
    assert {m.label: m.value for m in at.metric}["Refusal rate"] == "–"


def test_monitoring_page_when_api_is_down(api):
    api["/stats"] = requests.ConnectionError("refused")
    at = page("pages/1_Monitoring.py")
    assert "not reachable" in at.warning[0].value


def test_without_api_url_the_api_runs_in_this_process(monkeypatch):
    # Streamlit Community Cloud runs one process: the UI calls the same FastAPI app in-process.
    from fastapi.testclient import TestClient

    from src import api, monitor

    def fake_answer(q):
        return {"question": q, "answer": "Risk [1].", "refused": False,
                "citations": [{"n": 1, "id": "TSLA_1A_0002", "source_url": "https://sec/t"}],
                "chunks": [{"id": "TSLA_1A_0002", "ticker": "TSLA", "text": "t",
                            "source_url": "https://sec/t", "score": 0.6}]}  # fmt: skip

    local = TestClient(api.create_app(answer_fn=fake_answer, log=monitor.MemoryLog()))
    monkeypatch.setattr(client, "API_URL", "")
    monkeypatch.setattr(client, "_local_api", lambda: local)
    monkeypatch.setattr(requests, "request", lambda *a, **k: pytest.fail("no HTTP expected"))
    body = client.call("POST", "/ask", json={"question": "Tesla supply risk?"})
    assert body["answer"] == "Risk [1]."
    with pytest.raises(client.ApiError, match="String should have at least 3 characters"):
        client.call("POST", "/ask", json={"question": "hi"})  # same validation as over HTTP
    assert client.call("GET", "/stats", params={"days": 7})["total"] == 1


def test_streamlit_secrets_become_environment_variables(monkeypatch):
    import streamlit

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", "already-set")
    secrets = {"GEMINI_API_KEY": "from-secrets", "SUPABASE_DB_URL": "from-secrets"}
    monkeypatch.setattr(streamlit, "secrets", secrets)
    client.secrets_to_env()
    assert client.os.environ["GEMINI_API_KEY"] == "from-secrets"
    assert client.os.environ["SUPABASE_DB_URL"] == "already-set"  # env wins, e.g. Docker


def test_missing_streamlit_secrets_are_fine(monkeypatch):
    import streamlit

    class NoSecrets:
        def __contains__(self, key):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(streamlit, "secrets", NoSecrets())
    client.secrets_to_env()  # does not raise


def test_streamlit_cloud_requirements_cover_the_app():
    # Streamlit Community Cloud installs app/requirements.txt (next to the entrypoint).
    import re

    lines = (ROOT / "app" / "requirements.txt").read_text().splitlines()
    names = {re.split(r"[<>=\[ ]", ln)[0].lower() for ln in lines if ln and ln[0].isalpha()}
    needed = {"requests", "pyyaml", "numpy", "sentence-transformers", "psycopg", "pgvector",
              "fastapi", "httpx", "streamlit"}  # fmt: skip
    assert needed <= names, needed - names
    assert any("download.pytorch.org/whl/cpu" in ln for ln in lines)  # small CPU-only PyTorch


def test_setup_problems_name_missing_secrets(api):
    health = {"status": "ok", "model": "m", "top_k": 8}
    api["/health"] = (200, {**health, "llm_key_configured": False, "db_configured": True})
    assert client.setup_problems() == ["GEMINI_API_KEY"]
    api["/health"] = (200, {**health, "llm_key_configured": True, "db_configured": True})
    assert client.setup_problems() == []


def test_ask_page_explains_missing_secrets(api):
    api["/health"] = (200, {"status": "ok", "model": "m", "top_k": 8,
                            "llm_key_configured": False, "db_configured": False})  # fmt: skip
    at = page("Ask.py")
    assert "GEMINI_API_KEY" in at.error[0].value and "SUPABASE_DB_URL" in at.error[0].value
    assert "Secrets" in at.error[0].value


def test_ask_page_has_no_setup_error_when_configured(api):
    at = page("Ask.py")  # the api fixture's default /health reports both secrets set
    assert len(at.error) == 0


def test_in_process_api_is_rebuilt_when_secrets_arrive(monkeypatch):
    # Streamlit Cloud: the app can start before secrets are saved; it must not stay stuck
    # with an in-memory log once SUPABASE_DB_URL appears.
    monkeypatch.setattr(client, "secrets_to_env", lambda: None)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    client._build_local_api.cache_clear()
    first = client._local_api()
    assert client._local_api() is first  # cached while nothing changes
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@host:5432/db")
    assert client._local_api() is not first
    client._build_local_api.cache_clear()


def test_answer_text_cannot_render_images_or_links():
    # The answer comes from a model that also read the user's question: keep it plain text.
    text = "Risk [1]. ![x](http://evil.test/p.png) See [here](http://evil.test) and <b>hi</b>"
    safe = client.safe_markdown(text)
    assert "http://evil.test" not in safe and "![" not in safe
    assert "Risk [1]." in safe and "here" in safe
    assert "<b>" not in safe


def test_warmup_loads_the_model_once_in_the_background(monkeypatch):
    # Streamlit Cloud wakes the app on the first visit; loading the model then made the
    # first question slow. The warm-up starts loading as soon as the app starts.
    from src import index

    loaded = []
    monkeypatch.setattr(client, "API_URL", "")
    monkeypatch.setattr(index, "load_embedder", lambda name: loaded.append(name))
    client._warmup_thread.cache_clear()
    client.start_warmup().join(timeout=5)
    client.start_warmup().join(timeout=5)
    assert loaded == ["sentence-transformers/all-MiniLM-L6-v2"]
    client._warmup_thread.cache_clear()


def test_no_warmup_when_a_separate_api_serves_requests(monkeypatch):
    monkeypatch.setattr(client, "API_URL", "http://api.test")
    assert client.start_warmup() is None
