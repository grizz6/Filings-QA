import json

import pytest

from src import llm

CFG = {
    "endpoint": "https://example.test/v1beta",
    "model": "some-model",
    "temperature": 0.0,
    "max_tokens": 50,
}


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        if self._payload is None:  # same as requests on an empty or non-JSON body
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def _reply(text, finish="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]}


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.chat([{"role": "user", "content": "hi"}], CFG)


def test_payload_maps_roles():
    payload = llm.build_payload(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
        CFG,
    )
    assert payload["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [c["role"] for c in payload["contents"]] == ["user", "model", "user"]
    assert payload["contents"][2]["parts"][0]["text"] == "Q2"
    assert payload["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 50}


def test_chat_sends_expected_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def fake_post(url, headers, json, timeout, allow_redirects):
        assert allow_redirects is False
        seen.update(url=url, headers=headers, json=json)
        return FakeResponse(200, _reply(" hi \n"))

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.chat([{"role": "user", "content": "Say hi"}], CFG) == "hi"
    assert seen["url"] == "https://example.test/v1beta/models/some-model:generateContent"
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert seen["json"]["contents"][0]["parts"][0]["text"] == "Say hi"


def test_error_status_raises_after_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(429, text="quota exceeded")

    monkeypatch.setattr(llm.requests, "post", fake_post)
    sleeps = []
    with pytest.raises(llm.LLMError, match="429"):
        llm.chat([{"role": "user", "content": "hi"}], CFG, sleep=sleeps.append)
    assert len(calls) == llm.MAX_RETRIES + 1
    assert sleeps == [2, 4, 8, 16, 32]  # no delay given by the server: exponential backoff


def test_overloaded_then_success(monkeypatch):
    # Seen in CI: 503 "model is currently experiencing high demand".
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    responses = [FakeResponse(503, text="high demand"), FakeResponse(200, _reply("hi"))]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: responses.pop(0))
    assert llm.chat([{"role": "user", "content": "hi"}], CFG, sleep=lambda s: None) == "hi"


def test_client_errors_are_not_retried(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(404, text="model not found")

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError, match="404"):
        llm.chat([{"role": "user", "content": "hi"}], CFG, sleep=lambda s: None)
    assert len(calls) == 1


def test_non_json_200_raises_clear_error(monkeypatch):
    # Seen with the retired GitHub Models endpoint: HTTP 200 with a plain-text body.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(200, None, text="OK"))
    with pytest.raises(llm.LLMError, match="not valid JSON"):
        llm.chat([{"role": "user", "content": "hi"}], CFG)


def test_unexpected_shape_raises_clear_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm.requests, "post", lambda *a, **k: FakeResponse(200, {"error": "nope"}, text="{}")
    )
    with pytest.raises(llm.LLMError, match="unexpected response"):
        llm.chat([{"role": "user", "content": "hi"}], CFG)


def test_empty_text_reports_finish_reason(monkeypatch):
    # e.g. the output-token budget ran out before any visible text was produced.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm.requests, "post", lambda *a, **k: FakeResponse(200, _reply("", "MAX_TOKENS"))
    )
    with pytest.raises(llm.LLMError, match="MAX_TOKENS"):
        llm.chat([{"role": "user", "content": "hi"}], CFG)


def test_json_mode_sets_response_mime_type():
    payload = llm.build_payload([{"role": "user", "content": "hi"}], CFG, json_mode=True)
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseMimeType" not in llm.build_payload([], CFG)["generationConfig"]


# Shape of the 429 seen in CI (free tier: 5 requests/minute for gemini-3.8-flash).
def _quota_error(quota_id="GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay="41s"):
    return json.dumps(
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota ... Please retry in 41.977604342s.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": quota_id}],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
                ],
            }
        }
    )


def test_retry_delay_parsed_from_retry_info_or_message():
    assert llm.retry_delay_seconds(_quota_error(delay="41s")) == 41.0
    assert llm.retry_delay_seconds('{"error": {"message": "Please retry in 29.5s."}}') == 29.5
    assert llm.retry_delay_seconds("not json") is None


def test_rate_limit_waits_as_long_as_gemini_asks(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    responses = [FakeResponse(429, text=_quota_error(delay="41s")), FakeResponse(200, _reply("hi"))]
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: responses.pop(0))
    sleeps = []
    assert llm.chat([{"role": "user", "content": "hi"}], CFG, sleep=sleeps.append) == "hi"
    assert sleeps == [42.0]  # the server's delay plus one second of margin


def test_daily_quota_fails_fast_with_clear_message(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    body = _quota_error(quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(429, text=body)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.LLMError, match="daily free-tier quota"):
        llm.chat([{"role": "user", "content": "hi"}], CFG, sleep=lambda s: None)
    assert len(calls) == 1  # waiting a minute would not help
