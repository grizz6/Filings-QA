"""Call Google Gemini (free tier) through its REST API.

The API key is read from the GEMINI_API_KEY environment variable. It lives only in
GitHub Secrets (CI) or a git-ignored .env file (local runs), never in code.

Smoke test:  python -m src.llm "Say hi"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import requests

from src.config import load_config

TIMEOUT_SECONDS = 60
RETRY_STATUSES = (429, 500, 503)  # rate limited, or the model is temporarily overloaded
MAX_RETRIES = 5
MAX_WAIT_SECONDS = 90


class LLMError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set. See .env.example.")
    return key


def _headers(key: str) -> dict[str, str]:
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def build_payload(messages: list[dict[str, str]], llm_cfg: dict) -> dict:
    """Convert OpenAI-style messages (system/user/assistant) to a Gemini request body."""
    system = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    payload: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": llm_cfg.get("temperature", 0.0),
            "maxOutputTokens": llm_cfg.get("max_tokens", 2048),
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
    return payload


def retry_delay_seconds(body: str) -> float | None:
    """How long Gemini asks us to wait, from RetryInfo ("41s") or the message text."""
    try:
        error = json.loads(body).get("error", {})
    except (ValueError, AttributeError):
        return None
    for detail in error.get("details", []):
        if detail.get("@type", "").endswith("RetryInfo") and "retryDelay" in detail:
            return float(str(detail["retryDelay"]).rstrip("s"))
    m = re.search(r"retry in ([\d.]+)s", error.get("message", ""))
    return float(m.group(1)) if m else None


def is_daily_quota(body: str) -> bool:
    """A per-day quota resets once a day, so retrying within minutes cannot help."""
    return "PerDay" in body


def chat(messages: list[dict[str, str]], llm_cfg: dict | None = None, sleep=time.sleep) -> str:
    """Send chat messages and return the reply text.

    On 429/500/503 it waits as long as Gemini asks (plus a second), or 2, 4, 8, ... s when
    no delay is given, up to MAX_RETRIES times. A daily quota error fails immediately.
    """
    llm_cfg = llm_cfg or load_config()["llm"]
    url = f"{llm_cfg['endpoint'].rstrip('/')}/models/{llm_cfg['model']}:generateContent"
    key = _api_key()
    for attempt in range(MAX_RETRIES + 1):
        # Don't follow redirects: requests turns a redirected POST into a GET, which hides
        # the real error behind whatever page the redirect lands on.
        resp = requests.post(
            url,
            headers=_headers(key),
            json=build_payload(messages, llm_cfg),
            timeout=TIMEOUT_SECONDS,
            allow_redirects=False,
        )
        if resp.status_code == 429 and is_daily_quota(resp.text):
            raise LLMError(
                "Gemini daily free-tier quota is used up for this model; it resets daily. "
                f"Details: {resp.text[:300]}"
            )
        if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
            delay = retry_delay_seconds(resp.text)
            sleep(min(MAX_WAIT_SECONDS, delay + 1 if delay is not None else 2 ** (attempt + 1)))
            continue
        break
    if resp.status_code != 200:
        raise LLMError(f"Gemini returned {resp.status_code}: {resp.text[:500]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMError(
            f"Gemini returned 200 but the body is not valid JSON "
            f"(Content-Type: {resp.headers.get('Content-Type')!r}, body: {resp.text[:500]!r})"
        ) from exc
    try:
        candidate = data["candidates"][0]
        text = "".join(p.get("text", "") for p in candidate["content"]["parts"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Gemini returned an unexpected response: {str(data)[:500]}") from exc
    if not text:
        reason = candidate.get("finishReason")
        raise LLMError(f"Gemini returned no text (finishReason={reason!r})")
    return text


def list_models(llm_cfg: dict | None = None) -> list[str]:
    """Model IDs this key can use for generateContent (useful when a model is retired)."""
    llm_cfg = llm_cfg or load_config()["llm"]
    resp = requests.get(
        f"{llm_cfg['endpoint'].rstrip('/')}/models",
        headers=_headers(_api_key()),
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return sorted(
        m["name"].removeprefix("models/")
        for m in resp.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )


def _print_available_models() -> None:
    try:
        print("Available models:", ", ".join(list_models()), file=sys.stderr)
    except (LLMError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"Could not list models: {exc}", file=sys.stderr)


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Say hi"
    try:
        print(chat([{"role": "user", "content": prompt}]))
    except LLMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        _print_available_models()
        sys.exit(1)


if __name__ == "__main__":
    main()
