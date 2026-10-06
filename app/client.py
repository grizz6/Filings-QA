"""Shared helpers for the Streamlit pages: talk to the API, read the config.

Two ways to reach the API (src/api.py), with the same validation, limits and logging:
- API_URL set (Docker image): over HTTP to the API process next to the UI.
- API_URL not set (Streamlit Community Cloud, one process): the same FastAPI app is
  called in this process. Secrets come from Streamlit's secrets store.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

API_URL = os.environ.get("API_URL", "").rstrip("/")
TIMEOUT_SECONDS = 180  # an answer can wait out a Gemini rate limit
SECRETS = ("GEMINI_API_KEY", "SUPABASE_DB_URL")


class ApiError(RuntimeError):
    pass


def secrets_to_env() -> None:
    """Copy Streamlit secrets into the environment, where src/ reads them (env wins)."""
    try:
        import streamlit as st

        for name in SECRETS:
            if name in st.secrets and not os.environ.get(name):
                os.environ[name] = str(st.secrets[name])
    except Exception:  # no secrets file (Docker, tests): the environment is the source
        pass


def _local_api():
    """The in-process API, rebuilt when the secrets it was built with change.

    On Streamlit Cloud the app can start before its secrets are saved; caching on which
    secrets are present keeps it from staying stuck without the database.
    """
    secrets_to_env()
    return _build_local_api(tuple(bool(os.environ.get(name, "").strip()) for name in SECRETS))


@lru_cache(maxsize=4)
def _build_local_api(secrets_present: tuple[bool, ...]):
    from fastapi.testclient import TestClient

    from src.api import create_app

    # raise_server_exceptions=False: an unexpected error becomes a 500, as over HTTP.
    return TestClient(create_app(), raise_server_exceptions=False)


def call(method: str, path: str, **kwargs) -> dict:
    if not API_URL:
        resp = _local_api().request(method, path, **kwargs)  # in-process: no network timeout
    else:
        try:
            resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT_SECONDS, **kwargs)
        except requests.RequestException:
            raise ApiError(
                "The API is not reachable. It may still be starting; retry in a minute."
            ) from None
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, list):  # validation errors from FastAPI
            detail = "; ".join(d.get("msg", "") for d in detail)
        raise ApiError(detail or f"The API returned {resp.status_code}.")
    return resp.json()


def setup_problems() -> list[str]:
    """Names of the secrets the API reports as missing (empty when the app is ready)."""
    health = call("GET", "/health")
    flags = {"GEMINI_API_KEY": "llm_key_configured", "SUPABASE_DB_URL": "db_configured"}
    return [name for name, flag in flags.items() if not health.get(flag)]


def safe_markdown(text: str) -> str:
    """Model output as plain text with [n] citations: no images, outside links or HTML."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"<[^>]+>", "", text)  # HTML tags
    return re.sub(r"https?://\S+", "", text).strip()


def series_color() -> str:
    """Categorical slot 1 (blue), stepped for the active light or dark theme."""
    try:
        import streamlit as st

        if st.context.theme.type == "dark":
            return "#3987e5"
    except Exception:
        pass
    return "#2a78d6"


__all__ = [
    "ApiError",
    "call",
    "load_config",
    "safe_markdown",
    "secrets_to_env",
    "series_color",
    "setup_problems",
]
