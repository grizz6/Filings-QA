"""Shared helpers for the Streamlit pages: talk to the API, read the config."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 180  # an answer can wait out a Gemini rate limit


class ApiError(RuntimeError):
    pass


def call(method: str, path: str, **kwargs) -> dict:
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


def series_color() -> str:
    """Categorical slot 1 (blue), stepped for the active light or dark theme."""
    try:
        import streamlit as st

        if st.context.theme.type == "dark":
            return "#3987e5"
    except Exception:
        pass
    return "#2a78d6"


__all__ = ["ApiError", "call", "load_config", "series_color"]
