"""Monitoring page: usage, refusals, latency and retrieval confidence from the question log."""

import altair as alt
import pandas as pd
import streamlit as st
from client import ApiError, call, load_config, series_color

st.set_page_config(page_title="Monitoring · Filings Q&A", page_icon="📈", layout="wide")
cfg = load_config()["app"]

st.title("Monitoring")
days = st.radio("Period", [7, 14, 30], index=1, horizontal=True, format_func=lambda d: f"{d} days")

try:
    s = call("GET", "/stats", params={"days": days})
except ApiError as exc:
    st.warning(str(exc))
    st.stop()


def pct(x):
    return "–" if x is None else f"{x:.0%}"


def secs(x):
    return "–" if x is None else f"{x / 1000:.1f} s"


k = st.columns(4)
k[0].metric(
    "Questions",
    s["total"],
    help=f"{s['answered']} answered · {s['refused']} refused · {s['errors']} errors",
)
k[1].metric(
    "Refusal rate", pct(s["refusal_rate"]), help="Share answered 'Not found in the filings.'"
)
k[2].metric("Latency p50 / p95", f"{secs(s['latency_p50_ms'])} / {secs(s['latency_p95_ms'])}")
k[3].metric(
    "Low-confidence retrievals",
    pct(s["low_confidence_rate"]),
    help=f"Best chunk scored below {cfg['low_confidence_score']:.2f}. "
    "A rising share means questions drift away from what the filings cover.",
)

if s["total"] == 0:
    st.info("No questions in this period yet. Ask one on the Ask page.")
    st.stop()

daily = pd.DataFrame(s["daily"])
daily["date"] = pd.to_datetime(daily["date"])
daily["day"] = daily["date"].dt.strftime("%b %d")  # labels, so each bar gets a full-width slot
daily["latency_p50_s"] = daily["latency_p50_ms"] / 1000

color = series_color()
left, right = st.columns(2)
with left:
    st.subheader("Questions per day")
    bars = (
        alt.Chart(daily)
        .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("day:O", sort=None, title=None),
            y=alt.Y("questions:Q", title="questions"),
            tooltip=["day", "questions", "refused", "errors"],
        )
    )
    st.altair_chart(bars, width="stretch")
with right:
    st.subheader("Median latency per day")
    line = (
        alt.Chart(daily)
        .mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=color))
        .encode(
            x=alt.X("day:O", sort=None, title=None),
            y=alt.Y("latency_p50_s:Q", title="seconds"),
            tooltip=["day", alt.Tooltip("latency_p50_s:Q", title="median s", format=".1f")],
        )
    )
    st.altair_chart(line, width="stretch")

st.subheader("Recent questions")
st.caption("Outcome and timing only; question text is not shown publicly.")
recent = pd.DataFrame(s["recent"])
recent["ts"] = pd.to_datetime(recent["ts"]).dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(
    recent,
    hide_index=True,
    width="stretch",
    column_config={
        "ts": st.column_config.TextColumn("time (UTC)"),
        "top_score": st.column_config.NumberColumn("top score", format="%.2f"),
        "latency_ms": st.column_config.NumberColumn("latency (ms)"),
    },
)
with st.expander("Daily table"):
    st.dataframe(daily.drop(columns=["latency_p50_ms", "day"]), hide_index=True, width="stretch")
