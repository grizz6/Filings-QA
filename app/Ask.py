"""Ask page: question in, cited answer out."""

import streamlit as st
from client import ApiError, call, load_config

EXAMPLES = [
    "What does Tesla say about supply chain risk?",
    "How could cybersecurity incidents affect Walmart?",
    "What risks does Delta Air Lines see from fuel prices?",
    "How does Pfizer describe risks from drug pricing pressure?",
]

st.set_page_config(page_title="Filings Q&A", page_icon="📄", layout="centered")
cfg = load_config()

st.title("Filings Q&A")
st.write(
    "Ask about the **Risk Factors** section (Item 1A) of the latest annual report (10-K) of "
    "these companies. Answers come only from the filings and cite them."
)
st.caption(" · ".join(f"{name} ({t})" for t, name in cfg["companies"].items()))

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
        st.session_state.question = example

with st.form("ask"):
    question = st.text_input(
        "Your question",
        key="question",
        max_chars=cfg["app"]["max_question_chars"],
        placeholder="e.g. What does Netflix say about competition?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Searching the filings and writing a cited answer…"):
        try:
            result = call("POST", "/ask", json={"question": question})
        except ApiError as exc:
            st.warning(str(exc))
            st.stop()

    if result["refused"]:
        st.info(
            "**Not found in the filings.** The Risk Factors sections do not answer this "
            "(they cover risks, not figures like revenue or names of executives)."
        )
    else:
        st.markdown(result["answer"])
        st.markdown("**Sources**")
        for c in result["citations"]:
            st.markdown(
                f"[{c['n']}] {c['ticker']} · `{c['id']}` · [open the 10-K]({c['source_url']})"
            )

    with st.expander(f"Retrieved excerpts ({len(result['sources'])})"):
        for i, s in enumerate(result["sources"], 1):
            st.markdown(f"**[{i}] `{s['id']}`** · similarity {s['score']:.2f}")
            st.caption(s["snippet"])
    st.caption(f"Answered in {result['latency_ms'] / 1000:.1f} s · model {cfg['llm']['model']}")

st.divider()
st.caption(
    "Questions are logged (text, timing, scores) to monitor the app; see the Monitoring page. "
    "Not investment advice."
)
