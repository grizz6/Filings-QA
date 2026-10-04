import json

from src import testset

CHUNK_TEXT = (
    "We depend on a limited number of suppliers for battery cells. "
    "Any disruption in the supply of battery cells from our suppliers could limit "
    "production of our vehicles and energy storage products."
)
CHUNK = {"id": "TSLA_1A_0007", "ticker": "TSLA", "chunk_index": 7, "text": CHUNK_TEXT}
LLM_CFG = {"model": "m"}


def reply(question="What does Tesla say about battery cell suppliers?", quote=None):
    quote = quote or (
        "Any disruption in the supply of battery cells from our suppliers could limit "
        "production of our vehicles"
    )
    return json.dumps(
        {"question": question, "answer": "It could limit production.", "quote": quote}
    )


def test_contains_quote_tolerates_case_quotes_and_whitespace():
    text = "Our results depend on consumers’ demand — and on   pricing."
    assert testset.contains_quote(text, "our results depend on consumers' demand - and on pricing.")
    assert not testset.contains_quote(text, "Our results depend on investor demand")
    assert not testset.contains_quote(text, "pricing")  # too short to prove anything


def test_pick_chunks_evenly_spaced_per_company():
    chunks = [{"id": f"T_{i}", "ticker": "T", "chunk_index": i, "text": ""} for i in range(20)]
    chunks += [{"id": f"U_{i}", "ticker": "U", "chunk_index": i, "text": ""} for i in range(3)]
    picked = testset.pick_chunks(chunks, per_company=4)
    t = [c["chunk_index"] for c in picked if c["ticker"] == "T"]
    assert t == [4, 8, 12, 16]  # skips chunk 0, spread across the section
    assert len([c for c in picked if c["ticker"] == "U"]) <= 3  # never duplicates


def test_draft_candidate_grounded():
    cand = testset.draft_candidate(CHUNK, "Tesla", lambda m, c, json_mode: reply(), LLM_CFG)
    assert cand["grounded"] and cand["question"].startswith("What does Tesla")


def test_draft_candidate_retries_when_quote_is_invented():
    replies = [reply(quote="Tesla has many battery suppliers worldwide and is safe."), reply()]
    seen = []

    def chat(messages, cfg, json_mode):
        seen.append(messages)
        return replies.pop(0)

    cand = testset.draft_candidate(CHUNK, "Tesla", chat, LLM_CFG)
    assert cand["grounded"]
    assert "word for word" in seen[1][-1]["content"]  # second call explains the problem


def test_draft_candidate_gives_up_after_two_bad_replies():
    cand = testset.draft_candidate(CHUNK, "Tesla", lambda m, c, json_mode: "not json", LLM_CFG)
    assert cand == {"question": "", "answer": "", "quote": "", "grounded": False}


def test_make_candidates_spaces_calls():
    chunks = [{**CHUNK, "id": f"TSLA_1A_{i:04d}", "chunk_index": i} for i in range(10)]
    sleeps = []
    out = testset.make_candidates(
        chunks, {"TSLA": "Tesla"}, lambda m, c, json_mode: reply(), LLM_CFG, sleep=sleeps.append
    )
    assert len(out) == 4 and all(c["ticker"] == "TSLA" for c in out)
    assert sleeps == [testset.SECONDS_BETWEEN_CALLS] * 3


def _item(i, answerable=True, **kw):
    base = {"id": f"q{i:02d}", "question": "Q?", "ticker": "TSLA", "answerable": answerable}
    if answerable:
        base.update(quote="one two three four five six", expected_chunk_ids=["TSLA_1A_0001"])
    return {**base, **kw}


def test_check_test_set_valid():
    items = [_item(i) for i in range(40)] + [_item(i, False) for i in range(40, 50)]
    assert testset.check_test_set(items, {"TSLA"}) == []


def test_check_test_set_reports_problems():
    items = [_item(i) for i in range(39)] + [_item(1, quote="short", ticker="ZZZ")]
    problems = testset.check_test_set(items, {"TSLA"})
    assert "duplicate ids" in problems
    assert any("got 40 + 0" in p for p in problems)
    assert any("unknown ticker ZZZ" in p for p in problems)
    assert any("needs a quote" in p for p in problems)
