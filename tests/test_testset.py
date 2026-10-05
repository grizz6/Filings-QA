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


def test_verify_quotes_against_chunks():
    chunks = [{"id": "TSLA_1A_0007", "text": CHUNK_TEXT}]
    good = _item(1, quote="Any disruption in the supply of battery cells",
                 expected_chunk_ids=["TSLA_1A_0007"])  # fmt: skip
    bad = _item(2, quote="Tesla builds rockets on the moon every year",
                expected_chunk_ids=["TSLA_1A_0007"])  # fmt: skip
    missing = _item(3, expected_chunk_ids=["TSLA_1A_9999"])
    skipped = _item(4, answerable=False)
    problems = testset.verify_quotes([good, bad, missing, skipped], chunks)
    assert problems == [
        "q02: quote not found in TSLA_1A_0007",
        "q03: chunk TSLA_1A_9999 does not exist",
    ]


def test_committed_test_set_is_valid():
    from src.config import load_config

    items = testset.load_jsonl(testset.TEST_SET_PATH)
    assert testset.check_test_set(items, set(load_config()["companies"])) == []


SECTION = CHUNK_TEXT + " Our Chief Executive Officer is Elon Musk."


def _audit_reply(answerable, evidence=""):
    return lambda m, c, json_mode: json.dumps({"answerable": answerable, "evidence": evidence})


def test_audit_item_verdicts():
    item = {"id": "q01", "question": "Q?"}
    real = "Any disruption in the supply of battery cells from our suppliers could limit"
    assert testset.audit_item(item, SECTION, "Tesla", _audit_reply(True, real), {})["verdict"] == (
        "answered"
    )
    invented = "Tesla has no supply chain risks at all in any market"
    assert (
        testset.audit_item(item, SECTION, "Tesla", _audit_reply(True, invented), {})["verdict"]
        == "unclear"
    )
    assert testset.audit_item(item, SECTION, "Tesla", _audit_reply(False), {})["verdict"] == (
        "not_answered"
    )
    bad_json = lambda m, c, json_mode: "nope"  # noqa: E731
    assert testset.audit_item(item, SECTION, "Tesla", bad_json, {})["verdict"] == "unclear"


def test_audit_accepts_evidence_with_a_long_verbatim_span():
    # Seen in CI (q07): the model copied the start of a sentence and paraphrased the end.
    item = {"id": "q07", "question": "Q?"}
    partly_copied = (
        "Any disruption in the supply of battery cells from our suppliers could limit "
        "output and hurt our margins badly."
    )
    result = testset.audit_item(item, SECTION, "Tesla", _audit_reply(True, partly_copied), {})
    assert result["verdict"] == "answered"
    assert testset.grounded_in(SECTION, "Elon Musk is the Chief Executive Officer.") is False


def test_audit_item_reports_llm_errors_without_a_verdict():
    def overloaded(m, c, json_mode):
        raise testset.LLMError("Gemini returned 503: high demand")

    result = testset.audit_item({"id": "q01", "question": "Q?"}, SECTION, "Tesla", overloaded, {})
    assert result["verdict"] == "error" and "503" in result["evidence"]


def test_audit_problems_ignores_errors():
    items = [{"id": "q01", "answerable": True}]
    assert testset.audit_problems(items, [{"id": "q01", "verdict": "error", "evidence": ""}]) == []


def test_audit_problems_flags_both_directions():
    items = [
        {"id": "q01", "answerable": True},
        {"id": "q02", "answerable": False},
        {"id": "q03", "answerable": False},
    ]
    results = [
        {"id": "q01", "verdict": "not_answered", "evidence": ""},
        {"id": "q02", "verdict": "answered", "evidence": "Our CEO is Elon Musk."},
        {"id": "q03", "verdict": "not_answered", "evidence": ""},
    ]
    assert testset.audit_problems(items, results) == [
        "q01: expected answerable, audit says not_answered",
        "q02: expected unanswerable, but section says: Our CEO is Elon Musk.",
    ]


def test_run_audit_reads_each_company_section(tmp_path):
    (tmp_path / "TSLA.txt").write_text(SECTION)
    items = [{"id": "q01", "ticker": "TSLA", "question": "Who is the CEO?", "answerable": False}]
    ceo = "Our Chief Executive Officer is Elon Musk."
    results = testset.run_audit(
        items, {"TSLA": "Tesla"}, tmp_path, _audit_reply(True, ceo), {}, sleep=lambda s: None
    )
    assert results == [{"id": "q01", "verdict": "answered", "evidence": ceo}]


def test_run_audit_reports_each_result_as_it_goes(tmp_path):
    (tmp_path / "TSLA.txt").write_text(SECTION)
    items = [{"id": f"q0{i}", "ticker": "TSLA", "question": "Q?"} for i in (1, 2)]
    seen = []
    testset.run_audit(
        items, {"TSLA": "Tesla"}, tmp_path, _audit_reply(False), {},
        sleep=lambda s: None, on_result=seen.append,
    )  # fmt: skip
    assert [r["id"] for r in seen] == ["q01", "q02"]
