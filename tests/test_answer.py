from src import answer
from src.config import load_config

CFG = load_config()
CHUNKS = [
    {"id": f"TSLA_1A_000{i}", "ticker": "TSLA", "text": f"text {i}",
     "source_url": f"https://sec/{i}", "score": 0.9 - i / 10}
    for i in range(1, 4)
]  # fmt: skip


def fake_retrieve(question, cfg):
    return CHUNKS


def test_prompt_numbers_excerpts_and_states_rules():
    msgs = answer.build_messages("Q?", CHUNKS)
    assert msgs[0]["role"] == "system"
    assert answer.NOT_FOUND in msgs[0]["content"]
    assert "[1] (TSLA, TSLA_1A_0001)\ntext 1" in msgs[1]["content"]
    assert "[3] (TSLA, TSLA_1A_0003)" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: Q?")


def test_answer_maps_citations_to_chunks():
    reply = "Tesla depends on suppliers [2]. Shortages hurt output [2][3]. Bogus [9]."
    result = answer.answer("Q?", CFG, retrieve_fn=fake_retrieve, chat_fn=lambda m, c: reply)
    assert not result["refused"]
    assert [c["id"] for c in result["citations"]] == ["TSLA_1A_0002", "TSLA_1A_0003"]
    assert result["citations"][0]["source_url"] == "https://sec/2"


def test_refusal_detected():
    for reply in [
        "Not found in the filings.",
        "not found in the filings",
        " Not found in the filings. ",
    ]:
        result = answer.answer(
            "Q?", CFG, retrieve_fn=fake_retrieve, chat_fn=lambda m, c, r=reply: r
        )
        assert result["refused"]
        assert result["answer"] == answer.NOT_FOUND
        assert result["citations"] == []


def test_no_chunks_refuses_without_calling_llm():
    def boom(m, c):
        raise AssertionError("LLM should not be called")

    result = answer.answer("Q?", CFG, retrieve_fn=lambda q, cfg: [], chat_fn=boom)
    assert result["refused"]


def test_format_result_lists_citations():
    result = answer.answer("Q?", CFG, retrieve_fn=fake_retrieve, chat_fn=lambda m, c: "Answer [1].")
    out = answer.format_result(result)
    assert "A: Answer [1]." in out
    assert "[1] TSLA_1A_0001  https://sec/1" in out
