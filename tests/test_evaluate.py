import json

import numpy as np
import pytest

from src import evaluate
from src.config import load_config

CFG = load_config()

SUPPLY = "Battery cell supply could be disrupted by suppliers."
RATES = "Rising interest rates may reduce consumer spending."
CHUNKS = [
    {"id": "TSLA_1A_0000", "ticker": "TSLA", "text": SUPPLY, "source_url": "u"},
    {"id": "TSLA_1A_0001", "ticker": "TSLA", "text": RATES, "source_url": "u"},
    {"id": "WMT_1A_0000", "ticker": "WMT", "text": SUPPLY, "source_url": "u"},
]
VECS = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)


def embed_fn(q):
    return [1, 0, 0] if "battery" in q.lower() else [0, 1, 0]


def test_memory_index_ranks_and_filters_by_company():
    idx = evaluate.MemoryIndex(CHUNKS, VECS, embed_fn)
    rows = idx.search("battery supply?", "main", ["TSLA"], 2)
    assert [r[0] for r in rows] == ["TSLA_1A_0000", "TSLA_1A_0001"]  # no WMT chunk
    rows = idx.search("battery supply?", "main", None, 3)
    assert {r[0] for r in rows[:2]} == {"TSLA_1A_0000", "WMT_1A_0000"}


def test_first_hit_rank_uses_quote_not_chunk_id():
    retrieved = [{"text": "nothing here at all today"}, {"text": CHUNKS[1]["text"]}]
    assert evaluate.first_hit_rank(retrieved, "Rising interest rates may reduce consumer") == 2
    assert evaluate.first_hit_rank(retrieved, "a quote that is not present anywhere") is None


def test_retrieval_metrics():
    m = evaluate.retrieval_metrics([1, 2, None, 6], k=8)
    assert m["hit_rate_at_k"] == 0.75
    assert m["hit_rate_at_5"] == 0.5
    assert m["mrr"] == pytest.approx((1 + 0.5 + 1 / 6) / 4)


def test_evaluate_scores_retrieval_and_refusals():
    idx = evaluate.MemoryIndex(CHUNKS, VECS, embed_fn)
    items = [
        {"id": "q01", "question": "What does Tesla say about battery supply?",
         "quote": "Battery cell supply could be disrupted", "answerable": True},
        {"id": "q02", "question": "What does Tesla say about interest rates?",
         "quote": "Rising interest rates may reduce consumer", "answerable": True},
        {"id": "q03", "question": "Who is Tesla's CEO?", "answerable": False},
        {"id": "q04", "question": "What is Tesla's revenue?", "answerable": False},
    ]  # fmt: skip
    answers = {"Who is Tesla's CEO?": True, "What is Tesla's revenue?": False}
    metrics, rows = evaluate.evaluate(
        items, CFG, idx.search, k=1,
        answer_fn=lambda q: {"refused": answers[q], "answer": "x"},
    )  # fmt: skip
    assert metrics["hit_rate_at_k"] == 1.0 and metrics["mrr"] == 1.0
    assert metrics["refusal_rate"] == 0.5
    assert rows[0] == {"id": "q01", "rank": 1, "retrieved": ["TSLA_1A_0000"]}


def test_gate_failures():
    thresholds = {"hit_rate_at_k": 0.8, "mrr": 0.6}
    assert evaluate.gate_failures({"hit_rate_at_k": 0.85, "mrr": 0.7}, thresholds) == []
    failures = evaluate.gate_failures({"hit_rate_at_k": 0.5, "mrr": 0.7}, thresholds)
    assert failures == ["hit_rate_at_k 0.500 is below the threshold 0.800"]


def test_build_chunks_uses_requested_size(tmp_path):
    (tmp_path / "TSLA.txt").write_text(" ".join(f"w{i}" for i in range(500)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"ticker": "TSLA", "company": "Tesla", "url": "u",
                                     "filing_date": "2026-01-29"}]))  # fmt: skip
    small = evaluate.build_chunks(200, 50, sections_dir=tmp_path, manifest=manifest)
    large = evaluate.build_chunks(400, 50, sections_dir=tmp_path, manifest=manifest)
    assert len(small) == 3 and len(large) == 2


def test_markdown_table():
    table = evaluate.markdown_table({"B": {"hit_rate_at_k": 0.9, "mrr": 0.75, "n_chunks": 328}})
    assert "| B | 0.900 |  | 0.750 |  | 328 |" in table


def test_experiments_and_thresholds_files_are_valid():
    import yaml

    exps = yaml.safe_load(evaluate.EXPERIMENTS_PATH.read_text())["experiments"]
    assert len(exps) == 3
    for params in exps.values():
        assert set(params) == {"size_words", "overlap_words", "top_k"}
