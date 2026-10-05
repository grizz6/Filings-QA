"""Build and check the evaluation test set (Day 6).

Step 1, in the cloud: draft candidate questions from real chunks.
    python -m src.testset candidates   # data/chunks.jsonl -> data/eval/candidates.jsonl
  For a few evenly spaced chunks per company, Gemini writes a question, a short answer, and
  an exact supporting quote copied from the chunk. A candidate is kept only if that quote
  really appears in the chunk text ("grounded"), so every answer is traceable to the filing.

Step 2, by a person: review the candidates, edit or drop weak ones, add unanswerable
  questions, and save the result as eval/test_set.jsonl.

Step 3: validate the file.
    python -m src.testset check        # schema + 40 answerable / 10 unanswerable split
    python -m src.testset verify       # every quote appears in its expected chunk
    python -m src.testset audit        # full-section check of every question (LLM)

The audit gives Gemini the company's ENTIRE Risk Factors section and asks whether it
answers the question, with an exact supporting sentence. The sentence is checked against the
section text, so the audit cannot be satisfied by an invented quote. It flags unanswerable
questions that the section does answer, and answerable ones it does not support.

Each answerable item stores its quote, so a retrieved chunk counts as correct when it
contains the quote. That keeps the test set valid when chunk sizes (and IDs) change.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from src.config import ROOT, load_config

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
SECTIONS_DIR = ROOT / "data" / "sections"
CANDIDATES_PATH = ROOT / "data" / "eval" / "candidates.jsonl"
TEST_SET_PATH = ROOT / "eval" / "test_set.jsonl"

PER_COMPANY = 4
SECONDS_BETWEEN_CALLS = 5  # free tier allows ~15 requests/minute for gemini-3.1-flash-lite
# Audit calls send a whole Risk Factors section (up to ~20k tokens), so they are spaced
# further apart to stay under the free tier's tokens-per-minute limit.
AUDIT_SECONDS_BETWEEN_CALLS = 7
EXPECTED_ANSWERABLE = 40
EXPECTED_UNANSWERABLE = 10

PROMPT = """You are writing an evaluation question for a search system over the "Risk \
Factors" section of {company}'s annual report (Form 10-K).

Read the excerpt and write ONE question that:
- an investor might naturally ask,
- names the company ({company}),
- is answered specifically by this excerpt (not general knowledge).

Return JSON with exactly these keys:
- "question": the question.
- "answer": a 1-2 sentence answer based only on the excerpt.
- "quote": one sentence copied EXACTLY, word for word, from the excerpt that supports the
  answer (10 to 40 words, no changes, no ellipses).

Excerpt:
{text}"""


def normalize(text: str) -> str:
    """Lowercase, unify quote characters and dashes, collapse whitespace."""
    text = text.lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-")):  # fmt: skip
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def contains_quote(chunk_text: str, quote: str) -> bool:
    q = normalize(quote).strip(" .\"'")
    return len(q.split()) >= 5 and q in normalize(chunk_text)


def pick_chunks(chunks: list[dict], per_company: int = PER_COMPANY) -> list[dict]:
    """Evenly spaced chunks per company (skips the very first, usually boilerplate)."""
    by_ticker: dict[str, list[dict]] = {}
    for c in chunks:
        by_ticker.setdefault(c["ticker"], []).append(c)
    picked = []
    for group in by_ticker.values():
        group = sorted(group, key=lambda c: c["chunk_index"])
        n = len(group)
        idxs = sorted(
            {min(n - 1, round((i + 1) * n / (per_company + 1))) for i in range(per_company)}
        )
        picked.extend(group[i] for i in idxs)
    return picked


def parse_candidate(raw: str) -> dict:
    data = json.loads(raw)
    if not all(
        isinstance(data.get(k), str) and data[k].strip() for k in ("question", "answer", "quote")
    ):
        raise ValueError(f"missing keys in {raw[:200]}")
    return {k: data[k].strip() for k in ("question", "answer", "quote")}


def draft_candidate(chunk: dict, company: str, chat_fn, llm_cfg: dict) -> dict:
    """Ask the LLM for a question; retry once if the quote is not found verbatim."""
    messages = [{"role": "user", "content": PROMPT.format(company=company, text=chunk["text"])}]
    for _attempt in range(2):
        try:
            cand = parse_candidate(chat_fn(messages, llm_cfg, json_mode=True))
        except ValueError:  # includes json.JSONDecodeError
            continue
        if contains_quote(chunk["text"], cand["quote"]):
            return {**cand, "grounded": True}
        messages = messages + [
            {"role": "assistant", "content": json.dumps(cand)},
            {"role": "user", "content": "That quote is not in the excerpt word for word. "
             "Return the same JSON with a quote copied exactly from the excerpt."},
        ]  # fmt: skip
    return {"question": "", "answer": "", "quote": "", "grounded": False}


def make_candidates(chunks, companies, chat_fn, llm_cfg, sleep=time.sleep) -> list[dict]:
    out = []
    for i, chunk in enumerate(pick_chunks(chunks)):
        if i:
            sleep(SECONDS_BETWEEN_CALLS)
        cand = draft_candidate(chunk, companies[chunk["ticker"]], chat_fn, llm_cfg)
        out.append({"ticker": chunk["ticker"], "chunk_id": chunk["id"], **cand})
    return out


AUDIT_PROMPT = """Below is the complete "Risk Factors" section of {company}'s annual report.

Does this section answer the question? Answer only from the section.

Question: {question}

Return JSON with exactly these keys:
- "answerable": true if the section contains the answer, otherwise false.
- "evidence": if answerable, one sentence copied EXACTLY, word for word, from the section that
  answers the question; otherwise "".

Section:
{section}"""


def audit_item(item: dict, section: str, company: str, chat_fn, llm_cfg: dict) -> dict:
    """Ask whether the full section answers the question; verify the evidence sentence."""
    prompt = AUDIT_PROMPT.format(company=company, question=item["question"], section=section)
    try:
        data = json.loads(chat_fn([{"role": "user", "content": prompt}], llm_cfg, json_mode=True))
        says_answerable = bool(data.get("answerable"))
        evidence = str(data.get("evidence", "")).strip()
    except (ValueError, AttributeError):
        return {"id": item["id"], "verdict": "unclear", "evidence": ""}
    if says_answerable and contains_quote(section, evidence):
        verdict = "answered"
    elif says_answerable:
        verdict = "unclear"  # claims an answer but the sentence is not in the section
    else:
        verdict = "not_answered"
    return {"id": item["id"], "verdict": verdict, "evidence": evidence}


def audit_problems(items: list[dict], results: list[dict]) -> list[str]:
    """Mismatches between what the test set says and what the full-section audit found."""
    by_id = {r["id"]: r for r in results}
    problems = []
    for it in items:
        r = by_id[it["id"]]
        if it["answerable"] and r["verdict"] != "answered":
            problems.append(f"{it['id']}: expected answerable, audit says {r['verdict']}")
        if not it["answerable"] and r["verdict"] == "answered":
            problems.append(f"{it['id']}: expected unanswerable, but section says: {r['evidence']}")
    return problems


def run_audit(
    items, companies, sections_dir, chat_fn, llm_cfg, sleep=time.sleep, on_result=None
) -> list[dict]:
    """Audit every item; on_result sees each verdict as soon as it arrives (for live logs)."""
    results = []
    for i, it in enumerate(items):
        if i:
            sleep(AUDIT_SECONDS_BETWEEN_CALLS)
        section = (sections_dir / f"{it['ticker']}.txt").read_text()
        results.append(audit_item(it, section, companies[it["ticker"]], chat_fn, llm_cfg))
        if on_result:
            on_result(results[-1])
    return results


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_test_set(items: list[dict], tickers: set[str]) -> list[str]:
    """Return a list of problems (empty means the test set is valid)."""
    problems = []
    ids = [it.get("id") for it in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    answerable = [it for it in items if it.get("answerable")]
    unanswerable = [it for it in items if it.get("answerable") is False]
    if len(answerable) != EXPECTED_ANSWERABLE or len(unanswerable) != EXPECTED_UNANSWERABLE:
        problems.append(
            f"expected {EXPECTED_ANSWERABLE} answerable + {EXPECTED_UNANSWERABLE} unanswerable, "
            f"got {len(answerable)} + {len(unanswerable)}"
        )
    for it in items:
        tag = it.get("id", "?")
        if not str(it.get("question", "")).strip():
            problems.append(f"{tag}: empty question")
        if it.get("ticker") is not None and it["ticker"] not in tickers:
            problems.append(f"{tag}: unknown ticker {it['ticker']}")
        if it.get("answerable"):
            if len(str(it.get("quote", "")).split()) < 5:
                problems.append(f"{tag}: answerable item needs a quote of 5+ words")
            if not it.get("expected_chunk_ids"):
                problems.append(f"{tag}: answerable item needs expected_chunk_ids")
    return problems


def verify_quotes(items: list[dict], chunks: list[dict]) -> list[str]:
    """Check each answerable quote against the real chunk text; return problems."""
    by_id = {c["id"]: c for c in chunks}
    problems = []
    for it in items:
        if not it.get("answerable"):
            continue
        for cid in it["expected_chunk_ids"]:
            chunk = by_id.get(cid)
            if chunk is None:
                problems.append(f"{it['id']}: chunk {cid} does not exist")
            elif not contains_quote(chunk["text"], it["quote"]):
                problems.append(f"{it['id']}: quote not found in {cid}")
    return problems


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = load_config()
    if cmd == "candidates":
        from src.llm import chat

        cands = make_candidates(load_jsonl(CHUNKS_PATH), cfg["companies"], chat, cfg["llm"])
        CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")
        for c in cands:
            print(json.dumps(c, ensure_ascii=False))
        ok = sum(c["grounded"] for c in cands)
        print(f"{ok}/{len(cands)} candidates grounded -> {CANDIDATES_PATH.relative_to(ROOT)}")
    elif cmd == "verify":
        problems = verify_quotes(load_jsonl(TEST_SET_PATH), load_jsonl(CHUNKS_PATH))
        if problems:
            print("\n".join(f"ERROR: {p}" for p in problems), file=sys.stderr)
            sys.exit(1)
        print("every answerable quote was found in its expected chunk")
    elif cmd == "audit":
        from src.llm import chat

        items = load_jsonl(TEST_SET_PATH)
        results = run_audit(
            items, cfg["companies"], SECTIONS_DIR, chat, cfg["llm"],
            on_result=lambda r: print(json.dumps(r, ensure_ascii=False), flush=True),
        )  # fmt: skip
        problems = audit_problems(items, results)
        if problems:
            print("\n".join(f"ERROR: {p}" for p in problems), file=sys.stderr)
            sys.exit(1)
        print(f"audit: all {len(items)} questions agree with the full Risk Factors sections")
    elif cmd == "check":
        problems = check_test_set(load_jsonl(TEST_SET_PATH), set(cfg["companies"]))
        if problems:
            print("\n".join(f"ERROR: {p}" for p in problems), file=sys.stderr)
            sys.exit(1)
        print(f"{TEST_SET_PATH.relative_to(ROOT)} is valid")
    else:
        sys.exit("usage: python -m src.testset candidates|check|verify|audit")


if __name__ == "__main__":
    main()
