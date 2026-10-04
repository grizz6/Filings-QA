"""Answer a question from the retrieved chunks, with numbered citations.

The model sees only the top chunks, labeled [1]..[k], and must cite them. If they do not
contain the answer it must reply exactly NOT_FOUND, which callers treat as a refusal.

Run:  python -m src.answer "What does Tesla say about supply chain risk?"
"""

from __future__ import annotations

import re
import sys

from src.config import load_config
from src.llm import chat
from src.retrieve import retrieve

NOT_FOUND = "Not found in the filings."

SYSTEM_PROMPT = f"""You answer questions about the "Risk Factors" sections of companies' \
annual reports (Form 10-K).

Rules:
- Use ONLY the numbered excerpts provided. Do not use outside knowledge.
- Cite every claim with the excerpt number in square brackets, like [2] or [1][3].
- Keep the answer short: 2 to 5 sentences.
- If the excerpts do not contain the answer, reply with exactly: {NOT_FOUND}"""


def build_messages(question: str, chunks: list[dict]) -> list[dict[str, str]]:
    excerpts = "\n\n".join(
        f"[{i}] ({c['ticker']}, {c['id']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]


def is_refusal(text: str) -> bool:
    return text.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower()


def extract_citations(text: str, chunks: list[dict]) -> list[dict]:
    """Citation numbers used in the answer, mapped to chunks; out-of-range numbers dropped."""
    cited: list[int] = []
    for n in (int(x) for x in re.findall(r"\[(\d+)\]", text)):
        if 1 <= n <= len(chunks) and n not in cited:
            cited.append(n)
    return [
        {"n": n, "id": chunks[n - 1]["id"], "source_url": chunks[n - 1]["source_url"]}
        for n in cited
    ]


def answer(question: str, cfg: dict | None = None, retrieve_fn=retrieve, chat_fn=chat) -> dict:
    cfg = cfg or load_config()
    chunks = retrieve_fn(question, cfg=cfg)
    if not chunks:
        return {"question": question, "answer": NOT_FOUND, "refused": True, "citations": [],
                "chunks": []}  # fmt: skip
    text = chat_fn(build_messages(question, chunks), cfg["llm"])
    refused = is_refusal(text)
    return {
        "question": question,
        "answer": NOT_FOUND if refused else text,
        "refused": refused,
        "citations": [] if refused else extract_citations(text, chunks),
        "chunks": chunks,
    }


def format_result(result: dict) -> str:
    lines = [f"Q: {result['question']}", f"A: {result['answer']}"]
    for c in result["citations"]:
        lines.append(f"   [{c['n']}] {c['id']}  {c['source_url']}")
    top = ", ".join(f"{c['id']} ({c['score']:.2f})" for c in result["chunks"])
    lines.append(f"   retrieved: {top}")
    return "\n".join(lines)


def main() -> None:
    """python -m src.answer "question"   or   python -m src.answer --file questions.txt"""
    args = sys.argv[1:]
    if args[:1] == ["--file"] and len(args) == 2:
        with open(args[1], encoding="utf-8") as f:
            questions = [q.strip() for q in f if q.strip() and not q.startswith("#")]
    elif args:
        questions = [" ".join(args)]
    else:
        sys.exit('usage: python -m src.answer "question" | --file questions.txt')
    cfg = load_config()
    for q in questions:
        print(format_result(answer(q, cfg)), end="\n\n", flush=True)


if __name__ == "__main__":
    main()
