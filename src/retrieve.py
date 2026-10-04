"""Find the chunks that best match a question.

If the question names a company ("Tesla", "TSLA", "JPMorgan"...), only that company's
filing is searched, so an answer about Tesla cannot cite Walmart's risk factors.

Run:  python -m src.retrieve "What does Tesla say about supply chain risk?"
"""

from __future__ import annotations

import re
import sys

from src.config import load_config


def company_aliases(cfg: dict) -> dict[str, str]:
    """Map lowercase alias -> ticker: ticker, full name, first word of name, extra aliases."""
    aliases: dict[str, str] = {}
    for ticker, name in cfg["companies"].items():
        names = {ticker, name, name.split()[0]}
        names.update(cfg.get("aliases", {}).get(ticker, []))
        for n in names:
            aliases[n.lower()] = ticker
    return aliases


def detect_companies(question: str, cfg: dict) -> list[str]:
    """Tickers of the companies a question mentions, in order of first mention."""
    found: list[tuple[int, str]] = []
    for alias, ticker in company_aliases(cfg).items():
        m = re.search(rf"(?<![\w-]){re.escape(alias)}(?:'s)?(?![\w-])", question, re.IGNORECASE)
        if m:
            found.append((m.start(), ticker))
    tickers: list[str] = []
    for _, ticker in sorted(found):
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


def retrieve(
    question: str,
    k: int | None = None,
    index_name: str = "main",
    cfg: dict | None = None,
    search_fn=None,
) -> list[dict]:
    """Top-k chunks for the question as dicts: id, ticker, text, source_url, score."""
    cfg = cfg or load_config()
    k = k or cfg["retrieval"]["top_k"]
    tickers = detect_companies(question, cfg) or None
    if search_fn is None:
        from src.index import query  # needs sentence-transformers + psycopg

        search_fn = query
    rows = search_fn(question, index_name, tickers, k)
    return [
        {"id": r[0], "ticker": r[1], "text": r[2], "source_url": r[3], "score": float(r[4])}
        for r in rows
    ]


def main() -> None:
    question = " ".join(sys.argv[1:])
    if not question:
        sys.exit('usage: python -m src.retrieve "question"')
    for r in retrieve(question):
        print(f"{r['score']:.3f}  {r['id']}  {r['text'][:140]}...")


if __name__ == "__main__":
    main()
