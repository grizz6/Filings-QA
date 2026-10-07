"""Embed chunks and store them in Supabase Postgres (pgvector).

Rows are grouped by `index_name` so experiments (e.g. different chunk sizes) and CI
runs never overwrite the main index: re-indexing replaces only that index's rows.

Run:
  python -m src.index                               # index data/chunks.jsonl as "main"
  python -m src.index --index-name ci               # separate index (used on pull requests)
  python -m src.index --query "supply chain" --ticker TSLA   # quick search check

Needs SUPABASE_DB_URL (Supabase -> Connect -> Session pooler URI; GitHub runners have no
IPv6, so the direct db.<ref>.supabase.co host does not work there).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

from src.config import ROOT, load_config

CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
BATCH_SIZE = 64


class IndexBuildError(RuntimeError):
    pass


def schema_sql(dim: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""CREATE TABLE IF NOT EXISTS chunks (
            index_name  text    NOT NULL,
            id          text    NOT NULL,
            ticker      text    NOT NULL,
            company     text    NOT NULL,
            chunk_index integer NOT NULL,
            text        text    NOT NULL,
            source_url  text    NOT NULL,
            filing_date date    NOT NULL,
            embedding   vector({dim}) NOT NULL,
            PRIMARY KEY (index_name, id)
        )""",
        "CREATE INDEX IF NOT EXISTS chunks_index_ticker ON chunks (index_name, ticker)",
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)",
        # Close Supabase's Data API path to this table; the owner role used here is unaffected.
        "ALTER TABLE chunks ENABLE ROW LEVEL SECURITY",
    ]


INSERT_SQL = """INSERT INTO chunks
    (index_name, id, ticker, company, chunk_index, text, source_url, filing_date, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""

SEARCH_SQL = """SELECT id, ticker, text, source_url, 1 - (embedding <=> %s) AS score
    FROM chunks
    WHERE index_name = %s AND (%s::text[] IS NULL OR ticker = ANY(%s))
    ORDER BY embedding <=> %s
    LIMIT %s"""


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_for_insert(chunks: list[dict], vectors, index_name: str) -> list[tuple]:
    if len(chunks) != len(vectors):
        raise IndexBuildError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    return [
        (
            index_name,
            c["id"],
            c["ticker"],
            c["company"],
            c["chunk_index"],
            c["text"],
            c["source_url"],
            c["filing_date"],
            v,
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]


_EMBEDDER_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _build_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    return SentenceTransformer(model_name)


def load_embedder(model_name: str):
    """Load the model once per process, even if a warm-up and a request ask at once."""
    with _EMBEDDER_LOCK:
        return _build_embedder(model_name)


load_embedder.cache_clear = _build_embedder.cache_clear


def embed(model, texts: list[str]):
    """Unit-length vectors, so cosine similarity is a plain dot product."""
    return model.encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )


def db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL", "").strip()
    if not url:
        raise IndexBuildError("SUPABASE_DB_URL is not set. See .env.example.")
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def connect(url: str):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(url, autocommit=False)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # needed before register_vector
    conn.commit()
    register_vector(conn)
    return conn


def write_index(conn, rows: list[tuple], index_name: str, dim: int) -> None:
    """Replace all rows of `index_name` in one transaction."""
    with conn.cursor() as cur:
        for stmt in schema_sql(dim):
            cur.execute(stmt)
        cur.execute("DELETE FROM chunks WHERE index_name = %s", (index_name,))
        cur.executemany(INSERT_SQL, rows)
    conn.commit()


def search(conn, query_vector, index_name: str, tickers: list[str] | None, k: int) -> list[tuple]:
    """Top-k chunks by cosine similarity, optionally only from the given companies."""
    with conn.cursor() as cur:
        cur.execute(SEARCH_SQL, (query_vector, index_name, tickers, tickers, query_vector, k))
        return cur.fetchall()


def build(index_name: str) -> int:
    cfg = load_config()["retrieval"]
    chunks = load_chunks()
    model = load_embedder(cfg["embedding_model"])
    vectors = embed(model, [c["text"] for c in chunks])
    # Renamed in newer sentence-transformers; support both.
    get_dim = getattr(model, "get_embedding_dimension", None)
    dim = get_dim() if get_dim else model.get_sentence_embedding_dimension()
    conn = connect(db_url())
    try:
        write_index(conn, rows_for_insert(chunks, vectors, index_name), index_name, dim)
        (n,) = conn.execute(
            "SELECT count(*) FROM chunks WHERE index_name = %s", (index_name,)
        ).fetchone()
    finally:
        conn.close()
    if n != len(chunks):
        raise IndexBuildError(f"wrote {len(chunks)} chunks but the table has {n}")
    return n


def query(text: str, index_name: str, tickers: list[str] | None, k: int) -> list[tuple]:
    cfg = load_config()["retrieval"]
    model = load_embedder(cfg["embedding_model"])
    conn = connect(db_url())
    try:
        return search(conn, embed(model, [text])[0], index_name, tickers, k)
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--index-name", default="main")
    p.add_argument("--query")
    p.add_argument("--ticker")
    p.add_argument("-k", type=int, default=3)
    args = p.parse_args()
    try:
        if args.query:
            for cid, _ticker, text, _url, score in query(
                args.query, args.index_name, [args.ticker] if args.ticker else None, args.k
            ):
                print(f"{score:.3f}  {cid}  {text[:160]}...")
        else:
            n = build(args.index_name)
            print(f"Indexed {n} chunks into Supabase (index_name={args.index_name!r})")
    except IndexBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
