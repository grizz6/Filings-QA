import json

import pytest

from src import index

CHUNK = {
    "id": "TSLA_1A_0000",
    "ticker": "TSLA",
    "company": "Tesla",
    "chunk_index": 0,
    "text": "Supply chain risk.",
    "n_words": 3,
    "source_url": "https://www.sec.gov/x.htm",
    "filing_date": "2026-01-29",
}


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split()), params))

    def executemany(self, sql, rows):
        self.log.append(("executemany", " ".join(sql.split()), rows))


class FakeConn:
    def __init__(self):
        self.log = []
        self.committed = False

    def cursor(self):
        return FakeCursor(self.log)

    def commit(self):
        self.committed = True


def test_schema_uses_embedding_dimension():
    sql = " ".join(index.schema_sql(384))
    assert "vector(384)" in sql
    assert "PRIMARY KEY (index_name, id)" in sql
    assert "vector_cosine_ops" in sql


def test_rows_for_insert_order_matches_insert_sql():
    rows = index.rows_for_insert([CHUNK], [[0.1, 0.2]], "main")
    assert rows == [
        (
            "main",
            "TSLA_1A_0000",
            "TSLA",
            "Tesla",
            0,
            "Supply chain risk.",
            "https://www.sec.gov/x.htm",
            "2026-01-29",
            [0.1, 0.2],
        )
    ]
    assert index.INSERT_SQL.count("%s") == len(rows[0])


def test_rows_for_insert_length_mismatch():
    with pytest.raises(index.IndexBuildError):
        index.rows_for_insert([CHUNK, CHUNK], [[0.1]], "main")


def test_write_index_replaces_only_that_index_in_one_transaction():
    conn = FakeConn()
    rows = index.rows_for_insert([CHUNK], [[0.1]], "ci")
    index.write_index(conn, rows, "ci", dim=384)
    deletes = [e for e in conn.log if e[1].startswith("DELETE")]
    assert deletes == [("execute", "DELETE FROM chunks WHERE index_name = %s", ("ci",))]
    assert conn.log[-1] == ("executemany", " ".join(index.INSERT_SQL.split()), rows)
    assert conn.committed


def test_db_url_required_and_ssl_added(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    with pytest.raises(index.IndexBuildError, match="SUPABASE_DB_URL"):
        index.db_url()
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@host:5432/postgres")
    assert index.db_url().endswith("?sslmode=require")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://u:p@host/db?sslmode=disable")
    assert index.db_url().endswith("sslmode=disable")


def test_load_chunks(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(CHUNK) + "\n\n")
    assert index.load_chunks(path) == [CHUNK]


def test_search_filters_by_ticker_list():
    calls = []

    class Cur(FakeCursor):
        def fetchall(self):
            return [("TSLA_1A_0000", "TSLA", "t", "u", 0.9)]

    class Conn(FakeConn):
        def cursor(self):
            return Cur(calls)

    rows = index.search(Conn(), [0.1], "main", ["TSLA", "AAPL"], 5)
    _, sql, params = calls[0]
    assert "ticker = ANY(%s)" in sql
    assert params == ([0.1], "main", ["TSLA", "AAPL"], ["TSLA", "AAPL"], [0.1], 5)
    assert rows[0][0] == "TSLA_1A_0000"


def test_chunks_table_has_row_level_security():
    assert "ALTER TABLE chunks ENABLE ROW LEVEL SECURITY" in index.schema_sql(384)


def test_embedder_loads_once_even_when_requested_concurrently(monkeypatch):
    import sys
    import threading
    import time
    import types

    built = []

    class SlowModel:
        def __init__(self, name):
            time.sleep(0.2)  # loading a real model takes seconds
            built.append(name)

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=SlowModel))  # fmt: skip
    index.load_embedder.cache_clear()
    threads = [threading.Thread(target=index.load_embedder, args=("m",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built == ["m"]
    index.load_embedder.cache_clear()
