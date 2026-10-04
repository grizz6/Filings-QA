# Filings Q&A

Ask questions about the **Risk Factors** section of 10 companies' annual reports (10-K
filings) and get answers with citations, e.g. *"What does Tesla say about supply chain risk?"*

A retrieval-augmented generation (RAG) system built as an MLOps project: tracked experiments,
an evaluation test set, CI that blocks changes when retrieval quality drops, a containerized
app, and monitoring. Runs entirely on free cloud services, with no credit card and no local LLM.

> **Status:** Day 4 of 10 done (data indexed in Supabase). Next: retrieval + answers. See the roadmap below.

## Cloud stack (all free tiers)

| Piece | Service |
|---|---|
| Code, CI, eval gate, container registry | GitHub, GitHub Actions, GHCR |
| LLM | Google Gemini API, free tier (`gemini-3.8-flash`) |
| Vector store + question log | Supabase Postgres with pgvector |
| App hosting (API, UI, dashboard) | Hugging Face Spaces |
| Data source | SEC EDGAR (public, no key) |

## Secrets

No secret is ever committed. `.env` is git-ignored and CI scans every push with gitleaks.
Secrets live in **repo Settings → Secrets and variables → Actions**:

| Secret | Used for | Needed from |
|---|---|---|
| `GEMINI_API_KEY` | LLM calls (Google AI Studio key) | Day 1 |
| `SEC_USER_AGENT` | SEC requires `Name email` on every request | Day 2 |
| `SUPABASE_DB_URL` | Postgres connection string | Day 4 |
| `HF_TOKEN` | Deploying to Hugging Face Spaces | Day 9 |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # fill in values; never commit .env
ruff check . && pytest -q
python -m src.llm "Say hi"   # needs GEMINI_API_KEY in .env
python -m src.download       # needs SEC_USER_AGENT in .env; writes data/raw/
python -m src.parse          # data/raw/ -> data/sections/<TICKER>.txt (Item 1A only)
python -m src.chunk          # -> data/chunks.jsonl (400 words, 50 overlap, IDs like TSLA_1A_0012)
pip install -r requirements-index.txt   # sentence-transformers (CPU PyTorch), psycopg, pgvector
python -m src.index          # embed + store in Supabase; needs SUPABASE_DB_URL in .env
python -m src.index --query "supply chain risk" --ticker TSLA
python -m src.answer "What does Tesla say about supply chain risk?"   # needs GEMINI_API_KEY too
```

In the cloud, `.github/workflows/pipeline.yml` runs all of these in GitHub Actions. Pull
requests write to a separate `ci` index in Supabase so they never touch `main`.
`.github/workflows/qa-demo.yml` answers `eval/demo_questions.txt` (or a question you type in
under Actions → Q&A demo → Run workflow) and prints the answers with citations.

## Roadmap

| Day | What | Status |
|---|---|---|
| 1 | Setup, CI (ruff, pytest, gitleaks), LLM smoke test | done |
| 2 | Download 10 filings from SEC EDGAR (`src/download.py`, pipeline workflow) | done |
| 3 | Parse "Item 1A. Risk Factors" + chunk + tests (`src/parse.py`, `src/chunk.py`): 10/10 sections, 328 chunks | done |
| 4 | Embed + store in Supabase pgvector (`src/index.py`): 328 chunks in the `main` index | done |
| 5 | Retrieve + answer with citations (`src/retrieve.py`, `src/answer.py`) (v1.0) | in progress |
| 6 | 50-question test set | |
| 7 | Evaluation + MLflow experiments | |
| 8 | CI eval gate (blocks quality drops) | |
| 9 | FastAPI + Streamlit + Docker, deployed to Spaces | |
| 10 | Monitoring dashboard + README + demo | |
