# Journal

## Day 1: Setup

- Chose a fully cloud, $0 stack: GitHub Actions + Gemini API free tier (LLM), Supabase
  pgvector, Hugging Face Spaces. No local LLM; no credit card anywhere.
- First tried GitHub Models, but CI got a plain-text "OK" instead of a reply: GitHub retired
  GitHub Models on 2026-07-30. Lesson: verify a service is still live before building on it.
  Switched to Gemini; its key lives only in the GEMINI_API_KEY repo secret.
- Wrote `.gitignore` first (`.env`, `data/`, `mlruns/`, `*.db`), plus `.env.example`.
- CI: ruff (lint + format), pytest, gitleaks secret scan on full history, and an LLM
  smoke test ("say hi").
- All settings in `config.yaml` so later experiments change config, not code.

## Day 2: Download filings

- `src/download.py`: ticker -> CIK from `company_tickers.json`, latest original 10-K from
  `data.sec.gov/submissions` (skips 10-K/A amendments), HTML saved to `data/raw/`, plus a
  `manifest.json` with each file's source URL and accession number for citations later.
- SEC rules: `User-Agent` from the `SEC_USER_AGENT` secret, requests spaced >= 0.2 s apart,
  retries with backoff on 429/5xx only.
- Runs in the cloud: `.github/workflows/pipeline.yml` downloads the filings in GitHub Actions
  and keeps them as the `raw-filings` artifact (data never goes into git).
- Gemini `gemini-2.5-flash` is closed to new users; pinned `gemini-3.8-flash` (pinned, not the
  `-latest` alias, so evaluation runs stay comparable).
- First real run: 6/10 downloaded, then XOM failed. Exxon redomiciled to a new Texas holding
  company on 2026-07-01; the "XOM" ticker now maps to the new CIK, which has no 10-K yet.
  Added `cik_overrides` in config.yaml (XOM -> 34088, where the 10-Ks are).
- Gemini answered 503 "high demand" once; the client now retries 429/500/503 with backoff
  (2, 4, 8 s) and fails fast on other errors.

## Day 3: Parse + chunk

- `src/parse.py`: HTML -> text (one line per block element; hidden inline-XBRL header and
  `display:none` blocks dropped), then find Item 1A by heading lines only. The table of
  contents and "see Item 1A" cross-references also match, so every start is paired with the
  next end heading (Item 1B / 1C / 2 or "Unresolved Staff Comments") and the longest span
  wins. Page numbers and "Table of Contents" running headers are removed.
- Sanity check: a section under 1,500 or over 80,000 words fails the pipeline, which prints
  each company's word count and first/last words so a wrong span is obvious in the log.
- `src/chunk.py`: 400-word windows, 50-word overlap, IDs like `TSLA_1A_0012`; each chunk keeps
  company, source URL and filing date for citations.

## Day 4: Embed + store

- `all-MiniLM-L6-v2` (384 dims) with normalized vectors, stored in Supabase Postgres with
  pgvector, HNSW index on cosine distance.
- Rows are keyed by `(index_name, id)`: re-indexing replaces one index in a single
  transaction, PR runs write to `ci`, and later chunking experiments get their own index
  instead of overwriting `main`.
- PyTorch is large, so embedding deps live in `requirements-index.txt` (CPU wheels) and only
  the pipeline installs them; the fast CI job stays light.
- Supabase's direct host is IPv6-only and GitHub runners have no IPv6, so the connection uses
  the session pooler URL.

### Day 3-4 results on the real filings

First pipeline run (2026-10-03): all 10 sections found and inside the sanity range.

| Ticker | Words | Chunks | | Ticker | Words | Chunks |
|---|---|---|---|---|---|---|
| AAPL | 9,778 | 28 | | PFE | 12,683 | 37 |
| MSFT | 11,708 | 34 | | XOM | 5,099 | 15 |
| TSLA | 12,586 | 36 | | NKE | 14,413 | 42 |
| JPM | 15,489 | 45 | | NFLX | 11,192 | 32 |
| WMT | 13,541 | 39 | | DAL | 7,663 | 22 |

Total: 330 chunks. Reading each section's first and last words in the log showed page
furniture left at page breaks ("PART I", "Parts I and II", "2026 FORM 10-K 23",
"Delta Air Lines, Inc. | 2025 Form 10-K") and DAL's heading sharing a line with its first
sentence. Added tests with those exact strings, then filtered them out. Embedding worked;
storing failed only because the `SUPABASE_DB_URL` secret was not set yet.

### Day 4 results

After header/footer cleanup: 328 chunks (was 330). With `SUPABASE_DB_URL` set, the pipeline on
`main` stored all 328 in the `main` index, and the two test searches ran:

| Query | Top 3 (cosine similarity) | Verdict |
|---|---|---|
| "cybersecurity breach" | WMT_1A_0018 (0.673), TSLA_1A_0017 (0.667), WMT_1A_0015 (0.652) | all on-topic |
| "supply chain risk", TSLA only | TSLA_1A_0015 residual values (0.418), TSLA_1A_0000 intro (0.405), TSLA_1A_0032 dealer laws (0.378) | weak |

Likely cause of the weak result: `all-MiniLM-L6-v2` truncates input at 256 word pieces
(about 190 words), so roughly half of each 400-word chunk never reaches the embedding.
Day 7's chunk-size experiment (200 vs 400 words) will measure this with hit rate and MRR
instead of guessing.

## Day 5: Retrieve + answer

- `src/retrieve.py`: detects companies named in the question (ticker, name, first word of the
  name, plus aliases in config.yaml such as "JP Morgan") and searches only their chunks;
  otherwise searches all 10 filings. Whole-word matching, so "pineapple" is not Apple.
- `src/answer.py`: top-5 chunks labeled [1]..[5] go to Gemini with three rules: use only the
  excerpts, cite like [2], and say exactly "Not found in the filings." otherwise. Citation
  numbers are mapped back to chunk IDs and SEC URLs; out-of-range numbers are dropped.
- `qa-demo.yml` runs 10 questions in the cloud (8 answerable, 2 that Risk Factors cannot
  answer, to check refusals). The embedding model is cached so it loads once per run.

### Day 5 results

- First demo runs hit Gemini's free-tier limits: `gemini-3.8-flash` allows 5 requests/minute
  and its daily quota ran out after ~15-20 calls. The client now waits as long as Gemini asks
  ("retry in 42s") and fails fast with a clear message on a per-day quota. Switched to
  `gemini-3.1-flash-lite` (also free, ~15/min and ~1,000/day reported).
- Demo on Flash-Lite: 8/8 answerable questions answered with citations to the right company's
  chunks; 2/2 unanswerable ("Apple's total revenue", "Microsoft's CEO") answered
  "Not found in the filings." The Tesla supply-chain answer is thin because retrieval ranks
  weak chunks (the truncation issue noted on Day 4).

## Day 6: Test set

- Drafting in the cloud: for 4 evenly spaced chunks per company, Gemini wrote a question, a
  short answer and an exact quote; a draft was kept only if the quote appears word for word in
  the chunk. 40/40 drafts were grounded on the first run.
- Review: kept all 40, rewrote questions to read naturally, shortened answers to what the
  quote supports, and trimmed two quotes that contained page furniture ("Item 1A" inside an
  MSFT sentence) or bullet characters (JPM).
- Added 10 questions a Risk Factors section should not answer (exact revenue, CEO/CFO names,
  delivery counts, plan prices...), one per company, to measure refusals.
- Each answerable item keeps its quote, so a retrieved chunk counts as correct when it
  contains the quote. This survives chunk-size experiments that change chunk IDs.
- `testset-verify.yml` rebuilds the chunks and checks every quote against the real text.
- Audit (`testset-audit.yml`): Gemini reads each company's ENTIRE Risk Factors section and
  says whether it answers each question, with a supporting sentence checked against the text.
  First runs crashed on Gemini timeouts and "high demand" 503s; calls are now retried on
  timeouts, and a call that still fails is reported as "not audited" instead of failing.
  q07 was flagged only because the model copied half a sentence and paraphrased the rest, so
  evidence now counts when 10+ consecutive words match the section.
- Audit result: all 50 questions agree with the full sections (40/40 answered with
  evidence found in the text, 10/10 unanswerable confirmed not answered).

## Day 7: Experiments

`src/evaluate.py` rebuilds chunks from the sections for each experiment, embeds them, and
searches in memory (same model and company filter as the app, no Supabase writes). Results
are logged to MLflow (`filings-qa-retrieval`); the database is a workflow artifact.

| run | chunks | top_k | hit rate@k | hit rate@5 | MRR | refusals | chunks |
|---|---|---|---|---|---|---|---|
| A | 200 / 50 | 5 | 0.750 | 0.750 | 0.520 | 1.000 | 759 |
| B | 400 / 50 | 5 | 0.825 | 0.825 | 0.602 | 1.000 | 328 |
| C | 400 / 50 | 8 | 0.900 | 0.825 | 0.612 | 1.000 | 328 |

- Hypothesis from Day 4 was that MiniLM truncating at ~190 words hurts 400-word chunks, so
  200-word chunks should retrieve better. Wrong: A is worst. Smaller chunks split the risk
  factor's heading from its explanation, and there are 2.3x as many chunks competing.
- More results (k=8) recover 3 more questions; MRR barely moves, so the extra hits are at
  ranks 6-8. Cost: ~3,200 more words of context per Gemini call, well within Flash-Lite limits.
- Refusals were 10/10 in every run, so a bigger k did not make the model answer questions
  the filings cannot answer.
- Decision: `top_k: 8` in config.yaml (chunks unchanged, so the Supabase index is still valid).

## Day 8: CI eval gate

- `eval-gate` job in `ci.yml`: runs `python -m src.evaluate --gate` on every PR with the
  settings in config.yaml and fails if hit rate < 0.875 or MRR < 0.59 (`eval/thresholds.yaml`).
  Retrieval only, so no LLM quota is spent in CI and the result is deterministic.
- Thresholds allow one question to slip (0.900 -> 0.875). Raising them is a deliberate
  change in the same PR that improves retrieval.
- Proof: a throwaway PR set `top_k: 1`. The gate failed it with hit rate 0.475 and MRR 0.475
  (thresholds 0.875 / 0.59), and the PR was closed without merging.

## Day 9: App, Docker, deployment

- `src/api.py` (FastAPI): `POST /ask` returns the answer, citations (with ticker and SEC
  URL), the retrieved excerpts trimmed to 300 characters, and latency. `GET /health` says
  whether the secrets are configured, never their values. Errors are mapped to clear
  messages: Gemini quota used up -> 503 "try again tomorrow"; anything unexpected -> 500 with
  no internals (a test checks a password in an exception never reaches the response).
- Guardrails for a public demo on free tiers: questions up to 500 characters and 300
  questions per day in total, counted from the question log (HTTP 429 after that).
- `app/` (Streamlit): Ask page with example questions, the cited answer, links to the 10-K
  and the retrieved excerpts with their similarity scores.
- One Docker image runs both: the API on :8000 inside the container and the UI on :7860,
  the port Hugging Face Spaces serves. The embedding model is baked into the image, so the
  first question does not download it. Runs as user 1000, as Spaces requires.
- `docker.yml` builds the image on PRs, runs it with the real secrets and asks one question
  end to end (logged as source "ci" so it stays out of the dashboard).
- `deploy.yml` + `scripts/deploy_space.py`: on push to main, create the Space if needed, copy
  GEMINI_API_KEY and SUPABASE_DB_URL into the Space's encrypted secrets, upload exactly the
  files the image needs, then wait until the live UI passes a health check. Without the
  HF_TOKEN secret the job skips with a notice instead of failing.
- Checked locally: the UI and API run against a fake backend, screenshots of both pages
  looked right. Fixed a hairline bar chart (time axis gave each bar a thin slot; switched to
  an ordinal day axis with Altair).

## Day 10: Monitoring

- Every question is logged to Supabase (`qa_log`): status (answered / refused / error),
  latency, number of citations, top retrieval score, model. Logging failures are caught so
  monitoring can never break answering.
- `src/monitor.py` summarizes the log: questions per day, refusal rate, latency p50/p95
  (nearest-rank, errors excluded), and the low-confidence rate: share of questions whose
  best chunk scored below 0.40. In the Day 5 demo, answerable questions scored 0.57-0.71 and
  unanswerable ones 0.30-0.34, so 0.40 separates them; a rising share is a drift signal.
- Monitoring page in the app: KPI row, questions per day, median latency per day, recent
  questions. `GET /stats` serves the same numbers.
- `monitor.yml` (Mondays): re-downloads the latest 10-Ks and re-runs the test-set verify and
  the retrieval gate (no LLM). It fails when a company files a new 10-K that changes the text
  (time to refresh the test set and re-index) or when retrieval drifts below the thresholds.
  It also writes the week's usage report to the run summary.

## Pre-deploy test pass

- Coverage before: 68% overall, and 0-39% on the code that only runs in production (UI pages,
  deploy script). Added tests where a bug would only show on the live Space:
  - API: the real `answer()` -> `retrieve()` -> `chat()` chain wired through `/ask` (only the
    vector search and Gemini's HTTP call faked); no-database fallback; an unreadable log does
    not block questions; non-quota Gemini outage -> 503 "try again".
  - `PgLog` SQL against a fake driver: the INSERT has one placeholder per value in the right
    order. A mismatch would be silent in production because the API swallows logging errors.
  - Deploy script with a fake Hugging Face client: Space name, both secrets, staged files
    (no tests or .env), commit message; the wait loop's success, build-error and timeout paths.
  - UI pages run headless with Streamlit's AppTest against canned API responses: answered,
    refused, API error, example buttons, Monitoring with data / empty / API down.
- Coverage after: 81% overall; API, UI, monitoring and deploy at 89-100%. CI now fails below 80%.
