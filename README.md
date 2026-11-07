# Alphie - AI research analyst (local, Streamlit)

Alphie turns your own broker / Bloomberg research PDFs and Excel models into a searchable repository and
writes analysis on top of it (in Dutch), optionally blended with recent web news.

- **Case Builder** - five-section investment case (summary, track record, value-creation pillars, risks, conclusion)
- **Custom Query** - free-form questions with control over model, chunks, tokens, publication date and company filter
- **Uploaden** - bulk upload PDFs, ZIPs and Excel models; companies are auto-detected per document
- **Repository** - browse, filter and delete documents; rebuild the vector index
- **Events / Watchlist** - per-ticker summary of what happened in the last N days and the impact on the moat

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # add your OPENAI_API_KEY (and optionally TAVILY_API_KEY)
streamlit run app.py
```

Docker: `HOST_DATA_DIR=/path/to/data docker compose -f docker_compose.yml up -d --build` (app on port 8588).

## How it works

| Module | Role |
|---|---|
| `llm.py` | Model config, lazy OpenAI client, chat helper that handles both gpt-4o-style and reasoning (gpt-5, o-series) models |
| `rag.py` | PDF/Excel ingestion, chunking, embeddings, FAISS/Qdrant storage, retrieval |
| `scoring.py` | Date parsing (EN + NL relative periods), recency decay, company-name matching |
| `company_normalizer.py` | Company extraction from text and canonicalisation against companies already in the repository |
| `agent_policy.py` | Investment case builder and event summaries |
| `custom_query.py` | Custom query mode |
| `websearch.py` | Tavily web/news search (optional) |
| `db.py` | SQLite schema and migrations |

Retrieval details:
- Company and publication-date filters are applied **before** ranking (exact search over that company's chunks),
  so a company with few documents is still found in a large repository.
- Scores decay with age (`RECENCY_TAU_DAYS`, default 120 days; the investment case uses 365 because history matters).
- A requested company that is not in the repository is **not** silently mapped onto a different one
  (`COMPANY_MATCH_THRESHOLD`); the case builder then flags that it has no internal sources.
- Duplicates are skipped by content hash. Documents and indices live under `data/` (or `DATA_DIR`).

All settings are documented in `.env.example`.

## Development

```bash
pip install -r requirements-dev.txt
ruff format . && ruff check .
pytest            # offline: OpenAI and Tavily are stubbed
```

## Roadmap ideas
- Company registry (ISIN, exchange) + fuzzy matching
- Use the parsed Excel metrics in the case builder (numbers are ingested but not yet used in prompts)
- Scheduled watchlist summaries via cron + email/Slack
- Multi-user auth and S3 storage
