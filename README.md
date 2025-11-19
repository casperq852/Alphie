# Bloomberg Research RAG App (Local, Streamlit)

An MVP to:
1) Bulk upload Bloomberg (or other) PDF research (ZIP or individual PDFs)
2) Auto-map PDFs to companies (smart classification with OpenAI)
3) Build embeddings and chat with an agent grounded in your PDFs (+ optional web context)
4) Generate a daily summary for your watchlist combining recent PDFs and public news

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env && edit .env                   # paste your OPENAI_API_KEY
streamlit run app.py
```

## Notes
- Web augmentation uses Tavily if `TAVILY_API_KEY` is present. Otherwise the app falls back to PDF-only answers.
- Documents and indices are stored under `data/`. Safe to delete when you want a fresh start.
- The company classifier is **transparent**: it shows the parsed candidates and the model's final choice(s) per PDF.
- Duplicates are skipped using a content hash.
- For company extraction, we try to detect tickers (e.g., **AAPL US**, **ASML NA**) and names in the PDF text, then let the model confirm/clean it.

## Watchlist
Edit your watchlist in the sidebar (comma-separated tickers), then click **Save**.
The daily summary button will use these tickers.

## Roadmap ideas (easy next steps)
- Add company registry (ISIN, exchange) + fuzzy matching
- Multi-tenant auth and S3 storage
- Per-company vector stores for faster retrieval
- Scheduled daily summary via cron/docker + email/Slack
