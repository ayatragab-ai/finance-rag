# Financial QA RAG Assistant

A Retrieval-Augmented Generation (RAG) app over company **10-K filings**
(FinanceQA-style financial Q&A). You type a financial question; the app
retrieves the relevant filing passages and answers, grounded, with cited
sources.

The financial data is stored **locally** in `financeqa_data.json`, so the app
does **not** need to download any dataset at runtime.

## Pipeline

```text
financeqa_data.json         # the financial corpus (local, no download)
01_documents.py             # load + unify the local corpus
02_preprocessing.py         # finance-safe text normalization (keeps figures)
03_chunking.py              # overlapping word-window chunker
04_vector_representation.py # BM25 + embeddings + hybrid search
05_create_chroma_store.py   # OPTIONAL local Chroma export (run once)
06_retrieve_context.py      # context building (filter / dedupe / budget)
07_prompting.py             # grounded prompt + OpenRouter call
streamlit_app.py            # UI: type a question -> answer + sources
```

Retrieval: `hybrid = 0.5 * BM25 + 0.5 * all-MiniLM-L6-v2 embeddings`
(falls back to BM25 only if the embedding model is unavailable).

## Run locally

```bash
python -m pip install -r requirements.txt
cp .env.example .env          # paste your OpenRouter key into .env
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. On https://share.streamlit.io create an app from the repo, main file
   `streamlit_app.py`, Python 3.11 or 3.12.
3. In **Advanced settings -> Secrets** add:

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-your-key"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. Deploy.

## Notes on "offline"

- The **data** is local — no internet needed to load it.
- The **embedding model** (`all-MiniLM-L6-v2`) downloads once on first run; if it
  cannot, the app automatically falls back to BM25-only retrieval.
- The **answer** comes from OpenRouter, which always needs internet (it is a
  cloud LLM). To add more financial data, edit `financeqa_data.json`.
