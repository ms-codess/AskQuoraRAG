# Quora-QA RAG

Minimal RAG app to ingest Q/A data (CSV or JSONL), build a FAISS vector index with OpenAI embeddings, and query via a Streamlit UI.

Note: This README is a placeholder and can be expanded later.

## Quickstart

1. Create a virtual environment and install requirements:
   - `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\Activate.ps1` on Windows PowerShell)
   - `pip install -r requirements.txt`

2. Set up environment variables (copy `.env.example` to `.env`).

3. Ingest data (creates/updates FAISS index under `FAISS_INDEX_DIR`):
   - `python ingest_index.py path/to/data.csv`

4. Run the app:
   - `streamlit run app.py`
