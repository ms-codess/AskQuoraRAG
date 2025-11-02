import os
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from qa_utils import load_retriever, answer_question


@st.cache_data(show_spinner=False)
def _read_index_meta(index_dir: Path):
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_retriever(index_dir: str, k: int):
    return load_retriever(index_dir=index_dir, k=k, search_type="similarity")


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="Quora-QA RAG", page_icon="❓", layout="wide")

    st.markdown("## Quora-QA RAG")
    st.caption("Ask a question. Get an answer grounded in your ingested Q/A data.")

    app_dir = Path(__file__).parent.resolve()
    index_env = os.getenv("FAISS_INDEX_DIR")
    if index_env:
        idx_path = Path(index_env)
        index_dir = (app_dir / idx_path).resolve() if not idx_path.is_absolute() else idx_path.resolve()
    else:
        index_dir = (app_dir / "data/faiss").resolve()

    with st.sidebar:
        st.subheader("Index Status")
        st.write(f"Path: `{index_dir}`")
        meta = _read_index_meta(index_dir)
        if index_dir.exists() and meta:
            provider = meta.get("embedding_provider", "openai")
            dim = meta.get("embedding_dim", "?")
            chunks = len(meta.get("chunks", []))
            st.success(f"Loaded • {chunks} chunks • {provider} • dim {dim}")
        else:
            st.warning("No index found. Run ingestion first.")

        with st.expander("Settings", expanded=False):
            k = st.slider("Results to use (Top‑K)", min_value=1, max_value=10, value=4)

        st.markdown("---")
        st.markdown("Need an index?\n\n`python ingest_index.py <file.csv>`\n\nOr HF: `--hf-dataset toughdata/quora-question-answer-dataset`.")

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Ask a question",
            placeholder="e.g., How can I improve my public speaking?",
        )
    with col_btn:
        ask = st.button("Ask", type="primary")

    st.markdown("Try:")
    sugg_cols = st.columns(3)
    examples = [
        "How to overcome interview anxiety?",
        "Tips to learn Python faster?",
        "Best ways to stay motivated?",
    ]
    for i, ex in enumerate(examples):
        if sugg_cols[i].button(ex, key=f"ex_{i}"):
            query = ex
            st.session_state["_prefill_query"] = ex

    if not index_dir.exists():
        st.stop()

    try:
        retriever = _get_retriever(str(index_dir), st.session_state.get("k", 4) if "k" in st.session_state else 4)
    except Exception:
        st.info("Retriever not available yet. Ingest data to proceed.")
        st.stop()

    if (ask or st.session_state.pop("_prefill_query", None)) and query.strip():
        with st.spinner("Thinking..."):
            try:
                result = answer_question(
                    question=query.strip(),
                    retriever=retriever,
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.2,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return

        st.markdown("### Answer")
        st.success(result.get("answer", ""))

        sources = result.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for i, src in enumerate(sources, start=1):
                with st.expander(f"Source {i}"):
                    st.write(src.get("content", ""))
                    meta = {k: v for k, v in (src.get("metadata") or {}).items() if v is not None}
                    if meta:
                        st.caption(
                            ", ".join([f"{k}: {v}" for k, v in meta.items()])
                        )


if __name__ == "__main__":
    main()

