import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from qa_utils import load_retriever, answer_question


def main() -> None:
    load_dotenv()  # Load .env if present

    st.set_page_config(page_title="Quora-QA RAG", page_icon="❓", layout="wide")
    st.title("Quora-QA RAG")
    st.caption("Ask questions and retrieve answers grounded in your ingested Quora-style data.")

    index_dir = Path(os.getenv("FAISS_INDEX_DIR", "./data/faiss")).resolve()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    cols = st.columns(3)
    with cols[0]:
        k = st.number_input("Top-K passages", min_value=1, max_value=20, value=4, step=1)
    with cols[1]:
        search_type = st.selectbox("Search type", options=["similarity", "mmr"], index=0)
    with cols[2]:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    query = st.text_input("Your question", placeholder="e.g., How do I improve my public speaking?", label_visibility="visible")
    ask = st.button("Ask", type="primary", use_container_width=True)

    if not index_dir.exists():
        st.warning(
            f"Vector store not found at {index_dir}. Ingest data first via `ingest_index.py`."
        )

    retriever = None
    try:
        retriever = load_retriever(index_dir=str(index_dir), k=k, search_type=search_type)
    except Exception as e:
        st.info("Retriever not available yet. Ingest data to proceed.")
        st.stop()

    if ask and query.strip():
        with st.spinner("Thinking..."):
            try:
                result = answer_question(
                    question=query.strip(),
                    retriever=retriever,
                    model_name=model_name,
                    temperature=temperature,
                )
            except Exception as e:
                st.error(f"Error answering question: {e}")
                return

        st.subheader("Answer")
        st.write(result["answer"])  # type: ignore[index]

        sources = result.get("sources", [])  # type: ignore[assignment]
        if sources:
            st.subheader("Sources")
            for i, src in enumerate(sources, start=1):
                with st.expander(f"Source {i}"):
                    st.write(src.get("content", ""))
                    meta = {k: v for k, v in (src.get("metadata") or {}).items() if v is not None}
                    if meta:
                        st.caption(str(meta))


if __name__ == "__main__":
    main()
