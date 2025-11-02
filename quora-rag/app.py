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


def _resolve_index_dir() -> Path:
    app_dir = Path(__file__).parent.resolve()
    index_env = os.getenv("FAISS_INDEX_DIR")
    if index_env:
        idx_path = Path(index_env)
        return (app_dir / idx_path).resolve() if not idx_path.is_absolute() else idx_path.resolve()
    return (app_dir / "data/faiss").resolve()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Quora-QA RAG", page_icon="🤖", layout="wide")

    # Sidebar: status + settings
    index_dir = _resolve_index_dir()
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

        st.markdown("---")
        k = st.slider("Top‑K context", min_value=1, max_value=10, value=4)
        st.session_state["top_k"] = k
        st.caption("Higher K may improve grounding but can add noise.")

        st.markdown("---")
        st.markdown("Ingest: `python ingest_index.py <file.csv>` or `--hf-dataset toughdata/quora-question-answer-dataset`.")

    # Header
    st.markdown("### Chat with your Quora‑style knowledge base")

    # Guard on index
    if not index_dir.exists():
        st.stop()

    # Load retriever (cached per K)
    try:
        retriever = _get_retriever(str(index_dir), st.session_state.get("top_k", 4))
    except Exception:
        st.info("Retriever not available yet. Ingest data to proceed.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me anything. I’ll answer based on your indexed Q/A data."}
        ]

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(m["sources"], start=1):
                        st.markdown(f"**Source {i}**")
                        st.write(src.get("content", ""))

    # Chat input
    prompt = st.chat_input("Type your question…")
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    result = answer_question(
                        question=prompt.strip(),
                        retriever=retriever,
                        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        temperature=0.2,
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

            answer_text = result.get("answer", "")
            sources = result.get("sources", [])
            st.markdown(answer_text)
            if sources:
                with st.expander("Sources"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(f"**Source {i}**")
                        st.write(src.get("content", ""))

        # Save assistant message with sources
        st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": sources})


if __name__ == "__main__":
    main()
