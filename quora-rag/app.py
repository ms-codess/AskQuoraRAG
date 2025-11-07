import os
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from qa_utils import build_or_load_index, search

# Load .env so OPENAI_API_KEY and config are available when running from repo root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Quora-themed page config
st.set_page_config(page_title="Quora-RAG", page_icon="💬", layout="wide")

# Brand palette
Q_RED = "#B92B27"
Q_RED_HOVER = "#8B1A1A"
Q_BG = "#F7F3EE"  # beige background
Q_TEXT = "#262626"
Q_SUB = "#6B6B6B"
Q_DIV = "#E6E6E6"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HAS_OPENAI = importlib.util.find_spec("openai") is not None


@st.cache_resource
def get_index_bundle():
    return build_or_load_index()  # (model, texts, index)


model, texts, index = get_index_bundle()

# Header with Comfortaa font
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@600;700&display=swap');
      html, body {{ background:{Q_BG}; color:{Q_TEXT}; }}
      section.main > div.block-container {{ background: transparent; }}
      /* Red buttons */
      div.stButton > button:first-child {{
        background-color: {Q_RED}; color: #fff; border-radius: 6px;
        height: 2.5em; padding: 0 16px; font-weight: 700;
      }}
      div.stButton > button:hover {{ background-color:{Q_RED_HOVER}; color:#fff; }}
    </style>
    <div style='text-align:center; padding: 10px 0;'>
        <h1 style='color:{Q_RED}; font-family: "Comfortaa", Helvetica, Arial, sans-serif;'>💬 Quora-RAG Assistant</h1>
        <p style='color:{Q_SUB}; font-size:16px;'>Ask questions and get grounded answers from real Quora-style Q&A</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Top-K passages", 1, 10, 8,
        help="How many top passages to retrieve. Higher K improves recall but may add noise.",
    )
    use_llm = st.toggle(
        "Use LLM (requires OPENAI_API_KEY)",
        value=bool(OPENAI_API_KEY) and HAS_OPENAI,
        disabled=not HAS_OPENAI,
        help=(
            "Enable LLM answers when an API key is set." if HAS_OPENAI
            else "Install 'openai' to enable LLM answers (pip install openai)."
        ),
    )
    if not HAS_OPENAI:
        st.info("LLM disabled: 'openai' package not installed. Retrieval will still work.")
    else:
        if use_llm and OPENAI_API_KEY:
            st.caption(f"LLM: ON • model={MODEL_NAME}")
        elif not OPENAI_API_KEY:
            st.caption("LLM: OFF • missing OPENAI_API_KEY")
        else:
            st.caption("LLM: OFF")
    if st.button("New chat", use_container_width=True):
        st.session_state.pop("messages", None)

    # Simple Evaluation Dashboard
    with st.expander("Evaluation Dashboard"):
        st.caption("Recall@K, MRR and similarity on current index")
        max_eval = min(500, len(texts)) if len(texts) > 0 else 20
        eval_size = st.slider("Sample size", 20, max_eval, min(50, max_eval), key="eval_size")
        k_max = st.slider("Max K", 1, 20, 5, key="eval_kmax")
        if st.button("Run Evaluation", use_container_width=True, key="run_eval"):
            try:
                qs: List[str] = []
                gt_idx: List[int] = []
                for i, t in enumerate(texts):
                    q = t.split("\n", 1)[0]
                    if q.lower().startswith("q:"):
                        q = q[2:].strip()
                    qs.append(q)
                    gt_idx.append(i)
                n = min(eval_size, len(qs))
                qs_eval = qs[:n]
                gt_eval = np.array(gt_idx[:n])
                q_embs = model.encode(qs_eval, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                D, I = index.search(q_embs, k_max)
                recalls: Dict[int, float] = {}
                for k in range(1, k_max + 1):
                    recalls[k] = float(np.mean([int(gt_eval[i] in I[i, :k]) for i in range(n)]))
                rr = []
                for i in range(n):
                    ranks = np.where(I[i, :k_max] == gt_eval[i])[0]
                    rr.append(1.0 / (int(ranks[0]) + 1) if ranks.size > 0 else 0.0)
                mrr = float(np.mean(rr))
                top1 = D[:, 0].tolist()
                c1, c2, c3 = st.columns(3)
                c1.metric("Recall@1", f"{recalls.get(1,0):.2f}")
                c2.metric(f"Recall@{k_max}", f"{recalls.get(k_max,0):.2f}")
                c3.metric(f"MRR@{k_max}", f"{mrr:.2f}")
                st.subheader("Recall vs K")
                st.bar_chart([recalls[k] for k in sorted(recalls.keys())])
                st.subheader("Top-1 Cosine Similarity")
                st.bar_chart(top1)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {role: 'user'|'assistant', content: str, ctx: Optional[List[Tuple[str,float]]]}


def llm_answer(question: str, ctx_lines: List[str]) -> str:
    if not (use_llm and OPENAI_API_KEY and HAS_OPENAI):
        return "Enable LLM in Settings to generate a concise answer with citations."
    try:
        from openai import OpenAI
    except Exception:
        return "The 'openai' package is not installed. Install it to enable LLM answers."

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""You are a helpful assistant for Q&A.
Use ONLY the provided context to answer. When the context is sufficient, cite sources using bracket numbers like [1], [2].

If the context is insufficient or irrelevant:
- Provide a helpful, structured answer with 4–7 concise bullet points.
- Include concrete steps, examples, or resources (no URLs required).
- Do NOT add any citations.
- At the very end, append this exact line on its own: Based on general knowledge.
- Never reply with only that line; always include the bullet-point guidance above.

Question: {question}

Context:
{chr(10).join(ctx_lines)}
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI request failed: {e}"


def process_query(question: str) -> None:
    if not question:
        return
    # Show the user message immediately
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Retrieving context…"):
        hits = search(model, index, texts, question, k=top_k)
    if use_llm and OPENAI_API_KEY and HAS_OPENAI:
        with st.spinner("Generating answer…"):
            ctx_lines = [f"[{i}] {t}" for i, (t, _s) in enumerate(hits, start=1)]
            answer = llm_answer(question, ctx_lines)
    else:
        answer = "LLM is off. Showing retrieved context only (open the expander)."
    # Show assistant answer immediately
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "ctx": hits})

# Suggestions by specialization (top of page)
show_suggestions = len(st.session_state.messages) == 0
categories = {
    "Getting Started": [
        "How do I break into AI?",
        "Best roadmap to start a career in AI",
        "Do I need a CS degree for ML?",
        "How to switch from software engineering to ML",
        "What entry-level AI roles should I target?",
        "How to learn Python for AI quickly",
    ],
    "Math & Foundations": [
        "What math is needed for machine learning?",
        "How to learn linear algebra for ML",
        "Essentials of probability for ML",
        "Gradient descent explained simply",
        "Bias–variance tradeoff examples",
    ],
    "Machine Learning": [
        "How to choose an evaluation metric",
        "Precision vs recall vs F1",
        "Cross‑validation best practices",
        "Feature engineering ideas for tabular data",
        "How to handle class imbalance",
        "How to prevent overfitting",
    ],
    "Deep Learning": [
        "When to use CNN vs RNN vs Transformer",
        "Transfer learning: when and how",
        "Hyperparameter tuning tips",
        "Batch norm vs layer norm",
        "Vanishing/exploding gradients: fixes",
    ],
    "NLP": [
        "How do word embeddings work?",
        "Tokenization strategies (BPE, WordPiece)",
        "Fine‑tuning vs prompt‑tuning",
        "Evaluation metrics for NLP",
        "Named entity recognition basics",
    ],
    "Computer Vision": [
        "Image augmentation techniques",
        "Object detection vs segmentation",
        "Transfer learning for vision models",
        "Vision transformers: when to use",
        "Evaluation metrics for detection",
    ],
    "LLMs & RAG": [
        "What is Retrieval‑Augmented Generation (RAG)?",
        "How do embeddings work in semantic search?",
        "Choosing a vector index: FAISS vs DBs",
        "How to reduce hallucinations in RAG",
        "How to evaluate RAG (Recall@K, MRR)",
        "Prompting strategies for grounded answers",
    ],
    "MLOps & Deployment": [
        "How to deploy a small ML model for free",
        "Batch vs real‑time inference tradeoffs",
        "Model monitoring: drift and data quality",
        "Feature stores: do I need one?",
        "CI/CD for ML projects basics",
    ],
    "Data Engineering": [
        "Designing data pipelines for ML",
        "Efficient dataset versioning",
        "Parquet vs CSV vs JSON for ML",
        "Handling missing data at scale",
        "Optimizing data joins and merges",
    ],
    "Reinforcement Learning": [
        "What is Q‑learning?",
        "Policy gradients vs value methods",
        "Reward shaping pitfalls",
        "When to use RL in products",
        "Sim2Real challenges",
    ],
    "Time Series & Recommenders": [
        "Prophet vs ARIMA vs LSTM",
        "Feature engineering for time series",
        "Cold‑start problem in recommenders",
        "Implicit vs explicit feedback",
        "Evaluation for recommenders",
    ],
    "Generative AI": [
        "Text‑to‑image models overview",
        "Safety/guardrails for gen‑AI",
        "Prompt engineering tips",
        "Few‑shot vs zero‑shot prompting",
        "Content filtering strategies",
    ],
    "Career & Interviews": [
        "How to prepare for an ML interview",
        "Common ML system design questions",
        "Data scientist vs ML engineer",
        "How to showcase portfolio projects",
        "Negotiating an ML job offer",
    ],
    "Ethics & Safety": [
        "Bias and fairness in ML",
        "Responsible AI principles",
        "Privacy‑preserving ML (DP, FL)",
        "Evaluating harmful outputs in LLMs",
        "Secure prompt handling",
    ],
}

# Render category blocks with buttons in a grid
if show_suggestions:
    st.subheader("Suggestions by Specialization")
    for cat, items in categories.items():
        with st.expander(cat, expanded=False):
            cols = st.columns(3)
            for i, q in enumerate(items):
                if cols[i % 3].button(q, use_container_width=True, key=f"sugg_{cat}_{i}"):
                    process_query(q)
else:
    # Offer a quick way to clear chat and return to suggestions
    if st.button("Back to suggestions", help="Clear chat and show topic suggestions again"):
        st.session_state.messages = []
        try:
            st.rerun()
        except Exception:
            st.stop()

# Chat area at the bottom
st.markdown("---")
st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("ctx") and m["role"] == "assistant":
            with st.expander("Retrieved context"):
                for r, (txt, sc) in enumerate(m["ctx"], start=1):
                    st.markdown(f"**[{r}] score:** {sc:.3f}")
                    st.code(txt)

user_msg = st.chat_input("Type your question…")
for_submit = user_msg if user_msg else None
if for_submit:
    process_query(for_submit)
