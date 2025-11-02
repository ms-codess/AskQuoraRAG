import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
import faiss  # type: ignore
import requests


def _openai_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _embed_texts(texts: List[str], model: str) -> np.ndarray:
    res = _openai_request("embeddings", {"model": model, "input": texts})
    vectors = [np.array(item["embedding"], dtype=np.float32) for item in res["data"]]
    arr = np.vstack(vectors)
    # Normalize for cosine similarity
    faiss.normalize_L2(arr)
    return arr


@dataclass
class SimpleFaissRetriever:
    index: faiss.Index
    chunks: List[Dict[str, Any]]
    embedding_model: str
    default_k: int = 4

    def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        embs = _embed_texts([query], self.embedding_model)
        D, I = self.index.search(embs, k or self.default_k)
        hits = []
        for idx in I[0]:
            if idx == -1:
                continue
            hits.append(self.chunks[idx])
        return hits


def load_retriever(index_dir: str, k: int = 4, search_type: str = "similarity") -> SimpleFaissRetriever:
    # search_type kept for back-compat; only similarity supported
    index_path = Path(index_dir)
    index = faiss.read_index(str(index_path / "index.faiss"))
    meta = json.loads((index_path / "meta.json").read_text(encoding="utf-8"))
    chunks = meta["chunks"]
    embedding_model = meta.get("embedding_model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return SimpleFaissRetriever(index=index, chunks=chunks, embedding_model=embedding_model, default_k=k)


def _format_docs(docs: List[Dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(d.get("content", "") for d in docs)


def answer_question(
    question: str,
    retriever: SimpleFaissRetriever,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    load_dotenv()
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    docs = retriever.get_relevant_documents(question)
    context = _format_docs(docs)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant answering based strictly on the provided context. If the answer is not in the context, say you don't know.",
        },
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]

    res = _openai_request(
        "chat/completions",
        {"model": model, "messages": messages, "temperature": float(temperature)},
    )
    answer_text = res["choices"][0]["message"]["content"] or ""

    sources: List[Dict[str, Any]] = []
    for d in docs:
        sources.append({"content": d.get("content", ""), "metadata": d.get("metadata", {})})

    return {"answer": answer_text, "sources": sources}
