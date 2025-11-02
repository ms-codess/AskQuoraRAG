import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import faiss  # type: ignore
from dotenv import load_dotenv
import requests


def read_csv(path: Path) -> Iterable[Dict]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def read_jsonl(path: Path) -> Iterable[Dict]:
    import json

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_docs(records: Iterable[Dict]) -> List[Dict]:
    docs: List[Dict] = []
    for r in records:
        q = (r.get("question") or r.get("Question") or r.get("q") or "").strip()
        a = (r.get("answer") or r.get("Answer") or r.get("a") or "").strip()
        _id = (r.get("id") or r.get("_id") or r.get("uuid") or None)

        if not q and not a:
            continue

        content = f"Question: {q}\nAnswer: {a}".strip()
        meta = {"question": q or None, "id": _id}
        docs.append({"content": content, "metadata": meta})
    return docs


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - chunk_overlap, 0)
    return chunks


def chunk_docs(docs: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for d in docs:
        parts = chunk_text(d["content"])
        for p in parts:
            out.append({"content": p, "metadata": d.get("metadata", {})})
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _openai_request(endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


def _embed_texts(texts: List[str], model: str) -> np.ndarray:
    res = _openai_request("embeddings", {"model": model, "input": texts})
    data = res["data"]  # type: ignore[index]
    vecs = [np.array(item["embedding"], dtype=np.float32) for item in data]  # type: ignore[index]
    arr = np.vstack(vecs)
    faiss.normalize_L2(arr)
    return arr


def ingest(input_path: Path, index_dir: Path, embedding_model: str) -> None:
    ext = input_path.suffix.lower()
    if ext == ".csv":
        records = read_csv(input_path)
    elif ext in {".jsonl", ".ndjson"}:
        records = read_jsonl(input_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .jsonl")

    base_docs = build_docs(records)
    if not base_docs:
        print("No records to ingest.")
        return

    docs = chunk_docs(base_docs)
    print(f"Built {len(base_docs)} base docs -> {len(docs)} chunks")

    ensure_dir(index_dir)

    metas_path = index_dir / "meta.json"
    index_path = index_dir / "index.faiss"

    texts = [d["content"] for d in docs]
    embs = _embed_texts(texts, embedding_model)

    if index_path.exists() and metas_path.exists():
        index = faiss.read_index(str(index_path))
        index.add(embs)
        meta = json.loads(metas_path.read_text(encoding="utf-8"))
        meta["chunks"].extend(docs)
    else:
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        meta = {"embedding_model": embedding_model, "chunks": docs}

    faiss.write_index(index, str(index_path))
    metas_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"Saved FAISS index at {index_dir} with {len(meta['chunks'])} total chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Q/A data into a FAISS vector store")
    parser.add_argument("input", type=Path, help="Path to .csv or .jsonl with question/answer fields")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(os.getenv("FAISS_INDEX_DIR", "./data/faiss")),
        help="Directory to persist FAISS index",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    ingest(args.input.resolve(), args.index_dir.resolve(), args.embedding_model)


if __name__ == "__main__":
    main()
