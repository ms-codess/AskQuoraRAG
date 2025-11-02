import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import faiss  # type: ignore
from dotenv import load_dotenv
import requests
import time
import hashlib


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


def normalize_record(r: Dict) -> Optional[Dict]:
    q = (r.get("question") or r.get("Question") or r.get("q") or r.get("questions") or "").strip()
    a = (r.get("answer") or r.get("Answer") or r.get("a") or r.get("answers") or "").strip()
    _id = (r.get("id") or r.get("_id") or r.get("uuid") or None)
    if not q and not a:
        return None
    content = f"Question: {q}\nAnswer: {a}".strip()
    meta = {"question": q or None, "id": _id}
    return {"content": content, "metadata": meta}


def build_docs(records: Iterable[Dict]) -> List[Dict]:
    docs: List[Dict] = []
    for r in records:
        d = normalize_record(r)
        if d:
            docs.append(d)
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
    # basic retry with exponential backoff for 429/5xx
    delay = 1.0
    for attempt in range(7):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code < 400:
            return resp.json()  # type: ignore[return-value]
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()  # unreachable


def _embed_texts_openai(texts: List[str], model: str) -> np.ndarray:
    res = _openai_request("embeddings", {"model": model, "input": texts})
    data = res["data"]  # type: ignore[index]
    vecs = [np.array(item["embedding"], dtype=np.float32) for item in data]  # type: ignore[index]
    arr = np.vstack(vecs)
    faiss.normalize_L2(arr)
    return arr


def _embed_texts_hash(texts: List[str], dim: int = 1024) -> np.ndarray:
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, txt in enumerate(texts):
        for tok in txt.lower().split():
            h = int.from_bytes(hashlib.sha1(tok.encode("utf-8")).digest()[:8], "little")
            j = h % dim
            mat[i, j] += 1.0
    faiss.normalize_L2(mat)
    return mat


def ingest_records(
    records: Iterable[Dict],
    index_dir: Path,
    embedding_model: str,
    batch_size: int = 64,
    max_records: Optional[int] = None,
    embed_provider: str = "openai",
    hash_dim: int = 1024,
) -> None:
    ensure_dir(index_dir)
    metas_path = index_dir / "meta.json"
    index_path = index_dir / "index.faiss"

    # Load or init index and meta
    index = faiss.read_index(str(index_path)) if index_path.exists() else None
    if index is None:
        # initialize lazily when we know embedding dim
        pass
    meta = json.loads(metas_path.read_text(encoding="utf-8")) if metas_path.exists() else {"embedding_model": embedding_model, "embedding_provider": embed_provider, "embedding_dim": None, "chunks": []}

    buffer: List[Dict] = []
    total_new = 0

    def flush(buf: List[Dict]):
        nonlocal index, meta, total_new
        if not buf:
            return
        chunks = chunk_docs(buf)
        texts = [d["content"] for d in chunks]
        if embed_provider == "hash":
            embs = _embed_texts_hash(texts, dim=hash_dim)
            if meta.get("embedding_dim") is None:
                meta["embedding_dim"] = hash_dim
        else:
            embs = _embed_texts_openai(texts, embedding_model)
            if meta.get("embedding_dim") is None:
                meta["embedding_dim"] = int(embs.shape[1])
        if index is None:
            index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        meta["chunks"].extend(chunks)
        total_new += len(chunks)

    processed = 0
    for r in records:
        d = normalize_record(r)
        if not d:
            continue
        buffer.append(d)
        processed += 1
        if max_records is not None and processed >= max_records:
            # flush and break
            flush(buffer)
            buffer.clear()
            break
        if len(buffer) >= batch_size:
            flush(buffer)
            buffer.clear()

    flush(buffer)

    if index is None:
        print("No records to ingest.")
        return

    faiss.write_index(index, str(index_path))
    metas_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"Saved FAISS index at {index_dir} with {len(meta['chunks'])} total chunks (+{total_new} new)")


def read_hf(hf_dataset: str, split: str) -> Iterator[Dict]:
    from datasets import load_dataset  # lazy import

    ds = load_dataset(hf_dataset, split=split)
    for r in ds:
        yield dict(r)


def ingest(
    input_path: Optional[Path],
    index_dir: Path,
    embedding_model: str,
    hf_dataset: Optional[str] = None,
    split: str = "train",
    batch_size: int = 64,
    max_records: Optional[int] = None,
    embed_provider: str = "openai",
    hash_dim: int = 1024,
) -> None:
    if input_path:
        ext = input_path.suffix.lower()
        if ext == ".csv":
            records = read_csv(input_path)
        elif ext in {".jsonl", ".ndjson"}:
            records = read_jsonl(input_path)
        else:
            raise ValueError("Unsupported file type. Use .csv or .jsonl")
        ingest_records(records, index_dir, embedding_model, batch_size=batch_size, max_records=max_records, embed_provider=embed_provider, hash_dim=hash_dim)
    elif hf_dataset:
        records = read_hf(hf_dataset, split)
        ingest_records(records, index_dir, embedding_model, batch_size=batch_size, max_records=max_records, embed_provider=embed_provider, hash_dim=hash_dim)
    else:
        raise ValueError("Provide a file path or --hf-dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Q/A data into a FAISS vector store")
    parser.add_argument("input", nargs="?", type=Path, help="Path to .csv or .jsonl with question/answer fields")
    parser.add_argument("--hf-dataset", dest="hf_dataset", help="HuggingFace dataset id, e.g. toughdata/quora-question-answer-dataset")
    parser.add_argument("--split", default="train", help="Dataset split when using --hf-dataset (default: train)")
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
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    parser.add_argument("--max-records", type=int, default=None, help="Limit number of input records for ingestion")
    parser.add_argument("--embed-provider", choices=["openai", "hash"], default=os.getenv("EMBED_PROVIDER", "openai"), help="Embedding provider: openai or hash (local)")
    parser.add_argument("--hash-dim", type=int, default=int(os.getenv("HASH_DIM", 1024)), help="Hash embedding dimension when using --embed-provider hash")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    input_path = args.input.resolve() if args.input else None
    ingest(
        input_path,
        args.index_dir.resolve(),
        args.embedding_model,
        hf_dataset=args.hf_dataset,
        split=args.split,
        batch_size=args.batch_size,
        max_records=args.max_records,
        embed_provider=args.embed_provider,
        hash_dim=args.hash_dim,
    )


if __name__ == "__main__":
    main()
