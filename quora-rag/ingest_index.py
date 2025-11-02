import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def read_csv(path: Path):
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def read_jsonl(path: Path):
    import json

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_docs(records) -> List[Document]:
    docs: List[Document] = []
    for r in records:
        q = (r.get("question") or r.get("Question") or r.get("q") or "").strip()
        a = (r.get("answer") or r.get("Answer") or r.get("a") or "").strip()
        _id = (r.get("id") or r.get("_id") or r.get("uuid") or None)

        if not q and not a:
            continue

        content = f"Question: {q}\nAnswer: {a}".strip()
        meta = {"question": q or None, "id": _id}
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ".", "! ", "? ", "?", ", ", ",", " ", ""],
    )
    return splitter.split_documents(docs)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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

    embeddings = OpenAIEmbeddings(model=embedding_model)

    ensure_dir(index_dir)

    index_path = index_dir
    # Load existing or create new
    if (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists():
        db = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        db.add_documents(docs)
        db.save_local(str(index_path))
        print(f"Added {len(docs)} chunks to existing FAISS index at {index_path}")
    else:
        db = FAISS.from_documents(documents=docs, embedding=embeddings)
        db.save_local(str(index_path))
        print(f"Created new FAISS index at {index_path} with {len(docs)} chunks")


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
