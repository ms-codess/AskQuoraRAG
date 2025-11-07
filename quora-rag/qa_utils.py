import os, pickle
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index.bin")
TEXTS_PATH = os.getenv("TEXTS_PATH", "texts.pkl")
DATASET = os.getenv("DATASET", "toughdata/quora-question-answer-dataset")
MAX_ROWS = int(os.getenv("MAX_ROWS", "3000"))

def load_quora_pairs():
    ds = load_dataset(DATASET, split="train")
    rows = [{"q": r["question"], "a": r["answer"]} for r in ds]
    return rows

def build_corpus(rows, max_rows=3000):
    rows = rows[:max_rows]
    texts = [f"Q: {r['q']}\nA: {r['a']}" for r in rows]
    return texts

def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def encode(texts, model):
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def build_or_load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        with open(TEXTS_PATH, "rb") as f:
            texts = pickle.load(f)
        index = faiss.read_index(INDEX_PATH)
        model = get_embedder()
        return model, texts, index

    rows = load_quora_pairs()
    texts = build_corpus(rows, max_rows=MAX_ROWS)
    model = get_embedder()
    embs = encode(texts, model)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine (normalized vectors)
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    return model, texts, index

def search(model, index, texts, query, k=3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k)
    return [(texts[i], float(D[0][j])) for j, i in enumerate(I[0])]
