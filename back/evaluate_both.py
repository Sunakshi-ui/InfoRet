
"""Comparative retrieval evaluation pipeline for BEIR SciFact.

Evaluates sparse lexical retrieval (BM25 with Snowball stemming) against
zero-shot dense semantic retrieval (all-MiniLM-L6-v2) across standard
ranking metrics (Recall@k, MRR@k, nDCG@k) and CPU query latency.
"""
import os
import time
import numpy as np
import ir_datasets
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer
from measure import recall_at_k, mrr_at_k, ndcg_at_k


# Evaluation config
print("Loading SciFact...")
ds = ir_datasets.load("beir/scifact/test")
TOP_K = 10
RRF_POOL_K = 50
RRF_K = 60
DATASET_NAME = "beir/scifact/test"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_PATH = "doc_emb.npy"
DOCS = [{"id": d.doc_id, "text": f"{d.title} {d.text}".strip()} for d in ds.docs_iter()]
doc_ids = [d["id"] for d in DOCS]
doc_texts = [d["text"] for d in DOCS]

# load
qrels = {}
for q in ds.qrels_iter():
    qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance

# Filter out unjudged queries to prevent zero-score distortion across arms
queries = [{"id": q.query_id, "text": q.text} for q in ds.queries_iter() if qrels.get(q.query_id)]
print(f"Loaded {len(DOCS)} docs, {len(queries)} evaluated queries.")

K = 10

# 1. Unstemmed BM25 Index
print("\nIndexing BM25 (Unstemmed)...")
bm25_raw = bm25s.BM25(corpus=doc_ids, k1=1.5, b=0.75)
bm25_raw.index(bm25s.tokenize(doc_texts, show_progress=False))

# 2. BM25 (Stemmed)
print("\nIndexing BM25...")
stemmer = Stemmer.Stemmer("english")
bm25_stemmed = bm25s.BM25(corpus=doc_ids, k1=1.5, b=0.75)
bm25_stemmed.index(bm25s.tokenize(doc_texts, stemmer=stemmer, show_progress=False))

# 3. Dense Embeddings (MiniLM)
print("Encoding Docs with all-MiniLM-L6-v2 (CPU par ~2-3 mins lagenge)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CACHE_PATH = "doc_emb.npy"

if os.path.exists(CACHE_PATH):
    print("Loading cached embeddings from doc_emb.npy...")
    doc_emb = np.load(CACHE_PATH)
else:
    print("Encoding Docs with all-MiniLM-L6-v2 (CPU par ~8 mins lagenge)...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(CACHE_PATH, doc_emb)

# 4. Evaluation Loop
def evaluate_arm(name, retrieve_fn):
    t0 = time.perf_counter()
    runs = {q["id"]: retrieve_fn(q["text"]) for q in queries}
    latency = (time.perf_counter() - t0) / len(queries) * 1000

    rec = np.mean([recall_at_k(runs[q["id"]], qrels[q["id"]], K) for q in queries])
    mrr = np.mean([mrr_at_k(runs[q["id"]], qrels[q["id"]], K) for q in queries])
    ndcg = np.mean([ndcg_at_k(runs[q["id"]], qrels[q["id"]], K) for q in queries])

    print(f"\n--- {name} (K={K}) ---")
    print(f"Recall@10: {rec:.4f} | MRR@10: {mrr:.4f} | nDCG@10: {ndcg:.4f}")
    print(f"Latency:   {latency:.2f} ms/query")

# Search wrappers
def bm25_bm25_unstemmed(q_text: str, k: int = TOP_K) -> list[str]:
    tokens = bm25s.tokenize(q_text, show_progress=False)
    res, _ = bm25_raw.retrieve(tokens, k=k)
    return res[0].tolist()

def search_bm25_stemmed(q_text: str, k: int = TOP_K) -> list[str]:
    tokens = bm25s.tokenize(q_text, stemmer=stemmer, show_progress=False)
    res, _ = bm25_stemmed.retrieve(tokens, k=k)
    return res[0].tolist()

def dense_search(q_text: str, k: int = TOP_K) -> list[str]:
    q_vec = model.encode([q_text], normalize_embeddings=True, show_progress_bar=False)
    scores = doc_emb @ q_vec[0]
    top_k = np.argsort(-scores)[:K]
    return [doc_ids[i] for i in top_k]

def search_hybrid_rrf(q_text: str, k: int = TOP_K) -> list[str]:
    bm25_hits = search_bm25_stemmed(q_text, k=RRF_POOL_K)
    dense_hits = dense_search(q_text, k=RRF_POOL_K)

    rrf_scores: dict[str, float] = {}
    for rank, did in enumerate(bm25_hits, start=1):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
    for rank, did in enumerate(dense_hits, start=1):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [did for did, _ in ranked]

evaluate_arm("BM25 (Unstemmed)", bm25_bm25_unstemmed)
evaluate_arm("BM25 + Stemmer", search_bm25_stemmed)
evaluate_arm("Dense (all-MiniLM-L6-v2)", dense_search)
evaluate_arm("Hybrid RRF (BM25 + Dense)", search_hybrid_rrf)