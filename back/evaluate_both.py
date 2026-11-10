# evaluate_both.py
import time
import numpy as np
import ir_datasets
import bm25s
import Stemmer
from sentence_transformers import SentenceTransformer
from measure import recall_at_k, mrr_at_k, ndcg_at_k

# 1. Dataset & QRELS
print("Loading SciFact...")
ds = ir_datasets.load("beir/scifact/test")

DOCS = [{"id": d.doc_id, "text": f"{d.title} {d.text}".strip()} for d in ds.docs_iter()]
doc_ids = [d["id"] for d in DOCS]
doc_texts = [d["text"] for d in DOCS]

# Ground truth load karna
qrels = {}
for q in ds.qrels_iter():
    qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance

# Sirf judged queries retain karo
queries = [{"id": q.query_id, "text": q.text} for q in ds.queries_iter() if qrels.get(q.query_id)]
print(f"Loaded {len(DOCS)} docs, {len(queries)} evaluated queries.")

K = 10

# 2. BM25 (Stemmed)
print("\nIndexing BM25...")
stemmer = Stemmer.Stemmer("english")
retriever = bm25s.BM25(corpus=doc_ids, k1=1.5, b=0.75)
retriever.index(bm25s.tokenize(doc_texts, stemmer=stemmer, show_progress=False))

# 3. Dense Embeddings (MiniLM)
print("Encoding Docs with all-MiniLM-L6-v2 (CPU par ~2-3 mins lagenge)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

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
def bm25_search(q_text):
    tokens = bm25s.tokenize(q_text, stemmer=stemmer, show_progress=False)
    res, _ = retriever.retrieve(tokens, k=K)
    return res[0].tolist()

def dense_search(q_text):
    q_vec = model.encode([q_text], normalize_embeddings=True, show_progress_bar=False)
    scores = doc_emb @ q_vec[0]
    top_k = np.argsort(-scores)[:K]
    return [doc_ids[i] for i in top_k]

evaluate_arm("BM25 + Stemmer", bm25_search)
evaluate_arm("Dense (all-MiniLM-L6-v2)", dense_search)
