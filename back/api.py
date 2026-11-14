"""FastAPI REST service exposing BM25, Dense, and Hybrid RRF retrieval over SciFact."""

from contextlib import asynccontextmanager
from functools import lru_cache
import time
from typing import Literal

import bm25s
from fastapi import FastAPI, HTTPException
import ir_datasets
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import Stemmer

# Global state populated once on startup
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load corpus, BM25 index, and precomputed embeddings into memory at startup."""
    print("Initializing retrieval indices...")
    ds = ir_datasets.load("beir/scifact/test")

    doc_ids = []
    doc_texts = []
    doc_map = {}
    for d in ds.docs_iter():
        text = f"{d.title} {d.text}".strip()
        doc_ids.append(d.doc_id)
        doc_texts.append(text)
        doc_map[d.doc_id] = {"title": d.title, "abstract": d.text}

    # 1. Build BM25 + Stemmer index
    stemmer = Stemmer.Stemmer("english")
    bm25 = bm25s.BM25(corpus=doc_ids, k1=1.5, b=0.75)
    bm25.index(bm25s.tokenize(doc_texts, stemmer=stemmer, show_progress=False))

    # 2. Load Dense Encoder & Cached Matrix
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_emb = np.load("doc_emb.npy")

    state.update({
        "doc_ids": doc_ids,
        "doc_map": doc_map,
        "stemmer": stemmer,
        "bm25": bm25,
        "encoder": encoder,
        "doc_emb": doc_emb,
    })
    print(f"Server ready with {len(doc_ids)} indexed documents.")
    yield
    state.clear()


app = FastAPI(
    title="SciFact Hybrid Retrieval API",
    description="Low-latency Lexical (BM25) and Dense Bi-Encoder Search Engine",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Request / Response Schemas ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, example="Ivermectin treats onchocerciasis")
    mode: Literal["bm25", "dense", "hybrid"] = Field(default="bm25")
    top_k: int = Field(default=5, ge=1, le=50)


class DocumentHit(BaseModel):
    rank: int
    doc_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    latency_ms: float
    cached: bool
    results: list[DocumentHit]


# --- Core Retrieval Helpers (Cached for Repeat Queries) ---
@lru_cache(maxsize=256)
def execute_search_cached(query: str, mode: str, top_k: int) -> tuple[tuple[str, float], ...]:
    """Run retrieval and return immutable tuples so results can be LRU-cached."""
    doc_ids = state["doc_ids"]

    if mode == "bm25":
        tokens = bm25s.tokenize(query, stemmer=state["stemmer"], show_progress=False)
        res, scores = state["bm25"].retrieve(tokens, k=top_k)
        return tuple(zip(res[0].tolist(), [float(s) for s in scores[0]]))

    if mode == "dense":
        q_vec = state["encoder"].encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = state["doc_emb"] @ q_vec[0]
        top_idx = np.argsort(-scores)[:top_k]
        return tuple((doc_ids[i], float(scores[i])) for i in top_idx)

    if mode == "hybrid":
        # Reciprocal Rank Fusion (k=60) combining top-50 BM25 and Dense candidates
        pool_k = max(50, top_k)
        bm25_hits = execute_search_cached(query, "bm25", pool_k)
        dense_hits = execute_search_cached(query, "dense", pool_k)

        rrf_scores: dict[str, float] = {}
        for rank, (did, _) in enumerate(bm25_hits, start=1):
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + rank)
        for rank, (did, _) in enumerate(dense_hits, start=1):
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (60 + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return tuple(ranked)

    raise ValueError(f"Unsupported mode: {mode}")


# --- REST Endpoints ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "indexed_documents": len(state.get("doc_ids", [])),
    }


@app.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest):
    if "bm25" not in state:
        raise HTTPException(status_code=503, detail="Indices are still loading.")

    hits_before = execute_search_cached.cache_info().hits
    t0 = time.perf_counter()

    try:
        raw_hits = execute_search_cached(payload.query.strip(), payload.mode, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    is_cached = execute_search_cached.cache_info().hits > hits_before

    doc_map = state["doc_map"]
    formatted_hits = [
        DocumentHit(
            rank=idx,
            doc_id=doc_id,
            score=round(score, 4),
            title=doc_map[doc_id]["title"],
            snippet=doc_map[doc_id]["abstract"][:220] + "...",
        )
        for idx, (doc_id, score) in enumerate(raw_hits, start=1)
    ]

    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        latency_ms=latency_ms,
        cached=is_cached,
        results=formatted_hits,
    )