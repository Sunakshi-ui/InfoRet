# Lexical vs. Dense Information Retrieval Evaluation Engine

A reproducible benchmark evaluating sparse lexical retrieval (**BM25**) against zero-shot dense semantic embeddings (**all-MiniLM-L6-v2**) on scientific abstract search using the **BEIR SciFact** dataset.

## Overview
Finding relevant scientific evidence requires balancing exact domain nomenclature (genes, drugs, chemical entities) with semantic paraphrasing. This repository implements both retrieval paradigms under identical corpus and hardware constraints, alongside first-principles implementations of standard ranking metrics (**nDCG@10**, **Recall@10**, **MRR@10**).

## Repository Structure
```text
ir-engine/
└── back/
    ├── baseline_bm25.py    # check: corpus ingestion & top-k BM25 retrieval
    ├── measure.py          # From-scratch implementations of Recall@k, MRR@k, and nDCG@k
    └── evaluate_both.py    # Full 300-query head-to-head evaluation & latency profiler

```
.
