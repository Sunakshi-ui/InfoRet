
import math

def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if relevant.get(doc_id, 0) > 0)
    return found / len(relevant)

def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        (2 ** relevant.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(relevant.values(), reverse=True)[:k], start=1)
    )
    return dcg / ideal if ideal else 0.0