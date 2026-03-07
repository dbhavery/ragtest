"""Retrieval metrics — pure-Python implementations of standard IR metrics.

All functions operate on lists of document IDs.  No external dependencies.
"""

from __future__ import annotations

import math


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Fraction of relevant documents that appear in the top-k retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: List of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score in [0.0, 1.0].  Returns 0.0 when *relevant_ids* is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Fraction of top-k retrieved documents that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: List of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision score in [0.0, 1.0].  Returns 0.0 when *k* is 0.
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / k


def mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: List of ground-truth relevant document IDs.

    Returns:
        MRR score in [0.0, 1.0].  Returns 0.0 when no relevant document is
        found in *retrieved_ids*.
    """
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first).
        relevant_ids: List of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        NDCG score in [0.0, 1.0].  Returns 0.0 when *relevant_ids* is empty
        or *k* is 0.
    """
    if k == 0 or not relevant_ids:
        return 0.0

    relevant = set(relevant_ids)

    # DCG — binary relevance
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG — all relevant documents at the top
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
