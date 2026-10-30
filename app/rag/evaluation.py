
from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    category: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def recall_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    relevant = {
        doc_id for doc_id, grade in relevance.items()
        if grade > 0
    }
    if not relevant:
        return 0.0

    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant_count = sum(
        relevance.get(doc_id, 0) > 0
        for doc_id in ranked_ids[:k]
    )
    return relevant_count / k


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    def dcg(grades: Sequence[int]) -> float:
        return sum(
            (2 ** grade - 1) / log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    actual_grades = [
        relevance.get(doc_id, 0)
        for doc_id in ranked_ids[:k]
    ]

    ideal_grades = sorted(
        relevance.values(), reverse=True
    )[:k]

    ideal = dcg(ideal_grades)
    return dcg(actual_grades) / ideal if ideal > 0 else 0.0


def evaluate_query(
    query_id: str,
    category: str,
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int = 5,
) -> QueryEvaluation:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked_ids must be unique")
    if any(grade < 0 for grade in relevance.values()):
        raise ValueError("relevance grades must be nonnegative")

    return QueryEvaluation(
        query_id=query_id,
        category=category,
        recall_at_k=recall_at_k(ranked_ids, relevance, k),
        precision_at_k=precision_at_k(ranked_ids, relevance, k),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevance),
        ndcg_at_k=ndcg_at_k(ranked_ids, relevance, k),
    )
