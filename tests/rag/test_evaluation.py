
import pytest

from app.rag.evaluation import (
    evaluate_query,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_perfect_ranking():
    relevance = {"A": 3, "B": 2, "C": 1}
    ranked = ["A", "B", "C"]

    assert recall_at_k(ranked, relevance, 3) == 1.0
    assert precision_at_k(ranked, relevance, 3) == 1.0
    assert reciprocal_rank(ranked, relevance) == 1.0
    assert ndcg_at_k(ranked, relevance, 3) == pytest.approx(1.0)


def test_partial_retrieval():
    relevance = {"A": 3, "B": 2, "C": 0}
    ranked = ["C", "A", "X"]

    assert recall_at_k(ranked, relevance, 2) == 0.5
    assert precision_at_k(ranked, relevance, 2) == 0.5
    assert reciprocal_rank(ranked, relevance) == 0.5
    assert ndcg_at_k(ranked, relevance, 2) < 1.0


def test_no_relevant_documents_retrieved():
    relevance = {"A": 3}
    ranked = ["X", "Y"]

    assert recall_at_k(ranked, relevance, 2) == 0.0
    assert reciprocal_rank(ranked, relevance) == 0.0
    assert ndcg_at_k(ranked, relevance, 2) == 0.0


def test_evaluate_query():
    result = evaluate_query(
        query_id="q001",
        category="EXACT_POLICY",
        ranked_ids=["B", "A"],
        relevance={"A": 3, "B": 1},
        k=2,
    )

    assert result.query_id == "q001"
    assert result.category == "EXACT_POLICY"
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0


def test_rejects_duplicate_results():
    with pytest.raises(ValueError):
        evaluate_query(
            "q001",
            "GENERAL_FAQ",
            ["A", "A"],
            {"A": 3},
        )
