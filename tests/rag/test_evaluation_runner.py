
from app.rag.evaluation import QueryEvaluation
from app.rag.evaluation_runner import (
    QualityGate,
    evaluate_release,
)


def make_result(category, recall, mrr, ndcg):
    return QueryEvaluation(
        query_id=f"{category}-{recall}",
        category=category,
        recall_at_k=recall,
        precision_at_k=recall,
        reciprocal_rank=mrr,
        ndcg_at_k=ndcg,
    )


GATES = {
    "GENERAL_FAQ": QualityGate(0.85, 0.80, 0.80),
    "EXACT_POLICY": QualityGate(0.98, 0.95, 0.95),
}


def test_all_categories_pass():
    results = [
        make_result("GENERAL_FAQ", 0.90, 0.90, 0.90),
        make_result("EXACT_POLICY", 1.00, 1.00, 1.00),
    ]

    report = evaluate_release(results, GATES)

    assert report.passed is True
    assert all(c.passed for c in report.categories)


def test_policy_regression_blocks_release():
    results = [
        make_result("GENERAL_FAQ", 1.00, 1.00, 1.00),
        make_result("EXACT_POLICY", 0.80, 0.90, 0.90),
    ]

    report = evaluate_release(results, GATES)

    assert report.passed is False

    policy = next(
        c for c in report.categories
        if c.category == "EXACT_POLICY"
    )

    assert policy.passed is False
    assert "Recall below threshold" in policy.failures


def test_missing_category_blocks_release():
    results = [
        make_result("GENERAL_FAQ", 1.00, 1.00, 1.00)
    ]

    report = evaluate_release(results, GATES)

    assert report.passed is False
    assert any(
        c.category == "EXACT_POLICY" and not c.passed
        for c in report.categories
    )


def test_unknown_category_rejected():
    import pytest

    results = [
        make_result("UNRECOGNIZED", 1.00, 1.00, 1.00)
    ]

    with pytest.raises(ValueError):
        evaluate_release(results, GATES)
