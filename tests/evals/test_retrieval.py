
from app.evals.retrieval import recall_at_k, reciprocal_rank, precision_at_k
import pytest

RELEVANT = {
    "security-17",
    "pii-42",
}

RETRIEVED = [
    "metadata-8",
    "security-17",
    "billing-3",
    "pii-42",
    "ops-9",
]


def test_recall_at_k():
    assert recall_at_k(
        RELEVANT,
        RETRIEVED,
        k=3,
    ) == 0.5

    assert recall_at_k(
        RELEVANT,
        RETRIEVED,
        k=5,
    ) == 1.0


def test_precision_at_k():
    assert precision_at_k(
        RELEVANT,
        RETRIEVED,
        k=3,
    ) == pytest.approx(1 / 3)

    assert precision_at_k(
        RELEVANT,
        RETRIEVED,
        k=5,
    ) == pytest.approx(2 / 5)


def test_reciprocal_rank():
    assert reciprocal_rank(
        RELEVANT,
        RETRIEVED,
    ) == 0.5


def test_reciprocal_rank_returns_zero_when_no_relevant_chunk_found():
    assert reciprocal_rank(
        {"security-17"},
        ["metadata-8", "billing-3"],
    ) == 0.0


def test_recall_rejects_empty_ground_truth():
    with pytest.raises(ValueError):
        recall_at_k(
            set(),
            RETRIEVED,
            k=5,
        )


def test_recall_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k(
            RELEVANT,
            RETRIEVED,
            k=0,
        )


def test_precision_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k(
            RELEVANT,
            RETRIEVED,
            k=0,
        )
