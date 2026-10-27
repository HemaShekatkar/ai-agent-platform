
from app.rag.coverage_reranker import (
    RankedDocument,
    select_with_coverage,
)


def test_all_required_regions_are_covered():
    candidates = [
        RankedDocument("US-1", "US", 0.99),
        RankedDocument("US-2", "US", 0.97),
        RankedDocument("EU-1", "EU", 0.93),
        RankedDocument("US-3", "US", 0.91),
        RankedDocument("EU-2", "EU", 0.89),
        RankedDocument("CA-1", "CA", 0.75),
    ]

    result = select_with_coverage(
        candidates,
        required_regions={"US", "EU", "CA"},
        top_k=5,
    )

    assert result.complete is True
    assert result.missing_regions == frozenset()

    selected_regions = {
        doc.region for doc in result.documents
    }

    assert selected_regions == {"US", "EU", "CA"}
    assert len(result.documents) == 5


def test_missing_region_is_reported():
    candidates = [
        RankedDocument("US-1", "US", 0.99),
        RankedDocument("EU-1", "EU", 0.95),
    ]

    result = select_with_coverage(
        candidates,
        required_regions={"US", "EU", "CA"},
        top_k=3,
    )

    assert result.complete is False
    assert result.missing_regions == frozenset({"CA"})


def test_insufficient_context_budget_is_rejected():
    candidates = [
        RankedDocument("US-1", "US", 0.99),
    ]

    try:
        select_with_coverage(
            candidates,
            required_regions={"US", "EU", "CA"},
            top_k=2,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass
