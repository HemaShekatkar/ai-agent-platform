from app.agents.multi_agent_models import (
    ReviewEvidence,
    ReviewStatus,
)
from app.agents.review_freshness import (
    is_review_stale,
    get_reviews_to_rerun,
)


def make_review(
    reviewer="security",
    policy_version="v3",
    document_version="v2",
    jurisdiction="US",
):
    return ReviewEvidence(
        reviewer=reviewer,
        partner_id="partner-123",
        jurisdiction=jurisdiction,
        policy_version=policy_version,
        document_version=document_version,
        status=ReviewStatus.PASSED,
    )


def test_unchanged_versions_are_fresh():
    review = make_review()

    assert not is_review_stale(
        review, "v3", "v2", "US"
    )


def test_new_policy_invalidates_review():
    review = make_review()

    assert is_review_stale(
        review, "v4", "v2", "US"
    )


def test_new_document_invalidates_review():
    review = make_review()

    assert is_review_stale(
        review, "v3", "v3", "US"
    )


def test_jurisdiction_change_invalidates_review():
    review = make_review()

    assert is_review_stale(
        review, "v3", "v2", "EU"
    )


def test_only_changed_review_is_selected():
    reviews = {
        "security": make_review("security", "v3"),
        "contract": make_review("contract", "v5"),
    }

    current_versions = {
        "security": {
            "policy_version": "v4",
            "document_version": "v2",
            "jurisdiction": "US",
        },
        "contract": {
            "policy_version": "v5",
            "document_version": "v2",
            "jurisdiction": "US",
        },
    }

    assert get_reviews_to_rerun(
        reviews, current_versions
    ) == ["security"]


def test_missing_version_info_requires_rerun():
    reviews = {
        "security": make_review(),
    }

    assert get_reviews_to_rerun(
        reviews, {}
    ) == ["security"]
