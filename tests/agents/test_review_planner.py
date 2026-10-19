import pytest

from app.agents.multi_agent_models import (
    ReviewEvidence,
    ReviewStatus,
)
from app.agents.review_planner import (
    ReviewAction,
    ReviewRequest,
    plan_review,
)


def make_request():
    return ReviewRequest(
        reviewer="security",
        partner_id="partner-123",
        jurisdiction="US",
        policy_version="v4",
        document_version="v2",
    )


def make_cached_review(
    status=ReviewStatus.PASSED,
    policy_version="v4",
    partner_id="partner-123",
):
    return ReviewEvidence(
        reviewer="security",
        partner_id=partner_id,
        jurisdiction="US",
        policy_version=policy_version,
        document_version="v2",
        status=status,
    )


def test_valid_review_is_reused():
    result = plan_review(
        make_request(),
        make_cached_review(),
    )

    assert result.action == ReviewAction.REUSE


def test_missing_review_requires_execution():
    result = plan_review(
        make_request(),
        None,
    )

    assert result.action == ReviewAction.EXECUTE


def test_policy_change_requires_execution():
    result = plan_review(
        make_request(),
        make_cached_review(policy_version="v3"),
    )

    assert result.action == ReviewAction.EXECUTE


def test_partner_mismatch_requires_execution():
    result = plan_review(
        make_request(),
        make_cached_review(partner_id="partner-456"),
    )

    assert result.action == ReviewAction.EXECUTE


@pytest.mark.parametrize(
    "status",
    [
        ReviewStatus.ERROR,
        ReviewStatus.NEEDS_REVIEW,
    ],
)
def test_incomplete_review_is_not_reused(status):
    result = plan_review(
        make_request(),
        make_cached_review(status=status),
    )

    assert result.action == ReviewAction.EXECUTE


def test_completed_failed_review_can_be_reused():
    result = plan_review(
        make_request(),
        make_cached_review(status=ReviewStatus.FAILED),
    )

    assert result.action == ReviewAction.REUSE
