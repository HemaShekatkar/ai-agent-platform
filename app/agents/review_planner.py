from dataclasses import dataclass
from enum import Enum

from app.agents.multi_agent_models import (
    ReviewEvidence,
    ReviewStatus,
)
from app.agents.review_freshness import is_review_stale


class ReviewAction(str, Enum):
    REUSE = "REUSE"
    EXECUTE = "EXECUTE"


@dataclass(frozen=True)
class ReviewRequest:
    reviewer: str
    partner_id: str
    jurisdiction: str
    policy_version: str
    document_version: str


@dataclass(frozen=True)
class ReviewPlan:
    reviewer: str
    action: ReviewAction
    reason: str


def plan_review(
    request: ReviewRequest,
    cached_review: ReviewEvidence | None,
) -> ReviewPlan:

    if cached_review is None:
        return ReviewPlan(
            reviewer=request.reviewer,
            action=ReviewAction.EXECUTE,
            reason="No cached review",
        )

    # Never reuse another partner's or specialist's result.
    if (
        cached_review.partner_id != request.partner_id
        or cached_review.reviewer != request.reviewer
    ):
        return ReviewPlan(
            reviewer=request.reviewer,
            action=ReviewAction.EXECUTE,
            reason="Cached review identity mismatch",
        )

    if is_review_stale(
        cached_review,
        current_policy_version=request.policy_version,
        current_document_version=request.document_version,
        current_jurisdiction=request.jurisdiction,
    ):
        return ReviewPlan(
            reviewer=request.reviewer,
            action=ReviewAction.EXECUTE,
            reason="Review dependencies changed",
        )

    # An incomplete or failed execution cannot be reused.
    if cached_review.status in {
        ReviewStatus.ERROR,
        ReviewStatus.NEEDS_REVIEW,
    }:
        return ReviewPlan(
            reviewer=request.reviewer,
            action=ReviewAction.EXECUTE,
            reason="Previous review was incomplete",
        )

    # PASSED and FAILED are completed assessments.
    # Reusing FAILED does not mean permitting onboarding.
    return ReviewPlan(
        reviewer=request.reviewer,
        action=ReviewAction.REUSE,
        reason="Completed review with matching dependencies",
    )
