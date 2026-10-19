from enum import Enum
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ERROR = "ERROR"


class ReviewResult(BaseModel):
    reviewer: str
    status: ReviewStatus
    findings: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class OnboardingReadiness(str, Enum):
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    BLOCKED = "BLOCKED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


def merge_reviews(
    left: dict[str, ReviewResult],
    right: dict[str, ReviewResult],
) -> dict[str, ReviewResult]:
    overlap = set(left) & set(right)

    if overlap:
        raise ValueError(
            f"Duplicate review results: {sorted(overlap)}"
        )

    return {**left, **right}


class MultiAgentState(TypedDict, total=False):
    partner_id: str
    jurisdiction: str
    metadata: dict
    security_document: str
    contract_document: str

    reviews: Annotated[
        dict[str, ReviewResult],
        merge_reviews,
    ]

    readiness: OnboardingReadiness
    summary: str


class ReviewEvidence(BaseModel):
    reviewer: str
    partner_id: str
    jurisdiction: str
    policy_version: str
    document_version: str
    status: ReviewStatus
