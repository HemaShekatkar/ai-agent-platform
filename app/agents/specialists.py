from app.agents.multi_agent_models import (
    MultiAgentState,
    ReviewResult,
    ReviewStatus,
)
import asyncio
from typing import Awaitable, Callable


def validate_metadata(
    state: MultiAgentState,
) -> dict:

    metadata = state.get("metadata", {})

    required = {"name", "category", "support_email"}
    missing = required - set(metadata)

    if missing:
        result = ReviewResult(
            reviewer="metadata",
            status=ReviewStatus.FAILED,
            findings=[
                f"Missing fields: {sorted(missing)}"
            ],
        )
    else:
        result = ReviewResult(
            reviewer="metadata",
            status=ReviewStatus.PASSED,
            findings=["Required metadata fields present"],
        )

    return {"reviews": {"metadata": result}}


async def security_specialist(
    state: MultiAgentState,
) -> dict:

    document = state.get("security_document", "")

    # Temporary stub, NOT an actual security assessment.
    if not document.strip():
        result = ReviewResult(
            reviewer="security",
            status=ReviewStatus.NEEDS_REVIEW,
            findings=["Security evidence missing"],
        )
    else:
        result = ReviewResult(
            reviewer="security",
            status=ReviewStatus.NEEDS_REVIEW,
            findings=[
                "Security document received; specialist review pending"
            ],
        )

    return {"reviews": {"security": result}}


async def contract_specialist(
    state: MultiAgentState,
) -> dict:

    document = state.get("contract_document", "")

    # Temporary stub, NOT an actual contract assessment.
    if not document.strip():
        result = ReviewResult(
            reviewer="contract",
            status=ReviewStatus.NEEDS_REVIEW,
            findings=["Contract evidence missing"],
        )
    else:
        result = ReviewResult(
            reviewer="contract",
            status=ReviewStatus.NEEDS_REVIEW,
            findings=[
                "Contract document received; specialist review pending"
            ],
        )

    return {"reviews": {"contract": result}}

Specialist = Callable[
    [MultiAgentState],
    Awaitable[dict],
]


def with_recovery(
    reviewer: str,
    specialist: Specialist,
    max_attempts: int = 2,
    timeout_seconds: float = 2.0,
) -> Specialist:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    async def run(state: MultiAgentState) -> dict:
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                return await asyncio.wait_for(
                    specialist(state),
                    timeout=timeout_seconds,
                )
            except (TimeoutError, ConnectionError) as exc:
                last_error = exc

                if attempt + 1 < max_attempts:
                    await asyncio.sleep(0)

        return {
            "reviews": {
                reviewer: ReviewResult(
                    reviewer=reviewer,
                    status=ReviewStatus.ERROR,
                    findings=[
                        "Specialist unavailable after bounded retries"
                    ],
                    evidence=[],
                )
            }
        }

    return run
