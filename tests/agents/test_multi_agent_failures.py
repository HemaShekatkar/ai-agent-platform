import asyncio

import pytest

from app.agents.multi_agent_models import (
    ReviewStatus,
)
from app.agents.specialists import (
    with_recovery,
)
from app.agents.multi_agent_graph import aggregate_reviews
from app.agents.multi_agent_models import (
    OnboardingReadiness,
    ReviewResult,
)


@pytest.mark.asyncio
async def test_transient_failure_recovers():

    attempts = 0

    async def flaky_specialist(state):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise ConnectionError("Temporary outage")

        return {
            "reviews": {
                "security": {
                    "reviewer": "security",
                    "status": ReviewStatus.PASSED,
                }
            }
        }

    wrapped = with_recovery(
        "security",
        flaky_specialist,
        max_attempts=2,
    )

    result = await wrapped({})

    assert attempts == 2
    assert (
        result["reviews"]["security"]["status"]
        == ReviewStatus.PASSED
    )


@pytest.mark.asyncio
async def test_timeout_returns_error():

    async def slow_specialist(state):
        await asyncio.sleep(0.1)
        return {"reviews": {}}

    wrapped = with_recovery(
        "security",
        slow_specialist,
        max_attempts=2,
        timeout_seconds=0.01,
    )

    result = await wrapped({})

    assert (
        result["reviews"]["security"].status
        == ReviewStatus.ERROR
    )


@pytest.mark.asyncio
async def test_permanent_error_propagates():

    async def broken_specialist(state):
        raise ValueError("Invalid configuration")

    wrapped = with_recovery(
        "security",
        broken_specialist,
    )

    with pytest.raises(ValueError):
        await wrapped({})


@pytest.mark.asyncio
async def test_retry_exhaustion_returns_error():

    attempts = 0

    async def unavailable_specialist(state):
        nonlocal attempts
        attempts += 1
        raise ConnectionError("Service unavailable")

    wrapped = with_recovery(
        "contract",
        unavailable_specialist,
        max_attempts=3,
    )

    result = await wrapped({})

    assert attempts == 3
    assert (
        result["reviews"]["contract"].status
        == ReviewStatus.ERROR
    )


def test_specialist_error_requires_human_review():

    state = {
        "partner_id": "partner-123",
        "reviews": {
            "metadata": ReviewResult(
                reviewer="metadata",
                status=ReviewStatus.PASSED,
            ),
            "security": ReviewResult(
                reviewer="security",
                status=ReviewStatus.ERROR,
            ),
            "contract": ReviewResult(
                reviewer="contract",
                status=ReviewStatus.PASSED,
            ),
        },
    }

    result = aggregate_reviews(state)

    assert (
        result["readiness"]
        == OnboardingReadiness.HUMAN_REVIEW
    )
