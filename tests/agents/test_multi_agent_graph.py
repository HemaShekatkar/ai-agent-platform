import pytest

from app.agents.multi_agent_graph import (
    build_multi_agent_graph,
)

from app.agents.multi_agent_models import (
    OnboardingReadiness,
    ReviewStatus,
)


def make_state():
    return {
        "partner_id": "partner-123",
        "jurisdiction": "US",
        "metadata": {
            "name": "Example Streaming Co.",
            "category": "streaming",
            "support_email": "support@example.com",
        },
        "security_document": "Security assessment submitted",
        "contract_document": "Agreement submitted",
        "reviews": {},
    }


@pytest.mark.asyncio
async def test_parallel_reviews_are_aggregated():

    graph = build_multi_agent_graph()

    result = await graph.ainvoke(make_state())

    assert len(result["reviews"]) == 3

    assert (
        result["reviews"]["metadata"].status
        == ReviewStatus.PASSED
    )

    assert (
        result["reviews"]["security"].status
        == ReviewStatus.NEEDS_REVIEW
    )

    assert (
        result["reviews"]["contract"].status
        == ReviewStatus.NEEDS_REVIEW
    )

    assert (
        result["readiness"]
        == OnboardingReadiness.HUMAN_REVIEW
    )


@pytest.mark.asyncio
async def test_missing_metadata_blocks_onboarding():

    graph = build_multi_agent_graph()

    state = make_state()
    state["metadata"].pop("support_email")

    result = await graph.ainvoke(state)

    assert (
        result["reviews"]["metadata"].status
        == ReviewStatus.FAILED
    )

    assert (
        result["readiness"]
        == OnboardingReadiness.BLOCKED
    )


@pytest.mark.asyncio
async def test_missing_documents_require_review():

    graph = build_multi_agent_graph()

    state = make_state()
    state["security_document"] = ""
    state["contract_document"] = ""

    result = await graph.ainvoke(state)

    assert (
        result["reviews"]["security"].status
        == ReviewStatus.NEEDS_REVIEW
    )

    assert (
        result["reviews"]["contract"].status
        == ReviewStatus.NEEDS_REVIEW
    )

    assert (
        result["readiness"]
        == OnboardingReadiness.HUMAN_REVIEW
    )
