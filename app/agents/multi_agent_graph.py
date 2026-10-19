from langgraph.graph import StateGraph, START, END

from app.agents.multi_agent_models import (
    MultiAgentState,
    OnboardingReadiness,
    ReviewStatus,
)

from app.agents.specialists import (
    validate_metadata,
    security_specialist,
    contract_specialist,
    with_recovery,
)


def aggregate_reviews(
    state: MultiAgentState,
) -> dict:

    reviews = state.get("reviews", {})

    expected = {"metadata", "security", "contract"}

    if set(reviews) != expected:
        readiness = OnboardingReadiness.HUMAN_REVIEW

    elif any(
        review.status == ReviewStatus.FAILED
        for review in reviews.values()
    ):
        readiness = OnboardingReadiness.BLOCKED

    elif all(
        review.status == ReviewStatus.PASSED
        for review in reviews.values()
    ):
        readiness = OnboardingReadiness.READY_FOR_APPROVAL

    else:
        readiness = OnboardingReadiness.HUMAN_REVIEW

    return {
        "readiness": readiness,
        "summary": (
            f"Partner {state['partner_id']} "
            f"readiness: {readiness.value}"
        ),
    }


def build_multi_agent_graph():

    graph = StateGraph(MultiAgentState)

    graph.add_node(
        "validate_metadata",
        validate_metadata,
    )

    graph.add_node(
        "security_specialist",
        with_recovery(
            "security",
            security_specialist,
        ),
    )

    graph.add_node(
        "contract_specialist",
        with_recovery(
            "contract",
            contract_specialist,
        ),
    )

    graph.add_node(
        "aggregate_reviews",
        aggregate_reviews,
    )

    # Fan out to three independent checks.
    graph.add_edge(START, "validate_metadata")
    graph.add_edge(START, "security_specialist")
    graph.add_edge(START, "contract_specialist")

    # Fan in after all three finish.
    graph.add_edge(
        [
            "validate_metadata",
            "security_specialist",
            "contract_specialist",
        ],
        "aggregate_reviews",
    )

    graph.add_edge("aggregate_reviews", END)

    return graph.compile()
