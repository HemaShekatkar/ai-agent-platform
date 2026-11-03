
from app.agents.dependencies import (
    DependencyValidator,
    WorkflowGraph,
)
from app.agents.models import ActionType, AgentAction


CONTRACT = AgentAction(
    ActionType.REQUEST_CONTRACT, "ONB-123"
)
REVIEW = AgentAction(
    ActionType.INITIATE_SECURITY_REVIEW, "ONB-123"
)
STATUS = AgentAction(
    ActionType.GET_ONBOARDING_STATUS, "ONB-123"
)


def make_validator():
    graph = WorkflowGraph(
        version="v1",
        dependencies={
            CONTRACT: frozenset(),
            REVIEW: frozenset({CONTRACT}),
            STATUS: frozenset(),
        },
    )
    return DependencyValidator(graph)


def test_independent_action_can_execute():
    validator = make_validator()
    assert validator.can_execute(STATUS, set())


def test_dependent_action_is_blocked():
    validator = make_validator()
    assert not validator.can_execute(REVIEW, set())
    assert validator.missing_prerequisites(
        REVIEW, set()
    ) == frozenset({CONTRACT})


def test_dependent_action_allowed_after_prerequisite():
    validator = make_validator()
    assert validator.can_execute(REVIEW, {CONTRACT})


def test_blocked_prerequisite_does_not_block_independent_work():
    validator = make_validator()
    assert validator.can_execute(STATUS, set())
    assert not validator.can_execute(REVIEW, set())
