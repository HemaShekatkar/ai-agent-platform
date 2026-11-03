
from app.agents.models import (
    ActionType,
    AgentAction,
    AgentExecutionState,
    AgentStatus,
    ToolResult,
    ToolStatus,
)
from app.agents.runtime import run_agent


STATUS = AgentAction(ActionType.GET_ONBOARDING_STATUS, "ONB-123")
REVIEW = AgentAction(ActionType.INITIATE_SECURITY_REVIEW, "ONB-123")
CONTRACT = AgentAction(ActionType.REQUEST_CONTRACT, "ONB-123")


def sequence_planner(actions):
    remaining = iter(actions)
    return lambda _: next(remaining, None)


def test_successful_execution():
    result = run_agent(
        planner=sequence_planner([STATUS]),
        execute_tool=lambda _: ToolResult(ToolStatus.SUCCESS),
        validate_action=lambda _: True,
    )

    assert result.status == AgentStatus.COMPLETED
    assert result.steps_used == 1


def test_permission_denied_allows_independent_work():
    def execute(action):
        if action == REVIEW:
            return ToolResult(
                ToolStatus.PERMISSION_DENIED,
                reason="USER_NOT_AUTHORIZED",
            )
        return ToolResult(ToolStatus.SUCCESS)

    result = run_agent(
        planner=sequence_planner([REVIEW, STATUS]),
        execute_tool=execute,
        validate_action=lambda _: True,
    )

    assert result.steps_used == 2
    assert REVIEW in result.blocked_actions
    assert result.history[-1][0] == STATUS


def test_precondition_failure_blocks_repeated_action():
    result = run_agent(
        planner=sequence_planner([REVIEW, REVIEW]),
        execute_tool=lambda _: ToolResult(
            ToolStatus.PRECONDITION_FAILED,
            reason="CONTRACT_NOT_SIGNED",
            state_version="v1",
        ),
        validate_action=lambda _: True,
    )

    assert result.steps_used == 1
    assert REVIEW in result.blocked_actions


def test_transient_failure_is_retried_with_limit():
    attempts = 0

    def execute(_):
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            return ToolResult(ToolStatus.TRANSIENT_FAILURE)

        return ToolResult(ToolStatus.SUCCESS)

    result = run_agent(
        planner=sequence_planner([STATUS]),
        execute_tool=execute,
        validate_action=lambda _: True,
        max_retries_per_action=2,
    )

    assert attempts == 3
    assert result.status == AgentStatus.COMPLETED


def test_step_budget_is_enforced():
    result = run_agent(
        planner=sequence_planner([STATUS, CONTRACT, REVIEW]),
        execute_tool=lambda _: ToolResult(ToolStatus.SUCCESS),
        validate_action=lambda _: True,
        state=AgentExecutionState(max_steps=2),
    )

    assert result.steps_used == 2
    assert result.status == AgentStatus.BUDGET_EXCEEDED


def test_invalid_planner_cannot_loop_forever():
    calls = 0

    def planner(_):
        nonlocal calls
        calls += 1
        return REVIEW

    result = run_agent(
        planner=planner,
        execute_tool=lambda _: ToolResult(ToolStatus.SUCCESS),
        validate_action=lambda _: False,
        state=AgentExecutionState(
            max_steps=10,
            max_planning_attempts=3,
        ),
    )

    assert calls == 2
    assert result.steps_used == 0
    assert result.status == AgentStatus.BLOCKED
