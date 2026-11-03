
from dataclasses import dataclass
from typing import Callable

from app.agents.models import (
    AgentAction,
    AgentExecutionState,
    AgentStatus,
    ToolResult,
    ToolStatus,
)
from app.agents.policy import RuntimeDecision, classify_tool_result


@dataclass(frozen=True)
class AgentRunResult:
    status: AgentStatus
    steps_used: int
    blocked_actions: tuple[AgentAction, ...]
    history: tuple[tuple[AgentAction, ToolResult], ...]


Planner = Callable[[AgentExecutionState], AgentAction | None]
ToolExecutor = Callable[[AgentAction], ToolResult]
ActionValidator = Callable[[AgentAction], bool]


def run_agent(
    planner: Planner,
    execute_tool: ToolExecutor,
    validate_action: ActionValidator,
    state: AgentExecutionState | None = None,
    max_retries_per_action: int = 2,
) -> AgentRunResult:
    state = state or AgentExecutionState()

    if max_retries_per_action < 0:
        raise ValueError("max_retries_per_action must be >= 0")

    blocked_actions: set[AgentAction] = set()
    retry_counts: dict[AgentAction, int] = {}
    final_status = AgentStatus.COMPLETED

    while state.steps_used < state.max_steps:
        if state.planning_attempts >= state.max_planning_attempts:
            final_status = AgentStatus.BUDGET_EXCEEDED
            break

        state.planning_attempts += 1
        action = planner(state)

        if action is None:
            break

        if action in blocked_actions:
            final_status = AgentStatus.BLOCKED
            break

        if not validate_action(action):
            blocked_actions.add(action)
            final_status = AgentStatus.BLOCKED
            continue

        # Each tool invocation consumes a step.
        while state.steps_used < state.max_steps:
            result = execute_tool(action)
            state.steps_used += 1
            state.action_history.append((action, result))

            decision = classify_tool_result(result)

            if decision == RuntimeDecision.CONTINUE:
                break

            if decision == RuntimeDecision.BLOCK_ACTION:
                blocked_actions.add(action)
                final_status = AgentStatus.BLOCKED
                break

            if decision == RuntimeDecision.REPLAN:
                signature = (
                    action.action_type.value,
                    action.resource_id,
                    result.reason or "",
                    result.state_version,
                )
                state.blocked_signatures.add(signature)
                blocked_actions.add(action)
                final_status = AgentStatus.BLOCKED
                break

            if decision == RuntimeDecision.RETRY:
                count = retry_counts.get(action, 0)

                if count >= max_retries_per_action:
                    blocked_actions.add(action)
                    final_status = AgentStatus.BLOCKED
                    break

                retry_counts[action] = count + 1
                continue

            break

    else:
        final_status = AgentStatus.BUDGET_EXCEEDED

    return AgentRunResult(
        status=final_status,
        steps_used=state.steps_used,
        blocked_actions=tuple(blocked_actions),
        history=tuple(state.action_history),
    )
