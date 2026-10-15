from app.agents.messages import (
    ACTION_AWAITING_COMPLETION,
    ACTION_COMPLETED,
    action_ended,
)
from app.agents.state import AgentExecutionPhase, OnboardingAgentState
from app.agents.temporal_client import ToolWorkflowRunner


async def resume_execution(
    state: OnboardingAgentState,
    workflow_runner: ToolWorkflowRunner,
) -> OnboardingAgentState:

    if state.get("execution_phase") == AgentExecutionPhase.COMPLETED.value:
        return state

    workflow_id = state["temporal_workflow_id"]

    if not await workflow_runner.is_complete(workflow_id):
        return {
            **state,
            "execution_phase": AgentExecutionPhase.WAITING.value,
            "workflow_status": "RUNNING",
            "final_response": ACTION_AWAITING_COMPLETION,
        }

    result = await workflow_runner.get_result(workflow_id)

    phase = (
        AgentExecutionPhase.COMPLETED
        if result.status.value == "COMPLETED"
        else AgentExecutionPhase.FAILED
    )

    return {
        **state,
        "execution_phase": phase.value,
        "workflow_status": result.status.value,
        "tool_output": result.output,
        "error": result.error,
        "final_response": (
            ACTION_COMPLETED
            if phase == AgentExecutionPhase.COMPLETED
            else action_ended(result.status.value)
        ),
    }
