from app.agents.resume import resume_execution
from app.agents.state import AgentExecutionPhase
from app.agents.temporal_client import ToolWorkflowRunner


async def resume_agent(
    agent_run_id: str,
    graph,
    workflow_runner: ToolWorkflowRunner,
):
    config = {
        "configurable": {
            "thread_id": agent_run_id,
        }
    }

    # Load the existing LangGraph checkpoint
    snapshot = await graph.aget_state(config)

    if not snapshot.values:
        raise ValueError(
            f"No checkpoint found for agent run: {agent_run_id}"
        )

    state = snapshot.values

    # Already completed: do not call Temporal again
    if state.get("execution_phase") == AgentExecutionPhase.COMPLETED.value:
        return state

    # Observe existing Temporal workflow
    updated_state = await resume_execution(
        state=state,
        workflow_runner=workflow_runner,
    )

    # Persist updated agent state
    await graph.aupdate_state(
        config,
        updated_state,
    )

    return updated_state
