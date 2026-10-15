from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agents.onboarding_graph import build_onboarding_graph
from app.agents.recovery import resume_agent
from app.agents.temporal_client import ToolWorkflowRunner
from app.tools.base import BaseTool
from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
    ToolRisk,
    ToolRuntimeContext,
)
from app.tools.registry import ToolRegistry
from app.workflows.models import ToolWorkflowResult, ToolWorkflowStatus


AGENT_RUN_ID = "agent-run-123"
CONFIG = {
    "configurable": {
        "thread_id": AGENT_RUN_ID,
    }
}


class PartnerInput(BaseModel):
    partner_id: str


class ApprovePartnerTool(BaseTool[PartnerInput, str]):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="approve_partner",
            description="Approve a partner for onboarding.",
            input_schema=PartnerInput,
            risk=ToolRisk.HIGH,
            required_permissions={"partner.approve"},
            approval_requirement=ApprovalRequirement.HUMAN,
        )

    def execute(self, input: PartnerInput, runtime: ToolRuntimeContext) -> str:
        raise AssertionError("tools run inside the workflow, never the graph")


def completed_result(workflow_id: str) -> ToolWorkflowResult:
    return ToolWorkflowResult(
        execution_id=workflow_id.removeprefix("tool-"),
        tool_name="approve_partner",
        status=ToolWorkflowStatus.COMPLETED,
        output={"approved": True},
    )


@pytest.fixture
def fake_runner():
    runner = MagicMock(spec=ToolWorkflowRunner)

    # Mirrors TemporalToolWorkflowRunner: the workflow ID is derived from
    # the execution ID
    runner.start = AsyncMock(
        side_effect=lambda request: f"tool-{request.execution_id}"
    )
    runner.is_complete = AsyncMock(return_value=False)
    runner.get_result = AsyncMock(side_effect=completed_result)

    return runner


@pytest_asyncio.fixture
async def graph_with_pending_checkpoint(fake_runner):
    registry = ToolRegistry()
    registry.register(ApprovePartnerTool())

    graph = build_onboarding_graph(
        workflow_runner=fake_runner,
        registry=registry,
        checkpointer=InMemorySaver(),
    )

    # Run to the WAITING phase so a real checkpoint exists for this run
    await graph.ainvoke(
        {
            "agent_run_id": AGENT_RUN_ID,
            "agent_id": "onboarding-agent",
            "user_id": "user-1",
        },
        config=CONFIG,
    )

    snapshot = await graph.aget_state(CONFIG)
    assert snapshot.values["execution_phase"] == "WAITING"

    # Starting the workflow is covered by the graph tests; reset so each
    # test asserts only on what recovery itself does
    fake_runner.start.reset_mock()
    fake_runner.is_complete.reset_mock()
    fake_runner.get_result.reset_mock()

    return graph


@pytest.mark.asyncio
async def test_resume_agent_loads_checkpoint(
    graph_with_pending_checkpoint,
    fake_runner,
):
    graph = graph_with_pending_checkpoint
    checkpointed = (await graph.aget_state(CONFIG)).values

    result = await resume_agent(
        agent_run_id=AGENT_RUN_ID,
        graph=graph,
        workflow_runner=fake_runner,
    )

    # Recovery works from this run's checkpoint and polls its workflow
    assert result["agent_run_id"] == AGENT_RUN_ID
    assert result["execution_id"] == checkpointed["execution_id"]
    assert result["temporal_workflow_id"] == checkpointed["temporal_workflow_id"]
    fake_runner.is_complete.assert_awaited_once_with(
        checkpointed["temporal_workflow_id"]
    )


@pytest.mark.asyncio
async def test_resume_agent_updates_checkpoint(
    graph_with_pending_checkpoint,
    fake_runner,
):
    graph = graph_with_pending_checkpoint
    fake_runner.is_complete = AsyncMock(return_value=True)

    result = await resume_agent(
        agent_run_id=AGENT_RUN_ID,
        graph=graph,
        workflow_runner=fake_runner,
    )

    assert result["execution_phase"] == "COMPLETED"
    assert result["workflow_status"] == "COMPLETED"

    config = {
        "configurable": {
            "thread_id": "agent-run-123",
        }
    }

    snapshot = await graph.aget_state(config)

    # Persisted, not just returned: a later process sees the outcome
    assert snapshot.values["execution_phase"] == "COMPLETED"
    assert snapshot.values["workflow_status"] == "COMPLETED"
    assert snapshot.values["tool_output"] == {"approved": True}

    fake_runner.start.assert_not_called()


@pytest.mark.asyncio
async def test_resume_agent_does_not_restart_workflow(
    graph_with_pending_checkpoint,
    fake_runner,
):
    graph = graph_with_pending_checkpoint

    # Still pending: recovery must poll the existing workflow, not start one
    await resume_agent(
        agent_run_id=AGENT_RUN_ID,
        graph=graph,
        workflow_runner=fake_runner,
    )

    # And once complete, it must collect the result, still without starting
    fake_runner.is_complete = AsyncMock(return_value=True)
    await resume_agent(
        agent_run_id=AGENT_RUN_ID,
        graph=graph,
        workflow_runner=fake_runner,
    )

    fake_runner.start.assert_not_called()
    fake_runner.get_result.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_agent_missing_checkpoint(
    graph_with_pending_checkpoint,
    fake_runner,
):
    with pytest.raises(ValueError, match="agent-run-unknown"):
        await resume_agent(
            agent_run_id="agent-run-unknown",
            graph=graph_with_pending_checkpoint,
            workflow_runner=fake_runner,
        )

    # Fails before touching Temporal rather than guessing at a workflow
    fake_runner.start.assert_not_called()
    fake_runner.is_complete.assert_not_called()
    fake_runner.get_result.assert_not_called()


@pytest.mark.asyncio
async def test_resume_agent_completed_is_idempotent(
    graph_with_pending_checkpoint,
    fake_runner,
):
    graph = graph_with_pending_checkpoint
    fake_runner.is_complete = AsyncMock(return_value=True)

    first = await resume_agent(
        agent_run_id=AGENT_RUN_ID,
        graph=graph,
        workflow_runner=fake_runner,
    )

    fake_runner.is_complete.reset_mock()
    fake_runner.get_result.reset_mock()

    second = await resume_agent(
        agent_run_id=AGENT_RUN_ID,
        graph=graph,
        workflow_runner=fake_runner,
    )

    # Already COMPLETED in the checkpoint: returned without querying Temporal
    assert second["execution_phase"] == "COMPLETED"
    assert second["tool_output"] == first["tool_output"]
    fake_runner.start.assert_not_called()
    fake_runner.is_complete.assert_not_called()
    fake_runner.get_result.assert_not_called()
