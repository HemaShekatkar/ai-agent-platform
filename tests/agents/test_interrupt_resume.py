from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from app.agents.onboarding_graph import build_onboarding_graph
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
CONFIG = {"configurable": {"thread_id": AGENT_RUN_ID}}


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


class CountingProposer:
    # Stands in for the model: proposals are expensive and non-deterministic,
    # so a resume must never ask for a new one

    def __init__(self):
        self.calls = 0

    async def __call__(self, state):
        self.calls += 1
        return {
            "proposed_tool": "approve_partner",
            "proposed_arguments": {"partner_id": "partner-123"},
        }


def make_runner(*completion_checks: bool) -> MagicMock:
    # is_complete() answers with the given sequence, one value per check
    runner = MagicMock(spec=ToolWorkflowRunner)
    runner.start = AsyncMock(
        side_effect=lambda request: f"tool-{request.execution_id}"
    )
    runner.is_complete = AsyncMock(side_effect=list(completion_checks))
    runner.get_result = AsyncMock(
        side_effect=lambda workflow_id: ToolWorkflowResult(
            execution_id=workflow_id.removeprefix("tool-"),
            tool_name="approve_partner",
            status=ToolWorkflowStatus.COMPLETED,
            output={"approved": True},
        )
    )
    return runner


def make_graph(runner, proposer=None):
    registry = ToolRegistry()
    registry.register(ApprovePartnerTool())

    return build_onboarding_graph(
        workflow_runner=runner,
        registry=registry,
        action_proposer=proposer,
        checkpointer=InMemorySaver(),
    )


async def start_graph(graph):
    return await graph.ainvoke(
        {
            "agent_run_id": AGENT_RUN_ID,
            "agent_id": "onboarding-agent",
            "user_id": "user-1",
        },
        config=CONFIG,
    )


async def resume(graph, workflow_id):
    return await graph.ainvoke(
        Command(resume={"workflow_id": workflow_id}),
        config=CONFIG,
    )


async def paused_workflow_id(graph) -> str:
    snapshot = await graph.aget_state(CONFIG)
    return snapshot.values["temporal_workflow_id"]


@pytest.mark.asyncio
async def test_graph_interrupts_while_temporal_is_running():
    graph = make_graph(make_runner(False))

    result = await start_graph(graph)

    # Paused, not finished: the run yields an interrupt instead of a result
    (pending,) = result["__interrupt__"]
    workflow_id = await paused_workflow_id(graph)

    assert pending.value == {
        "execution_id": workflow_id.removeprefix("tool-"),
        "workflow_id": workflow_id,
        "reason": "WAITING_FOR_TEMPORAL",
    }

    snapshot = await graph.aget_state(CONFIG)
    assert snapshot.next == ("wait_for_execution",)
    assert snapshot.values["execution_phase"] == "WAITING"
    assert "tool_output" not in snapshot.values


@pytest.mark.asyncio
async def test_resuming_checks_temporal_again():
    runner = make_runner(False, True)
    graph = make_graph(runner)
    await start_graph(graph)
    workflow_id = await paused_workflow_id(graph)

    result = await resume(graph, workflow_id)

    # One check on the first run, one more after the resume
    assert runner.is_complete.await_count == 2
    runner.is_complete.assert_awaited_with(workflow_id)
    assert result["execution_phase"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}


@pytest.mark.asyncio
async def test_resuming_does_not_propose_action_again():
    proposer = CountingProposer()
    graph = make_graph(make_runner(False, True), proposer)
    await start_graph(graph)

    await resume(graph, await paused_workflow_id(graph))

    assert proposer.calls == 1


@pytest.mark.asyncio
async def test_resuming_does_not_restart_temporal_workflow():
    runner = make_runner(False, True)
    graph = make_graph(runner)
    await start_graph(graph)

    await resume(graph, await paused_workflow_id(graph))

    # Started once on the first run; the resume re-entered at the pause
    runner.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_resumes_until_temporal_completes():
    # Pending on the first run and two resumes, complete on the third resume
    runner = make_runner(False, False, False, True)
    proposer = CountingProposer()
    graph = make_graph(runner, proposer)
    await start_graph(graph)
    workflow_id = await paused_workflow_id(graph)

    for _ in range(2):
        result = await resume(graph, workflow_id)

        assert "__interrupt__" in result
        snapshot = await graph.aget_state(CONFIG)
        assert snapshot.next == ("wait_for_execution",)
        assert snapshot.values["execution_phase"] == "WAITING"

    result = await resume(graph, workflow_id)

    assert "__interrupt__" not in result
    assert result["execution_phase"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}
    assert (await graph.aget_state(CONFIG)).next == ()

    assert runner.is_complete.await_count == 4
    runner.get_result.assert_awaited_once_with(workflow_id)
    runner.start.assert_awaited_once()
    assert proposer.calls == 1


@pytest.mark.asyncio
async def test_resume_with_wrong_workflow_id_is_rejected():
    runner = make_runner(False, True)
    graph = make_graph(runner)
    await start_graph(graph)
    workflow_id = await paused_workflow_id(graph)

    result = await resume(graph, "tool-some-other-workflow")

    # Rejected: the run pauses again and reports why, without advancing
    (pending,) = result["__interrupt__"]
    assert pending.value["rejected_resume"] == "Workflow identity mismatch"
    assert pending.value["workflow_id"] == workflow_id

    # Temporal was not consulted on behalf of the wrong workflow
    assert runner.is_complete.await_count == 1
    runner.get_result.assert_not_called()
    snapshot = await graph.aget_state(CONFIG)
    assert snapshot.next == ("wait_for_execution",)
    assert "tool_output" not in snapshot.values


@pytest.mark.asyncio
async def test_rejected_resume_does_not_wedge_the_run():
    # LangGraph persists resume values; a rejected payload must not be
    # replayed to block every later, correct resume
    runner = make_runner(False, True)
    graph = make_graph(runner)
    await start_graph(graph)
    workflow_id = await paused_workflow_id(graph)

    await resume(graph, "tool-some-other-workflow")
    await resume(graph, "tool-yet-another-workflow")
    result = await resume(graph, workflow_id)

    assert result["execution_phase"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}
    runner.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_with_malformed_payload_is_rejected():
    runner = make_runner(False, True)
    graph = make_graph(runner)
    await start_graph(graph)

    result = await graph.ainvoke(Command(resume="not-a-dict"), config=CONFIG)

    (pending,) = result["__interrupt__"]
    assert pending.value["rejected_resume"] == "Invalid resume payload"
    assert runner.is_complete.await_count == 1
