from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from temporalio.exceptions import WorkflowAlreadyStartedError

from app.agents.onboarding_graph import build_onboarding_graph
from app.agents.temporal_client import (
    TemporalToolWorkflowRunner,
    ToolWorkflowRunner,
)
from app.tools.base import BaseTool
from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
    ToolRisk,
    ToolRuntimeContext,
)
from app.tools.registry import ToolRegistry
from app.workflows.models import (
    ToolWorkflowResult,
    ToolWorkflowStatus,
)


class PartnerInput(BaseModel):
    partner_id: str


class ApprovePartnerTool(BaseTool[PartnerInput, str]):
    # Only its definition is used: the graph reads approval requirements from
    # the registry, and execution happens inside the (faked) workflow

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
        raise AssertionError("the graph must not execute tools directly")


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ApprovePartnerTool())
    return registry


def make_state(agent_run_id: str = "run-1") -> dict:
    return {
        "user_request": "Approve onboarding for partner-123",
        "agent_id": "onboarding-agent",
        "user_id": "user-1",
        "agent_run_id": agent_run_id,
    }


class FakeWorkflowRunner(ToolWorkflowRunner):

    def __init__(
        self,
        complete: bool = True,
        status: ToolWorkflowStatus = ToolWorkflowStatus.COMPLETED,
    ):
        self.complete = complete
        self.status = status
        self.requests = []
        self.result_requests: list[str] = []

    async def start(self, request):
        self.requests.append(request)
        return f"tool-{request.execution_id}"

    async def is_complete(self, workflow_id):
        return self.complete

    async def get_result(self, workflow_id):
        self.result_requests.append(workflow_id)
        request = self.requests[-1]
        return ToolWorkflowResult(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            status=self.status,
            output=(
                {"approved": True}
                if self.status == ToolWorkflowStatus.COMPLETED
                else None
            ),
            error=(
                None
                if self.status == ToolWorkflowStatus.COMPLETED
                else self.status.value
            ),
        )


@pytest.mark.asyncio
async def test_graph_executes_temporal_workflow():
    runner = FakeWorkflowRunner()
    graph = build_onboarding_graph(runner, make_registry())

    result = await graph.ainvoke(make_state())

    assert len(runner.requests) == 1
    assert runner.requests[0].tool_name == "approve_partner"
    assert result["workflow_status"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}
    assert result["temporal_workflow_id"] == (
        f"tool-{runner.requests[0].execution_id}"
    )
    assert result["final_response"] == "Action completed successfully."


@pytest.mark.asyncio
async def test_running_workflow_routes_to_waiting():
    # e.g. still blocked on a human approval
    runner = FakeWorkflowRunner(complete=False)
    graph = build_onboarding_graph(runner, make_registry())

    result = await graph.ainvoke(make_state())

    assert result["workflow_status"] == "RUNNING"
    assert result["final_response"] == "Action is awaiting completion."

    # The graph must not block on, or fetch, an unfinished result
    assert runner.result_requests == []
    assert "tool_output" not in result


@pytest.mark.asyncio
async def test_unsuccessful_workflow_routes_to_observe_result():
    runner = FakeWorkflowRunner(status=ToolWorkflowStatus.REJECTED)
    graph = build_onboarding_graph(runner, make_registry())

    result = await graph.ainvoke(make_state())

    assert result["workflow_status"] == "REJECTED"
    assert result["final_response"] == "Action ended with status: REJECTED"


def proposing(partner_id: str):
    # Stands in for the model's proposal so tests can vary the action
    async def propose_action(state):
        return {
            "proposed_tool": "approve_partner",
            "proposed_arguments": {"partner_id": partner_id},
        }

    return propose_action


async def execution_id_for(
    agent_run_id: str,
    partner_id: str = "partner-123",
) -> str:
    runner = FakeWorkflowRunner()
    graph = build_onboarding_graph(
        runner,
        make_registry(),
        action_proposer=proposing(partner_id),
    )

    await graph.ainvoke(make_state(agent_run_id))

    return runner.requests[0].execution_id


@pytest.mark.asyncio
async def test_same_run_and_action_produce_same_execution_id():
    runner = FakeWorkflowRunner()
    graph = build_onboarding_graph(runner, make_registry())

    # Two invocations with the same trusted run ID and the same action
    await graph.ainvoke(make_state("run-1"))
    await graph.ainvoke(make_state("run-1"))

    first, second = (request.execution_id for request in runner.requests)

    assert first == second


@pytest.mark.asyncio
async def test_different_run_produces_different_execution_id():
    assert await execution_id_for("run-1") != await execution_id_for("run-2")


@pytest.mark.asyncio
async def test_same_run_with_different_arguments_produces_different_execution_id():
    # One run may legitimately take two different actions; each needs its
    # own execution (and its own approval)
    first = await execution_id_for("run-1", partner_id="partner-123")
    second = await execution_id_for("run-1", partner_id="partner-456")

    assert first != second


@pytest.mark.asyncio
async def test_approval_requirement_comes_from_registry():
    runner = FakeWorkflowRunner()
    graph = build_onboarding_graph(runner, make_registry())

    await graph.ainvoke(make_state())

    # approve_partner's trusted definition requires HUMAN approval, so the
    # workflow must be told to wait for a reviewer
    assert runner.requests[0].requires_approval is True


class FakeWorkflowHandle:

    def __init__(self, result: ToolWorkflowResult):
        self._result = result

    async def describe(self):
        return SimpleNamespace(status=SimpleNamespace(name="COMPLETED"))

    async def result(self) -> ToolWorkflowResult:
        return self._result


class RecordingTemporalClient:
    # Behaves like Temporal for workflow IDs: starting an ID that already
    # exists raises WorkflowAlreadyStartedError

    def __init__(self):
        self.start_attempts: list[str] = []
        self.started: dict[str, ToolWorkflowResult] = {}

    async def start_workflow(self, workflow, request, *, id, task_queue):
        self.start_attempts.append(id)

        if id in self.started:
            raise WorkflowAlreadyStartedError(id, "ToolExecutionWorkflow")

        self.started[id] = ToolWorkflowResult(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            status=ToolWorkflowStatus.COMPLETED,
            output={"approved": True},
        )

    def get_workflow_handle(self, workflow_id, result_type=None):
        return FakeWorkflowHandle(self.started[workflow_id])


@pytest.mark.asyncio
async def test_reexecuted_action_targets_same_temporal_workflow_id():
    client = RecordingTemporalClient()
    runner = TemporalToolWorkflowRunner(client, task_queue="onboarding")
    graph = build_onboarding_graph(runner, make_registry())

    # LangGraph re-executing the action, e.g. resuming after a crash between
    # starting the workflow and checkpointing its result
    first_result = await graph.ainvoke(make_state("run-1"))
    second_result = await graph.ainvoke(make_state("run-1"))

    first, second = client.start_attempts

    assert first == second
    assert first.startswith("tool-")

    # One workflow exists; the re-execution reconnected to it rather than
    # starting a second one that could run the tool twice
    assert list(client.started) == [first]
    assert second_result["temporal_workflow_id"] == first
    assert second_result["workflow_status"] == first_result["workflow_status"]


@pytest.mark.asyncio
async def test_graph_runs_with_checkpointer():
    runner = FakeWorkflowRunner()
    graph = build_onboarding_graph(
        runner,
        make_registry(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "run-1"}}

    result = await graph.ainvoke(make_state("run-1"), config=config)

    assert result["workflow_status"] == "COMPLETED"

    # State is persisted per thread, so a later invocation can resume it
    saved = await graph.aget_state(config)
    assert saved.values["temporal_workflow_id"] == result["temporal_workflow_id"]
