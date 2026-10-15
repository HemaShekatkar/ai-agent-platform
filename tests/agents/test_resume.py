import pytest

from app.agents.resume import resume_execution
from app.agents.state import AgentExecutionPhase
from app.agents.temporal_client import ToolWorkflowRunner
from app.workflows.models import ToolWorkflowResult, ToolWorkflowStatus


ORIGINAL_EXECUTION_ID = "exec-123"
ORIGINAL_WORKFLOW_ID = f"tool-{ORIGINAL_EXECUTION_ID}"


class RecordingWorkflowRunner(ToolWorkflowRunner):
    # Records every Temporal operation so tests can prove what resumption
    # did and, more importantly, did not do

    def __init__(self, complete: bool, result: ToolWorkflowResult | None = None):
        self.complete = complete
        self.result = result
        self.start_calls = []
        self.is_complete_calls: list[str] = []
        self.get_result_calls: list[str] = []

    async def start(self, request):
        self.start_calls.append(request)
        return f"tool-{request.execution_id}"

    async def is_complete(self, workflow_id):
        self.is_complete_calls.append(workflow_id)
        return self.complete

    async def get_result(self, workflow_id):
        self.get_result_calls.append(workflow_id)
        return self.result


def make_checkpointed_state(**overrides) -> dict:
    # State as checkpointed after the workflow was started
    state = {
        "agent_run_id": "run-1",
        "agent_id": "onboarding-agent",
        "user_id": "user-1",
        "proposed_tool": "approve_partner",
        "proposed_arguments": {"partner_id": "partner-123"},
        "execution_id": ORIGINAL_EXECUTION_ID,
        "requires_approval": True,
        "temporal_workflow_id": ORIGINAL_WORKFLOW_ID,
        "workflow_status": "RUNNING",
        "execution_phase": AgentExecutionPhase.WAITING.value,
    }
    state.update(overrides)
    return state


def completed_result() -> ToolWorkflowResult:
    return ToolWorkflowResult(
        execution_id=ORIGINAL_EXECUTION_ID,
        tool_name="approve_partner",
        status=ToolWorkflowStatus.COMPLETED,
        output={"approved": True},
    )


@pytest.mark.asyncio
async def test_resume_pending_workflow():
    runner = RecordingWorkflowRunner(complete=False)

    result = await resume_execution(make_checkpointed_state(), runner)

    assert result["execution_phase"] == "WAITING"
    assert result["workflow_status"] == "RUNNING"

    # Same wording the graph uses for a workflow that is not finished
    assert result["final_response"] == "Action is awaiting completion."

    # Resumption only polls the existing workflow
    assert runner.is_complete_calls == [ORIGINAL_WORKFLOW_ID]
    assert runner.start_calls == []
    assert runner.get_result_calls == []

    assert result["execution_id"] == ORIGINAL_EXECUTION_ID
    assert result["temporal_workflow_id"] == ORIGINAL_WORKFLOW_ID
    assert "tool_output" not in result


@pytest.mark.asyncio
async def test_resume_completed_workflow():
    runner = RecordingWorkflowRunner(complete=True, result=completed_result())
    original_execution_id = ORIGINAL_EXECUTION_ID
    original_workflow_id = ORIGINAL_WORKFLOW_ID

    result = await resume_execution(make_checkpointed_state(), runner)

    assert result["execution_phase"] == "COMPLETED"
    assert result["workflow_status"] == "COMPLETED"
    assert result["tool_output"] == {"approved": True}
    assert result["final_response"] == "Action completed successfully."
    assert result["execution_id"] == original_execution_id
    assert result["temporal_workflow_id"] == original_workflow_id

    # The result came from the existing workflow; nothing was re-started
    assert runner.start_calls == []
    assert runner.get_result_calls == [original_workflow_id]


@pytest.mark.asyncio
async def test_resume_completed_agent_is_idempotent():
    runner = RecordingWorkflowRunner(complete=True, result=completed_result())
    completed_state = make_checkpointed_state(
        execution_phase=AgentExecutionPhase.COMPLETED.value,
        workflow_status="COMPLETED",
        tool_output={"approved": True},
        final_response="Action completed successfully.",
    )

    result = await resume_execution(completed_state, runner)

    # Returned as-is: a finished agent is never re-polled or re-executed
    assert result == completed_state
    assert runner.start_calls == []
    assert runner.is_complete_calls == []
    assert runner.get_result_calls == []


@pytest.mark.asyncio
async def test_resume_unsuccessful_workflow_reports_its_status():
    runner = RecordingWorkflowRunner(
        complete=True,
        result=ToolWorkflowResult(
            execution_id=ORIGINAL_EXECUTION_ID,
            tool_name="approve_partner",
            status=ToolWorkflowStatus.REJECTED,
            error="Approval rejected",
        ),
    )

    result = await resume_execution(make_checkpointed_state(), runner)

    assert result["execution_phase"] == "FAILED"
    assert result["final_response"] == "Action ended with status: REJECTED"
    assert runner.start_calls == []
