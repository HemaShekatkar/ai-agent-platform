import asyncio
from contextlib import asynccontextmanager

import pytest
from temporalio import activity
from temporalio.client import WorkflowHandle
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.tools.approval import ApprovalDecision, ApprovalStatus
from app.tools.fingerprint import action_fingerprint
from app.tools.models import ToolResult
from app.workflows.activities import ExecuteToolActivityRequest
from app.workflows.models import (
    ToolWorkflowRequest,
    ToolWorkflowResult,
    ToolWorkflowStatus,
)
from app.workflows.tool_execution_workflow import ToolExecutionWorkflow


TASK_QUEUE = "test-tool-execution"


def make_workflow_request(
    requires_approval: bool = False,
) -> ToolWorkflowRequest:
    return ToolWorkflowRequest(
        execution_id="exec-123",
        tool_name="approve_partner",
        arguments={"partner_id": "partner-123"},
        agent_id="onboarding-agent",
        user_id="user-1",
        requires_approval=requires_approval,
    )


async def run_workflow(fake_activity) -> ToolWorkflowResult:
    # The fake is registered under the real activity's name, so the workflow
    # dispatches to it exactly as it would to ToolExecutionActivities
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ToolExecutionWorkflow],
            activities=[fake_activity],
        ):
            return await env.client.execute_workflow(
                ToolExecutionWorkflow.run,
                make_workflow_request(),
                id="workflow-exec-123",
                task_queue=TASK_QUEUE,
            )


@pytest.mark.asyncio
async def test_transient_failure_retries_and_succeeds():
    attempts = []

    @activity.defn(name="execute_tool")
    async def fake_execute_tool(
        request: ExecuteToolActivityRequest,
    ) -> ToolResult:
        attempts.append(request.execution_id)

        if len(attempts) < 3:
            raise ApplicationError(
                "Temporary service outage",
                type="TransientToolError",
            )

        return ToolResult(
            tool_name=request.tool_name,
            success=True,
            output={"approved": True},
        )

    result = await run_workflow(fake_execute_tool)

    assert result.status == ToolWorkflowStatus.COMPLETED
    assert result.output == {"approved": True}

    # Two transient failures, then success on the third and final attempt;
    # every retry carries the same execution ID
    assert attempts == ["exec-123", "exec-123", "exec-123"]


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried_and_goes_to_human_review():
    attempts = []

    @activity.defn(name="execute_tool")
    async def fake_execute_tool(
        request: ExecuteToolActivityRequest,
    ) -> ToolResult:
        attempts.append(request.execution_id)

        raise ApplicationError(
            "Invalid partner state",
            type="PermanentToolError",
            non_retryable=True,
        )

    result = await run_workflow(fake_execute_tool)

    # Non-retryable: one attempt despite the policy allowing three
    assert attempts == ["exec-123"]
    assert result.status == ToolWorkflowStatus.HUMAN_REVIEW
    assert result.error == "PermanentToolError"


# Approval workflow: the workflow pauses for a human decision delivered by
# signal, and must not touch the tool until an approval arrives


class RecordingActivity:
    # Fake activity that records every call and the approval it carried

    def __init__(self):
        self.calls: list[ExecuteToolActivityRequest] = []

        @activity.defn(name="execute_tool")
        async def execute_tool(
            request: ExecuteToolActivityRequest,
        ) -> ToolResult:
            self.calls.append(request)
            return ToolResult(
                tool_name=request.tool_name,
                success=True,
                output={"approved": True},
            )

        self.execute_tool = execute_tool


@asynccontextmanager
async def started_approval_workflow(fake: RecordingActivity):
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ToolExecutionWorkflow],
            activities=[fake.execute_tool],
        ):
            # start_workflow, not execute_workflow, so the test can signal it
            yield await env.client.start_workflow(
                ToolExecutionWorkflow.run,
                make_workflow_request(requires_approval=True),
                id="workflow-exec-123",
                task_queue=TASK_QUEUE,
            )


async def wait_until_waiting_for_approval(
    handle: WorkflowHandle,
    timeout_seconds: float = 10,
) -> None:
    # Synchronize on the workflow's own reported state rather than sleeping
    # for a guessed duration; the timeout only bounds a broken workflow
    async with asyncio.timeout(timeout_seconds):
        while (
            await handle.query(ToolExecutionWorkflow.status)
            != ToolWorkflowStatus.WAITING_FOR_APPROVAL
        ):
            await asyncio.sleep(0.05)


# Fingerprint of the action every approval in this file is granted for
APPROVED_ACTION = action_fingerprint(
    "approve_partner",
    {"partner_id": "partner-123"},
)


def decision(
    status: ApprovalStatus,
    execution_id: str = "exec-123",
    fingerprint: str = APPROVED_ACTION,
) -> ApprovalDecision:
    return ApprovalDecision(
        execution_id=execution_id,
        action_fingerprint=fingerprint,
        status=status,
        approved_by="reviewer-1",
    )


@pytest.mark.asyncio
async def test_workflow_waits_for_approval():
    fake = RecordingActivity()

    async with started_approval_workflow(fake) as handle:
        await wait_until_waiting_for_approval(handle)

        # Blocked on the human decision: the tool has not been touched
        assert fake.calls == []

        description = await handle.describe()
        assert description.status.name == "RUNNING"

        # Release the workflow so the test environment shuts down cleanly
        await handle.signal(
            ToolExecutionWorkflow.submit_approval,
            decision(ApprovalStatus.REJECTED),
        )
        await handle.result()


@pytest.mark.asyncio
async def test_workflow_executes_after_approval():
    fake = RecordingActivity()

    async with started_approval_workflow(fake) as handle:
        await wait_until_waiting_for_approval(handle)
        assert fake.calls == []

        await handle.signal(
            ToolExecutionWorkflow.submit_approval,
            decision(ApprovalStatus.APPROVED),
        )

        result = await handle.result()

        assert result.status == ToolWorkflowStatus.COMPLETED
        assert result.output == {"approved": True}
        assert await handle.query(ToolExecutionWorkflow.status) == (
            ToolWorkflowStatus.COMPLETED
        )

    # Exactly one execution, carrying the approval for this execution so the
    # executor can re-verify it at the boundary
    assert len(fake.calls) == 1
    assert fake.calls[0].approval is not None
    assert fake.calls[0].approval.status == ApprovalStatus.APPROVED
    assert fake.calls[0].approval.execution_id == "exec-123"


@pytest.mark.asyncio
async def test_workflow_rejects_without_execution():
    fake = RecordingActivity()

    async with started_approval_workflow(fake) as handle:
        await wait_until_waiting_for_approval(handle)

        await handle.signal(
            ToolExecutionWorkflow.submit_approval,
            decision(ApprovalStatus.REJECTED),
        )

        result = await handle.result()

        assert result.status == ToolWorkflowStatus.REJECTED
        assert result.error == "Approval rejected"

    assert fake.calls == []


# Approval binding: the workflow schedules the activity only for an approval
# granted for this execution and this exact action


async def run_with_decision(approval: ApprovalDecision):
    fake = RecordingActivity()

    async with started_approval_workflow(fake) as handle:
        await wait_until_waiting_for_approval(handle)
        await handle.signal(ToolExecutionWorkflow.submit_approval, approval)
        result = await handle.result()

    return result, fake


@pytest.mark.asyncio
async def test_approval_with_wrong_fingerprint_blocks_execution():
    # Right execution, but the reviewer approved a different partner
    other_action = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-456"},
    )

    result, fake = await run_with_decision(
        decision(ApprovalStatus.APPROVED, fingerprint=other_action)
    )

    assert result.status == ToolWorkflowStatus.FAILED
    assert result.error == "Approval fingerprint mismatch"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_approval_with_matching_execution_and_fingerprint_executes():
    result, fake = await run_with_decision(
        decision(ApprovalStatus.APPROVED)
    )

    assert result.status == ToolWorkflowStatus.COMPLETED
    assert len(fake.calls) == 1

    # The approval reaches the activity so the executor can re-verify it
    assert fake.calls[0].approval.action_fingerprint == APPROVED_ACTION


@pytest.mark.asyncio
async def test_approval_for_another_execution_blocks_execution():
    result, fake = await run_with_decision(
        decision(ApprovalStatus.APPROVED, execution_id="exec-999")
    )

    assert result.status == ToolWorkflowStatus.FAILED
    assert result.error == "Approval execution ID mismatch"
    assert fake.calls == []
