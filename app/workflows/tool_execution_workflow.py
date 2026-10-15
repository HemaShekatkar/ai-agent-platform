from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
)

# Workflow code runs in Temporal's deterministic sandbox, which re-imports
# modules on every run; pass these through since they are side-effect free
with workflow.unsafe.imports_passed_through():
    from app.tools.approval import (
        ApprovalDecision,
        ApprovalStatus,
    )
    from app.tools.fingerprint import action_fingerprint
    from app.workflows.activities import (
        ExecuteToolActivityRequest,
        ToolExecutionActivities,
    )
    from app.workflows.models import (
        ToolWorkflowRequest,
        ToolWorkflowResult,
        ToolWorkflowStatus,
    )


@workflow.defn
class ToolExecutionWorkflow:

    def __init__(self):
        self._approval: ApprovalDecision | None = None
        self._status = ToolWorkflowStatus.CREATED

    @workflow.signal
    async def submit_approval(
        self,
        decision: ApprovalDecision,
    ) -> None:
        self._approval = decision

    @workflow.query
    def status(self) -> ToolWorkflowStatus:
        # Lets callers (and tests) see where a running workflow is, e.g.
        # whether it is blocked waiting for a human decision
        return self._status

    @workflow.run
    async def run(
        self,
        request: ToolWorkflowRequest,
    ) -> ToolWorkflowResult:
        result = await self._run(request)
        self._status = result.status
        return result

    async def _run(
        self,
        request: ToolWorkflowRequest,
    ) -> ToolWorkflowResult:
        approval = None

        if request.requires_approval:
            self._status = ToolWorkflowStatus.WAITING_FOR_APPROVAL

            # Don't reset self._approval here: a signal sent right after the
            # workflow starts can be handled before run() reaches this line
            await workflow.wait_condition(
                lambda: self._approval is not None
            )

            approval = self._approval

            if approval.execution_id != request.execution_id:
                return ToolWorkflowResult(
                    execution_id=request.execution_id,
                    tool_name=request.tool_name,
                    status=ToolWorkflowStatus.FAILED,
                    error="Approval execution ID mismatch",
                )

            # The request is the workflow's immutable action; an approval
            # granted for different arguments or a different tool must not
            # schedule this one. Deterministic (sorted-key SHA-256), so safe
            # to compute inside workflow code. The executor re-checks it.
            expected_fingerprint = action_fingerprint(
                request.tool_name,
                request.arguments,
            )

            if approval.action_fingerprint != expected_fingerprint:
                return ToolWorkflowResult(
                    execution_id=request.execution_id,
                    tool_name=request.tool_name,
                    status=ToolWorkflowStatus.FAILED,
                    error="Approval fingerprint mismatch",
                )

            if approval.status != ApprovalStatus.APPROVED:
                return ToolWorkflowResult(
                    execution_id=request.execution_id,
                    tool_name=request.tool_name,
                    status=ToolWorkflowStatus.REJECTED,
                    error="Approval rejected",
                )

        self._status = ToolWorkflowStatus.EXECUTING

        activity_request = ExecuteToolActivityRequest(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            agent_id=request.agent_id,
            user_id=request.user_id,
            # The executor re-checks the approval against this execution
            approval=approval,
        )

        try:
            result = await workflow.execute_activity(
                ToolExecutionActivities.execute_tool,
                activity_request,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    backoff_coefficient=2.0,
                    maximum_interval=timedelta(seconds=10),
                    maximum_attempts=3,
                ),
            )

        except ActivityError as exc:
            cause = exc.cause

            if isinstance(cause, ApplicationError):
                if cause.type in {
                    "ToolAuthorizationError",
                    "ToolApprovalRequiredError",
                    "ToolApprovalRejectedError",
                    "ToolInputValidationError",
                    "ToolNotFoundError",
                }:
                    return ToolWorkflowResult(
                        execution_id=request.execution_id,
                        tool_name=request.tool_name,
                        status=ToolWorkflowStatus.FAILED,
                        error=cause.type,
                    )

                if cause.type == "PermanentToolError":
                    return ToolWorkflowResult(
                        execution_id=request.execution_id,
                        tool_name=request.tool_name,
                        status=ToolWorkflowStatus.HUMAN_REVIEW,
                        error=cause.type,
                    )

            return ToolWorkflowResult(
                execution_id=request.execution_id,
                tool_name=request.tool_name,
                status=ToolWorkflowStatus.FAILED,
                error="Activity execution failed",
            )

        return ToolWorkflowResult(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            status=ToolWorkflowStatus.COMPLETED,
            output=result.output,
        )
