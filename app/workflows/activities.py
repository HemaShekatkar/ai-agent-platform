from dataclasses import dataclass
from typing import Any, Protocol

from temporalio import activity
from temporalio.exceptions import ApplicationError

from app.tools.approval import ApprovalDecision
from app.tools.exceptions import (
    TransientToolError,
    PermanentToolError,
    ToolAuthorizationError,
    ToolInputValidationError,
    ToolApprovalRequiredError,
    ToolApprovalRejectedError,
    ToolNotFoundError,
)
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall, ToolExecutionContext, ToolResult


@dataclass(frozen=True)
class ExecuteToolActivityRequest:
    execution_id: str
    tool_name: str
    arguments: dict[str, Any]

    agent_id: str
    user_id: str
    approval: ApprovalDecision | None = None


class IdentityResolver(Protocol):

    def resolve(
        self,
        agent_id: str,
        user_id: str,
    ) -> ToolExecutionContext:
        """Return the agent's and user's permissions as they are right now."""
        ...


class ToolExecutionActivities:

    def __init__(
        self,
        identity_resolver: IdentityResolver,
        executor: ToolExecutor,
    ) -> None:
        self.identity_resolver = identity_resolver
        self.executor = executor

    @activity.defn
    def execute_tool(
        self,
        request: ExecuteToolActivityRequest,
    ) -> ToolResult:

        # Resolve permissions when the tool runs, not when the workflow
        # started: an approval wait can last hours, and a permission revoked
        # in the meantime must block execution. The request carries only IDs.
        context = self.identity_resolver.resolve(
            agent_id=request.agent_id,
            user_id=request.user_id,
        )

        call = ToolCall(
            tool_name=request.tool_name,
            arguments=request.arguments,
        )

        try:
            return self.executor.execute(
                call=call,
                execution_id=request.execution_id,
                context=context,
                approval=request.approval,
            )

        except TransientToolError as exc:
            raise ApplicationError(
                str(exc),
                type="TransientToolError",
                non_retryable=False,
            ) from exc

        except (
            PermanentToolError,
            ToolNotFoundError,
            ToolAuthorizationError,
            ToolInputValidationError,
            ToolApprovalRequiredError,
            ToolApprovalRejectedError,
        ) as exc:
            raise ApplicationError(
                str(exc),
                type=type(exc).__name__,
                non_retryable=True,
            ) from exc
