from pydantic import ValidationError

from app.tools.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalStatus,
)
from app.tools.exceptions import (
    ToolApprovalRejectedError,
    ToolApprovalRequiredError,
    ToolAuthorizationError,
    ToolInputValidationError,
)
from app.tools.fingerprint import action_fingerprint
from app.tools.models import (
    ToolCall,
    ToolExecutionContext,
    ToolRuntimeContext,
    ToolResult,
)
from app.tools.policy import ToolAuthorizer
from app.tools.registry import ToolRegistry
from app.tools.fingerprint import action_fingerprint


class ToolExecutor:

    def __init__(
        self,
        registry: ToolRegistry,
        authorizer: ToolAuthorizer,
        approval_policy: ApprovalPolicy,
    ) -> None:
        self.registry = registry
        self.authorizer = authorizer
        self.approval_policy = approval_policy

    def execute(
        self,
        call: ToolCall,
        execution_id: str,
        context: ToolExecutionContext,
        approval: ApprovalDecision | None = None,
    ) -> ToolResult:

        tool = self.registry.get(call.tool_name)
        definition = tool.definition

        authorization = self.authorizer.authorize(
            definition,
            context,
        )

        if not authorization.allowed:
            raise ToolAuthorizationError(
                f"Not authorized to execute '{definition.name}'."
            )

        try:
            validated_input = definition.input_schema.model_validate(
                call.arguments
            )
        except ValidationError as exc:
            raise ToolInputValidationError(
                f"Invalid input for '{definition.name}'."
            ) from exc

        validated_arguments = validated_input.model_dump(mode="json")

        expected_fingerprint = action_fingerprint(
            definition.name,
            validated_arguments,
        )

        approval_status = self.approval_policy.evaluate(
            execution_id=execution_id,
            tool=definition,
            approval=approval,
            expected_fingerprint=expected_fingerprint,
        )

        if approval_status == ApprovalStatus.PENDING:
            raise ToolApprovalRequiredError(
                f"Approval required for '{definition.name}'."
            )

        if approval_status == ApprovalStatus.REJECTED:
            raise ToolApprovalRejectedError(
                f"Approval rejected for '{definition.name}'."
            )

        runtime = ToolRuntimeContext(
            execution_id=execution_id,
        )

        output = tool.execute(
            validated_input,
            runtime,
        )

        return ToolResult(
            tool_name=definition.name,
            success=True,
            output=output,
        )
