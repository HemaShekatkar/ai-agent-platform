from dataclasses import dataclass

from app.tools.models import (
    ToolDefinition,
    ToolExecutionContext,
)


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    missing_permissions: set[str]


class ToolAuthorizer:

    def authorize(
        self,
        tool: ToolDefinition,
        context: ToolExecutionContext,
    ) -> AuthorizationDecision:

        missing = (
            tool.required_permissions
            - context.effective_permissions
        )

        return AuthorizationDecision(
            allowed=not missing,
            missing_permissions=missing,
        )
