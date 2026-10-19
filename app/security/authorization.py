from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    READ_DOCUMENT = "read_document"
    CREATE_TICKET = "create_ticket"
    APPROVE_SECURITY = "approve_security"


@dataclass(frozen=True)
class Principal:
    principal_id: str
    tenant_id: str
    permissions: frozenset[Action]


@dataclass(frozen=True)
class WorkflowState:
    workflow_id: str
    tenant_id: str
    partner_id: str
    security_review_completed: bool
    security_approval_granted: bool


@dataclass(frozen=True)
class ToolRequest:
    action: Action
    workflow_id: str
    tenant_id: str
    partner_id: str


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str


def authorize(
    principal: Principal,
    workflow: WorkflowState,
    request: ToolRequest,
) -> AuthorizationDecision:

    # Check tenant isolation.
    if (
        principal.tenant_id != workflow.tenant_id
        or request.tenant_id != workflow.tenant_id
    ):
        return AuthorizationDecision(
            False, "TENANT_MISMATCH"
        )

    # Check resource binding.
    if (
        request.workflow_id != workflow.workflow_id
        or request.partner_id != workflow.partner_id
    ):
        return AuthorizationDecision(
            False, "RESOURCE_MISMATCH"
        )

    # Check principal permissions.
    if request.action not in principal.permissions:
        return AuthorizationDecision(
            False, "PERMISSION_DENIED"
        )

    # Enforce security approval prerequisites.
    if request.action == Action.APPROVE_SECURITY:
        if not workflow.security_review_completed:
            return AuthorizationDecision(
                False, "REVIEW_INCOMPLETE"
            )

        if not workflow.security_approval_granted:
            return AuthorizationDecision(
                False, "HUMAN_APPROVAL_REQUIRED"
            )

    return AuthorizationDecision(True, "AUTHORIZED")
