from pydantic import BaseModel

from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
    ToolExecutionContext,
    ToolRisk,
)
from app.tools.policy import ToolAuthorizer


class PartnerInput(BaseModel):
    partner_id: str


def make_tool(required_permissions: set[str]) -> ToolDefinition:
    return ToolDefinition(
        name="approve_partner",
        description="Approve a partner for onboarding.",
        input_schema=PartnerInput,
        risk=ToolRisk.HIGH,
        required_permissions=required_permissions,
        approval_requirement=ApprovalRequirement.HUMAN,
    )


def make_context(
    agent_permissions: set[str],
    user_permissions: set[str],
) -> ToolExecutionContext:
    return ToolExecutionContext(
        agent_id="onboarding-agent",
        user_id="user-123",
        agent_permissions=agent_permissions,
        user_permissions=user_permissions,
    )


def test_authorization_succeeds_when_all_permissions_exist():
    decision = ToolAuthorizer().authorize(
        make_tool({"partner.approve"}),
        make_context(
            agent_permissions={"partner.approve", "partner.read"},
            user_permissions={"partner.approve"},
        ),
    )

    assert decision.allowed is True
    assert decision.missing_permissions == set()


def test_authorization_fails_when_permission_missing():
    decision = ToolAuthorizer().authorize(
        make_tool({"partner.approve"}),
        make_context(
            agent_permissions={"partner.read"},
            user_permissions={"partner.read"},
        ),
    )

    assert decision.allowed is False
    assert decision.missing_permissions == {"partner.approve"}


def test_authorization_requires_all_permissions():
    # Holding one of several required permissions is not enough
    decision = ToolAuthorizer().authorize(
        make_tool({"partner.approve", "contract.sign"}),
        make_context(
            agent_permissions={"partner.approve", "contract.sign"},
            user_permissions={"partner.approve"},
        ),
    )

    assert decision.allowed is False
    assert decision.missing_permissions == {"contract.sign"}


def test_user_cannot_inherit_agent_permission():
    # The agent may approve partners, but this user may not; acting through
    # the agent must not grant the user the agent's privileges
    decision = ToolAuthorizer().authorize(
        make_tool({"partner.approve"}),
        make_context(
            agent_permissions={"partner.approve"},
            user_permissions=set(),
        ),
    )

    assert decision.allowed is False
    assert decision.missing_permissions == {"partner.approve"}


def test_agent_cannot_inherit_user_permission():
    # The user may approve partners, but this agent was not granted that
    # capability; a privileged user does not widen the agent's scope
    decision = ToolAuthorizer().authorize(
        make_tool({"partner.approve"}),
        make_context(
            agent_permissions=set(),
            user_permissions={"partner.approve"},
        ),
    )

    assert decision.allowed is False
    assert decision.missing_permissions == {"partner.approve"}
