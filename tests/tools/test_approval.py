from pydantic import BaseModel

from app.tools.approval import ApprovalDecision, ApprovalPolicy, ApprovalStatus
from app.tools.fingerprint import action_fingerprint
from app.tools.models import ApprovalRequirement, ToolDefinition, ToolRisk


class PartnerInput(BaseModel):
    partner_id: str


def read_only_tool() -> ToolDefinition:
    return ToolDefinition(
        name="lookup_partner",
        description="Look up a partner's onboarding status.",
        input_schema=PartnerInput,
        risk=ToolRisk.READ_ONLY,
        required_permissions={"partner.read"},
        approval_requirement=ApprovalRequirement.NONE,
    )


def high_risk_tool() -> ToolDefinition:
    return ToolDefinition(
        name="approve_partner",
        description="Approve a partner for onboarding.",
        input_schema=PartnerInput,
        risk=ToolRisk.HIGH,
        required_permissions={"partner.approve"},
        approval_requirement=ApprovalRequirement.HUMAN,
    )


# Fingerprint of the action every approval in this file is granted for
APPROVED_ACTION = action_fingerprint(
    "approve_partner",
    {"partner_id": "partner-123"},
)


def approval(
    execution_id: str,
    status: ApprovalStatus = ApprovalStatus.APPROVED,
) -> ApprovalDecision:
    return ApprovalDecision(
        execution_id=execution_id,
        action_fingerprint=APPROVED_ACTION,
        status=status,
        approved_by="reviewer-1",
    )


def test_read_only_tool_does_not_require_approval():
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=read_only_tool(),
        approval=None,
        expected_fingerprint=APPROVED_ACTION,
    )

    assert status is ApprovalStatus.NOT_REQUIRED


def test_high_risk_tool_without_approval_is_pending():
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=high_risk_tool(),
        approval=None,
        expected_fingerprint=APPROVED_ACTION,
    )

    assert status is ApprovalStatus.PENDING


def test_high_risk_tool_with_matching_approval_is_approved():
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=high_risk_tool(),
        approval=approval("exec-1"),
        expected_fingerprint=APPROVED_ACTION,
    )

    assert status is ApprovalStatus.APPROVED


def test_approval_for_different_execution_is_not_accepted():
    # A reviewer approved exec-2, not this call
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=high_risk_tool(),
        approval=approval("exec-2"),
        expected_fingerprint=APPROVED_ACTION,
    )

    assert status is ApprovalStatus.PENDING


def test_rejected_action_cannot_execute():
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=high_risk_tool(),
        approval=approval("exec-1", status=ApprovalStatus.REJECTED),
        expected_fingerprint=APPROVED_ACTION,
    )

    assert status is ApprovalStatus.REJECTED
    assert status is not ApprovalStatus.APPROVED


def test_approval_cannot_be_reused_for_another_execution():
    policy = ApprovalPolicy()
    tool = high_risk_tool()
    granted = approval("exec-1")

    first = policy.evaluate(
        execution_id="exec-1",
        tool=tool,
        approval=granted,
        expected_fingerprint=APPROVED_ACTION,
    )

    # Replaying the same approval for a new call of the same tool must not
    # authorize it; each execution needs its own human decision
    second = policy.evaluate(
        execution_id="exec-2",
        tool=tool,
        approval=granted,
        expected_fingerprint=APPROVED_ACTION,
    )

    assert first is ApprovalStatus.APPROVED
    assert second is ApprovalStatus.PENDING


def test_approval_for_different_arguments_is_not_accepted():
    # The reviewer approved partner-123; the call now targets partner-999
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=high_risk_tool(),
        approval=approval("exec-1"),
        expected_fingerprint=action_fingerprint(
            "approve_partner",
            {"partner_id": "partner-999"},
        ),
    )

    assert status is ApprovalStatus.PENDING


def test_approval_for_different_tool_is_not_accepted():
    status = ApprovalPolicy().evaluate(
        execution_id="exec-1",
        tool=high_risk_tool(),
        approval=approval("exec-1"),
        expected_fingerprint=action_fingerprint(
            "delete_partner",
            {"partner_id": "partner-123"},
        ),
    )

    assert status is ApprovalStatus.PENDING
