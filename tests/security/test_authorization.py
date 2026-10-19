import pytest

from app.security.authorization import (
    Action,
    Principal,
    WorkflowState,
    ToolRequest,
    authorize,
)


def make_principal(
    tenant_id="tenant-a",
    permissions=None,
):
    if permissions is None:
        permissions = frozenset({
            Action.READ_DOCUMENT,
            Action.CREATE_TICKET,
        })

    return Principal(
        principal_id="agent-service",
        tenant_id=tenant_id,
        permissions=permissions,
    )


def make_workflow(
    tenant_id="tenant-a",
    review_completed=False,
    approval_granted=False,
):
    return WorkflowState(
        workflow_id="workflow-123",
        tenant_id=tenant_id,
        partner_id="partner-123",
        security_review_completed=review_completed,
        security_approval_granted=approval_granted,
    )


def make_request(
    action=Action.APPROVE_SECURITY,
    tenant_id="tenant-a",
    workflow_id="workflow-123",
    partner_id="partner-123",
):
    return ToolRequest(
        action=action,
        workflow_id=workflow_id,
        tenant_id=tenant_id,
        partner_id=partner_id,
    )


def test_agent_cannot_approve_without_permission():
    decision = authorize(
        make_principal(),
        make_workflow(),
        make_request(),
    )

    assert not decision.allowed
    assert decision.reason == "PERMISSION_DENIED"


def test_review_must_be_completed():
    decision = authorize(
        make_principal(
            permissions=frozenset({
                Action.APPROVE_SECURITY,
            })
        ),
        make_workflow(),
        make_request(),
    )

    assert not decision.allowed
    assert decision.reason == "REVIEW_INCOMPLETE"


def test_human_approval_required():
    decision = authorize(
        make_principal(
            permissions=frozenset({
                Action.APPROVE_SECURITY,
            })
        ),
        make_workflow(review_completed=True),
        make_request(),
    )

    assert not decision.allowed
    assert decision.reason == "HUMAN_APPROVAL_REQUIRED"


def test_approved_action_allowed():
    decision = authorize(
        make_principal(
            permissions=frozenset({
                Action.APPROVE_SECURITY,
            })
        ),
        make_workflow(
            review_completed=True,
            approval_granted=True,
        ),
        make_request(),
    )

    assert decision.allowed


def test_cross_tenant_action_blocked():
    decision = authorize(
        make_principal(tenant_id="tenant-a"),
        make_workflow(tenant_id="tenant-b"),
        make_request(tenant_id="tenant-b"),
    )

    assert not decision.allowed
    assert decision.reason == "TENANT_MISMATCH"


def test_workflow_id_tampering_blocked():
    decision = authorize(
        make_principal(),
        make_workflow(),
        make_request(workflow_id="workflow-999"),
    )

    assert not decision.allowed
    assert decision.reason == "RESOURCE_MISMATCH"


def test_partner_id_tampering_blocked():
    decision = authorize(
        make_principal(),
        make_workflow(),
        make_request(partner_id="partner-999"),
    )

    assert not decision.allowed
    assert decision.reason == "RESOURCE_MISMATCH"


def test_authorized_ticket_creation():
    decision = authorize(
        make_principal(),
        make_workflow(),
        make_request(action=Action.CREATE_TICKET),
    )

    assert decision.allowed
