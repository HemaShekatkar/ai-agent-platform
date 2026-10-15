from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from app.tools.models import (
    ApprovalRequirement,
    ApprovePartnerInput,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolRisk,
)


def make_definition(**overrides) -> ToolDefinition:
    values = {
        "name": "approve_partner",
        "description": "Approve a partner for onboarding.",
        "input_schema": ApprovePartnerInput,
        "risk": ToolRisk.HIGH,
        "required_permissions": {"partner.approve"},
        "approval_requirement": ApprovalRequirement.HUMAN,
    }

    values.update(overrides)

    return ToolDefinition(**values)


def test_tool_risk_values_are_stable():
    # Risk levels are persisted and used in policy checks; renames break both
    assert {risk.value for risk in ToolRisk} == {"READ_ONLY", "LOW", "HIGH"}


def test_tool_definition_holds_its_fields():
    definition = make_definition()

    assert definition.name == "approve_partner"
    assert definition.input_schema is ApprovePartnerInput
    assert definition.risk is ToolRisk.HIGH
    assert definition.required_permissions == {"partner.approve"}


def test_tool_definition_is_frozen():
    definition = make_definition()

    with pytest.raises(FrozenInstanceError):
        definition.risk = ToolRisk.READ_ONLY


def test_tool_call_holds_name_and_arguments():
    call = ToolCall(
        tool_name="approve_partner",
        arguments={"partner_id": "partner-123"},
    )

    assert call.tool_name == "approve_partner"
    assert call.arguments == {"partner_id": "partner-123"}


def test_tool_result_defaults_to_no_output_or_error():
    result = ToolResult(tool_name="approve_partner", success=True)

    assert result.output is None
    assert result.error is None


def test_tool_result_records_failure():
    result = ToolResult(
        tool_name="approve_partner",
        success=False,
        error="PERMISSION_DENIED",
    )

    assert result.success is False
    assert result.error == "PERMISSION_DENIED"
    assert result.output is None


def test_approve_partner_input_accepts_partner_id():
    parsed = ApprovePartnerInput.model_validate({"partner_id": "partner-123"})

    assert parsed.partner_id == "partner-123"


def test_approve_partner_input_rejects_missing_partner_id():
    with pytest.raises(ValidationError):
        ApprovePartnerInput.model_validate({})


def test_tool_call_arguments_validate_against_input_schema():
    # A model-proposed call is only trusted after schema validation
    definition = make_definition()
    call = ToolCall(
        tool_name="approve_partner",
        arguments={"partner_id": "partner-123"},
    )

    parsed = definition.input_schema.model_validate(call.arguments)

    assert parsed.partner_id == "partner-123"
