import pytest
from pydantic import BaseModel

from app.tools.base import BaseTool
from app.tools.exceptions import DuplicateToolError, ToolNotFoundError
from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
    ToolRisk,
    ToolRuntimeContext,
)
from app.tools.registry import ToolRegistry


class PartnerInput(BaseModel):
    partner_id: str


class FakeTool(BaseTool[PartnerInput, str]):

    def __init__(self, name: str, risk: ToolRisk = ToolRisk.READ_ONLY):
        self._definition = ToolDefinition(
            name=name,
            description=f"Fake tool {name}.",
            input_schema=PartnerInput,
            risk=risk,
            required_permissions=set(),
            approval_requirement=ApprovalRequirement.NONE,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, input: PartnerInput, runtime: ToolRuntimeContext) -> str:
        return f"{self._definition.name}:{input.partner_id}"


def test_register_and_get_tool():
    registry = ToolRegistry()
    tool = FakeTool("lookup_partner")

    registry.register(tool)

    assert registry.get("lookup_partner") is tool


def test_unknown_tool_is_rejected():
    registry = ToolRegistry()
    registry.register(FakeTool("lookup_partner"))

    # A model can propose any tool name; only registered tools may run
    with pytest.raises(ToolNotFoundError, match="delete_partner"):
        registry.get("delete_partner")


def test_duplicate_tool_registration_is_rejected():
    registry = ToolRegistry()
    original = FakeTool("approve_partner", risk=ToolRisk.HIGH)
    registry.register(original)

    # A second tool with the same name must not silently replace the first,
    # e.g. swapping a HIGH-risk definition for a weaker one
    with pytest.raises(DuplicateToolError, match="approve_partner"):
        registry.register(FakeTool("approve_partner", risk=ToolRisk.READ_ONLY))

    assert registry.get("approve_partner") is original


def test_registry_returns_tool_definitions():
    registry = ToolRegistry()
    lookup = FakeTool("lookup_partner")
    approve = FakeTool("approve_partner", risk=ToolRisk.HIGH)

    registry.register(lookup)
    registry.register(approve)

    assert registry.definitions() == [lookup.definition, approve.definition]
    assert all(
        isinstance(definition, ToolDefinition)
        for definition in registry.definitions()
    )
