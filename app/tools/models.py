from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolRisk(Enum):
    READ_ONLY = "READ_ONLY"
    LOW = "LOW"
    HIGH = "HIGH"


class ApprovalRequirement(Enum):
    NONE = "NONE"
    HUMAN = "HUMAN"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: type
    risk: ToolRisk
    required_permissions: set[str]
    # No default: every tool must state whether it needs human approval
    approval_requirement: ApprovalRequirement


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class ToolExecutionContext:
    agent_id: str
    user_id: str
    agent_permissions: set[str]
    user_permissions: set[str]

    @property
    def effective_permissions(self) -> set[str]:
        # Intersection: a call may only use permissions that both the agent
        # and the user on whose behalf it acts hold
        return (
            self.agent_permissions
            & self.user_permissions
        )


class ApprovePartnerInput(BaseModel):
    partner_id: str


@dataclass(frozen=True)
class ToolExecution:
    execution_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolRuntimeContext:
    execution_id: str
