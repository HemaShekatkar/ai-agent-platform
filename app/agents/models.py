
from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    GET_ONBOARDING_STATUS = "GET_ONBOARDING_STATUS"
    REQUEST_CONTRACT = "REQUEST_CONTRACT"
    INITIATE_SECURITY_REVIEW = "INITIATE_SECURITY_REVIEW"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"


class AgentStatus(str, Enum):
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    resource_id: str


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    reason: str | None = None
    retryable: bool = False
    state_version: str | None = None


@dataclass
class AgentExecutionState:
    steps_used: int = 0
    max_steps: int = 10

    planning_attempts: int = 0
    max_planning_attempts: int = 15

    action_history: list[tuple[AgentAction, ToolResult]] = field(
        default_factory=list
    )

    blocked_signatures: set[
        tuple[str, str, str, str | None]
    ] = field(default_factory=set)
