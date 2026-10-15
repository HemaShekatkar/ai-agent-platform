from dataclasses import dataclass
from enum import StrEnum
from typing import Any


# StrEnum: crosses Temporal's JSON payload boundary as the workflow result
class ToolWorkflowStatus(StrEnum):
    CREATED = "CREATED"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass(frozen=True)
class ToolWorkflowRequest:
    execution_id: str
    tool_name: str
    arguments: dict[str, Any]

    agent_id: str
    user_id: str

    requires_approval: bool


@dataclass
class ToolWorkflowResult:
    execution_id: str
    tool_name: str
    status: ToolWorkflowStatus
    output: Any | None = None
    error: str | None = None
