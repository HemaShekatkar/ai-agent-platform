from typing import Any, TypedDict

from enum import Enum


class AgentExecutionPhase(str, Enum):
    NEW = "NEW"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OnboardingAgentState(TypedDict, total=False):
    agent_run_id: str
    user_request: str

    agent_id: str
    user_id: str

    proposed_tool: str
    proposed_arguments: dict[str, Any]

    execution_id: str
    requires_approval: bool

    temporal_workflow_id: str
    workflow_status: str

    execution_phase: str

    tool_output: Any
    error: str | None
    final_response: str
