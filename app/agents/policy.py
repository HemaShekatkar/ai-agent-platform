
from enum import Enum

from app.agents.models import ToolResult, ToolStatus


class RuntimeDecision(str, Enum):
    CONTINUE = "CONTINUE"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    BLOCK_ACTION = "BLOCK_ACTION"


def classify_tool_result(result: ToolResult) -> RuntimeDecision:
    """Classify a tool result using trusted runtime policy."""

    if result.status == ToolStatus.SUCCESS:
        return RuntimeDecision.CONTINUE

    if result.status == ToolStatus.PERMISSION_DENIED:
        return RuntimeDecision.BLOCK_ACTION

    if result.status == ToolStatus.PRECONDITION_FAILED:
        return RuntimeDecision.REPLAN

    if result.status == ToolStatus.TRANSIENT_FAILURE:
        return RuntimeDecision.RETRY

    raise ValueError(f"Unsupported tool status: {result.status}")
