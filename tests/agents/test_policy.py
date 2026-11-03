
import pytest

from app.agents.models import ToolResult, ToolStatus
from app.agents.policy import (
    RuntimeDecision,
    classify_tool_result,
)


@pytest.mark.parametrize(
    "status, expected",
    [
        (ToolStatus.SUCCESS, RuntimeDecision.CONTINUE),
        (ToolStatus.TRANSIENT_FAILURE, RuntimeDecision.RETRY),
        (ToolStatus.PRECONDITION_FAILED, RuntimeDecision.REPLAN),
        (ToolStatus.PERMISSION_DENIED, RuntimeDecision.BLOCK_ACTION),
    ],
)
def test_tool_result_classification(status, expected):
    result = ToolResult(status=status)
    assert classify_tool_result(result) == expected


def test_permission_denied_cannot_be_overridden_by_retry_hint():
    result = ToolResult(
        status=ToolStatus.PERMISSION_DENIED,
        reason="USER_NOT_AUTHORIZED",
        retryable=True,
    )

    assert classify_tool_result(result) == RuntimeDecision.BLOCK_ACTION


def test_precondition_failure_does_not_retry_same_action():
    result = ToolResult(
        status=ToolStatus.PRECONDITION_FAILED,
        reason="CONTRACT_NOT_SIGNED",
        retryable=True,
    )

    assert classify_tool_result(result) == RuntimeDecision.REPLAN
