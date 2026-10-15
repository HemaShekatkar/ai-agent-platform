# app/tools/approval.py

from dataclasses import dataclass
from enum import StrEnum

from app.tools.models import (
    ApprovalRequirement,
    ToolDefinition,
)


# StrEnum: crosses Temporal's JSON payload boundary inside activity requests
class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ApprovalDecision:
    execution_id: str
    action_fingerprint: str
    status: ApprovalStatus
    approved_by: str | None = None


class ApprovalPolicy:

    def evaluate(
        self,
        execution_id: str,
        tool: ToolDefinition,
        approval: ApprovalDecision | None,
        expected_fingerprint: str,
    ) -> ApprovalStatus:

        if tool.approval_requirement == ApprovalRequirement.NONE:
            return ApprovalStatus.NOT_REQUIRED

        if approval is None:
            return ApprovalStatus.PENDING

        if approval.execution_id != execution_id:
            return ApprovalStatus.PENDING

        if approval.action_fingerprint != expected_fingerprint:
            return ApprovalStatus.PENDING

        if approval.status not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
        }:
            return ApprovalStatus.PENDING

        return approval.status
