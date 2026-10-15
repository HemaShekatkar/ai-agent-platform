from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentEvalDimension(str, Enum):
    TASK_SUCCESS = "TASK_SUCCESS"
    TOOL_CORRECTNESS = "TOOL_CORRECTNESS"
    POLICY_COMPLIANCE = "POLICY_COMPLIANCE"
    RELIABILITY = "RELIABILITY"
    EFFICIENCY = "EFFICIENCY"


@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    user_request: str

    expected_tool: str | None
    expected_arguments: dict[str, Any]

    expected_success: bool
    expected_approval: bool

    description: str = ""
    expected_refusal: bool = False
    trusted_permissions: frozenset[str] = frozenset()
    untrusted_content: str = ""


@dataclass(frozen=True)
class AgentEvalInput:
    user_request: str
    trusted_permissions: frozenset[str]
    untrusted_content: str = ""


@dataclass
class AgentEvalResult:
    case_id: str

    task_success: bool
    tool_correct: bool
    policy_compliant: bool
    reliable: bool

    latency_ms: float = 0.0
    token_count: int = 0

    errors: list[str] = field(default_factory=list)


@dataclass
class AgentEvalScorecard:
    total_cases: int

    task_success_rate: float
    tool_accuracy: float
    policy_compliance_rate: float
    reliability_rate: float

    avg_latency_ms: float
    avg_tokens: float


def calculate_scorecard(
    results: list[AgentEvalResult],
) -> AgentEvalScorecard:

    if not results:
        raise ValueError("Evaluation results cannot be empty")

    total = len(results)

    return AgentEvalScorecard(
        total_cases=total,
        task_success_rate=(
            sum(r.task_success for r in results) / total
        ),
        tool_accuracy=(
            sum(r.tool_correct for r in results) / total
        ),
        policy_compliance_rate=(
            sum(r.policy_compliant for r in results) / total
        ),
        reliability_rate=(
            sum(r.reliable for r in results) / total
        ),
        avg_latency_ms=(
            sum(r.latency_ms for r in results) / total
        ),
        avg_tokens=(
            sum(r.token_count for r in results) / total
        ),
    )
