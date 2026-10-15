from dataclasses import dataclass, field
from enum import Enum


class ReleaseDecision(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ReleasePolicy:
    min_eval_cases: int = 100

    min_task_success_rate: float = 0.95
    min_tool_accuracy: float = 0.98

    max_critical_enforcement_failures: int = 0


@dataclass(frozen=True)
class ReleaseEvidence:
    # Actual agent proposal evaluations
    evaluated_cases: int
    successful_tasks: int
    correct_tool_decisions: int

    # Independent trusted-platform integration tests
    enforcement_tests_run: int
    critical_enforcement_failures: int


@dataclass
class ReleaseGateResult:
    decision: ReleaseDecision
    reasons: list[str] = field(default_factory=list)


def evaluate_release(
    evidence: ReleaseEvidence,
    policy: ReleasePolicy,
) -> ReleaseGateResult:

    reasons = []

    counts = [
        evidence.evaluated_cases,
        evidence.successful_tasks,
        evidence.correct_tool_decisions,
        evidence.enforcement_tests_run,
        evidence.critical_enforcement_failures,
    ]

    if any(count < 0 for count in counts):
        raise ValueError("Evidence counts cannot be negative")

    if (
        evidence.successful_tasks > evidence.evaluated_cases
        or evidence.correct_tool_decisions > evidence.evaluated_cases
        or evidence.critical_enforcement_failures
        > evidence.enforcement_tests_run
    ):
        raise ValueError("Evidence counts are inconsistent")

    # Critical failures take precedence over missing evidence.
    if (
        evidence.critical_enforcement_failures
        > policy.max_critical_enforcement_failures
    ):
        reasons.append("Critical platform enforcement failures detected")

    if reasons:
        return ReleaseGateResult(
            decision=ReleaseDecision.BLOCK,
            reasons=reasons,
        )

    # Do not infer safety from missing test evidence.
    if evidence.evaluated_cases < policy.min_eval_cases:
        reasons.append("Insufficient agent evaluation cases")

    if evidence.enforcement_tests_run == 0:
        reasons.append("No platform enforcement test evidence")

    if reasons:
        return ReleaseGateResult(
            decision=ReleaseDecision.INSUFFICIENT_EVIDENCE,
            reasons=reasons,
        )

    task_success_rate = (
        evidence.successful_tasks / evidence.evaluated_cases
    )

    tool_accuracy = (
        evidence.correct_tool_decisions / evidence.evaluated_cases
    )

    if task_success_rate < policy.min_task_success_rate:
        reasons.append("Task success rate below release threshold")

    if tool_accuracy < policy.min_tool_accuracy:
        reasons.append("Tool accuracy below release threshold")

    if reasons:
        return ReleaseGateResult(
            decision=ReleaseDecision.BLOCK,
            reasons=reasons,
        )

    return ReleaseGateResult(
        decision=ReleaseDecision.PASS,
        reasons=["All configured release criteria satisfied"],
    )
