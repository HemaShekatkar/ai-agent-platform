from dataclasses import dataclass
from enum import Enum

from app.evals.model_comparison import (
    SegmentedMetrics,
    RiskLevel,
)


class ReleaseDecision(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ReleasePolicy:
    min_total_cases: int = 100
    min_high_risk_cases: int = 20
    max_accuracy_regression: float = 0.0
    max_cost_increase_pct: float = 40.0
    max_p95_latency_increase_pct: float = 60.0


@dataclass(frozen=True)
class ReleaseResult:
    decision: ReleaseDecision
    reasons: list[str]


def evaluate_release(
    baseline: SegmentedMetrics,
    candidate: SegmentedMetrics,
    policy: ReleasePolicy,
) -> ReleaseResult:

    reasons = []

    # Evidence sufficiency is evaluated first.
    baseline_high = baseline.by_risk.get(RiskLevel.HIGH)
    candidate_high = candidate.by_risk.get(RiskLevel.HIGH)

    if (
        baseline.overall.total_cases < policy.min_total_cases
        or candidate.overall.total_cases < policy.min_total_cases
        or baseline_high is None
        or candidate_high is None
        or baseline_high.total_cases < policy.min_high_risk_cases
        or candidate_high.total_cases < policy.min_high_risk_cases
    ):
        return ReleaseResult(
            decision=ReleaseDecision.INSUFFICIENT_EVIDENCE,
            reasons=["Evaluation dataset is insufficient"],
        )

    # Critical safety gate.
    if candidate_high.false_approvals > baseline_high.false_approvals:
        reasons.append(
            "High-risk false approvals increased"
        )

    # Accuracy regression gate.
    if (
        candidate.overall.accuracy
        < baseline.overall.accuracy
        - policy.max_accuracy_regression
    ):
        reasons.append("Overall accuracy regressed")

    # Cost regression gate.
    baseline_cost = baseline.overall.total_cost_usd
    candidate_cost = candidate.overall.total_cost_usd

    if baseline_cost > 0:
        cost_increase_pct = (
            (candidate_cost - baseline_cost)
            / baseline_cost
        ) * 100

        if cost_increase_pct > policy.max_cost_increase_pct:
            reasons.append("Inference cost exceeded budget")
    elif candidate_cost > 0:
        reasons.append("Inference cost increased from zero")

    # Latency regression gate.
    baseline_p95 = baseline.p95_latency_ms
    candidate_p95 = candidate.p95_latency_ms

    if baseline_p95 > 0:
        latency_increase_pct = (
            (candidate_p95 - baseline_p95)
            / baseline_p95
        ) * 100

        if latency_increase_pct > policy.max_p95_latency_increase_pct:
            reasons.append("p95 latency exceeded budget")
    elif candidate_p95 > 0:
        reasons.append("p95 latency increased from zero")

    return ReleaseResult(
        decision=(
            ReleaseDecision.BLOCK
            if reasons
            else ReleaseDecision.PASS
        ),
        reasons=reasons,
    )
