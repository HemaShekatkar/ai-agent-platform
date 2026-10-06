from dataclasses import dataclass

from app.evals.models import EvalSummary, ReleaseDecision
from app.evals.policy import ReleasePolicy


@dataclass
class ReleasePolicy:
    min_pass_rate: float
    max_critical_failures: int = 0


class ReleasePolicyEvaluator:

    def __init__(self, policy: ReleasePolicy):
        self.policy = policy

    def evaluate(
        self,
        summary: EvalSummary,
    ) -> ReleaseDecision:

        reasons = []

        if summary.critical_failures > self.policy.max_critical_failures:
            reasons.append(
                f"Critical failures "
                f"{summary.critical_failures} exceed allowed "
                f"{self.policy.max_critical_failures}"
            )

        if summary.pass_rate < self.policy.min_pass_rate:
            reasons.append(
                f"Pass rate {summary.pass_rate:.2%} is below "
                f"required {self.policy.min_pass_rate:.2%}"
            )

        return ReleaseDecision(
            approved=len(reasons) == 0,
            reasons=reasons,
        )


@dataclass
class RoutingPolicy:
    version: str
    latency_weight: float
    cost_weight: float
