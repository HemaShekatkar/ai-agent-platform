from dataclasses import dataclass
from enum import Enum

from app.observability.agent_tracing import TraceAnalysis


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    code: str
    severity: AlertSeverity
    message: str


@dataclass(frozen=True)
class AlertPolicy:
    max_iterations: int = 6
    max_tokens: int = 15000
    max_latency_ms: float = 10000


def evaluate_alerts(
    trace: TraceAnalysis,
    policy: AlertPolicy,
) -> list[Alert]:

    alerts = []

    if trace.has_policy_conflict:
        alerts.append(
            Alert(
                code="POLICY_CONFLICT",
                severity=AlertSeverity.CRITICAL,
                message="Multiple policy versions detected",
            )
        )

    if trace.has_agent_loop:
        alerts.append(
            Alert(
                code="AGENT_LOOP",
                severity=AlertSeverity.WARNING,
                message="Potential repeated agent execution",
            )
        )

    if trace.total_iterations > policy.max_iterations:
        alerts.append(
            Alert(
                code="ITERATION_BUDGET_EXCEEDED",
                severity=AlertSeverity.WARNING,
                message="Agent iteration budget exceeded",
            )
        )

    if trace.total_tokens > policy.max_tokens:
        alerts.append(
            Alert(
                code="TOKEN_BUDGET_EXCEEDED",
                severity=AlertSeverity.WARNING,
                message="Token budget exceeded",
            )
        )

    if trace.total_latency_ms > policy.max_latency_ms:
        alerts.append(
            Alert(
                code="LATENCY_BUDGET_EXCEEDED",
                severity=AlertSeverity.WARNING,
                message="Execution latency budget exceeded",
            )
        )

    return alerts
