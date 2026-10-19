from app.observability.agent_tracing import (
    AgentStep,
    analyze_agent_trace,
)
from app.observability.agent_alerts import (
    AlertPolicy,
    AlertSeverity,
    evaluate_alerts,
)


def test_healthy_trace_produces_no_alerts():
    trace = analyze_agent_trace([
        AgentStep(1, "retrieve", "v2", 500, 100, 200),
        AgentStep(2, "validate", "v2", 500, 100, 300),
    ])

    alerts = evaluate_alerts(trace, AlertPolicy())

    assert alerts == []


def test_policy_conflict_produces_critical_alert():
    trace = analyze_agent_trace([
        AgentStep(1, "retrieve", "v1", 500, 100, 200),
        AgentStep(2, "retrieve", "v2", 500, 100, 200),
    ])

    alerts = evaluate_alerts(trace, AlertPolicy())

    assert any(
        alert.code == "POLICY_CONFLICT"
        and alert.severity == AlertSeverity.CRITICAL
        for alert in alerts
    )


def test_token_budget_alert():
    trace = analyze_agent_trace([
        AgentStep(1, "retrieve", "v2", 12000, 5000, 200),
    ])

    alerts = evaluate_alerts(trace, AlertPolicy())

    assert any(
        alert.code == "TOKEN_BUDGET_EXCEEDED"
        for alert in alerts
    )


def test_latency_budget_alert():
    trace = analyze_agent_trace([
        AgentStep(1, "retrieve", "v2", 500, 100, 12000),
    ])

    alerts = evaluate_alerts(trace, AlertPolicy())

    assert any(
        alert.code == "LATENCY_BUDGET_EXCEEDED"
        for alert in alerts
    )


def test_agent_loop_alert():
    trace = analyze_agent_trace([
        AgentStep(i, "retrieve", "v2", 500, 100, 200)
        for i in range(1, 5)
    ])

    alerts = evaluate_alerts(trace, AlertPolicy())

    assert any(
        alert.code == "AGENT_LOOP"
        for alert in alerts
    )
