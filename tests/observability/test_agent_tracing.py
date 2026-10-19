import pytest

from app.observability.agent_tracing import (
    AgentStep,
    analyze_agent_trace,
)


def test_normal_execution():
    steps = [
        AgentStep(1, "retrieve_policy", "v2", 500, 100, 200),
        AgentStep(2, "validate_policy", "v2", 600, 100, 300),
    ]

    result = analyze_agent_trace(steps)

    assert result.total_iterations == 2
    assert result.total_tokens == 1300
    assert result.has_policy_conflict is False
    assert result.has_agent_loop is False


def test_policy_conflict():
    steps = [
        AgentStep(1, "retrieve_policy", "v1", 500, 100, 200),
        AgentStep(2, "retrieve_policy", "v2", 500, 100, 200),
    ]

    result = analyze_agent_trace(steps)

    assert result.has_policy_conflict is True
    assert result.policy_versions == {"v1", "v2"}


def test_repeated_tool_calls():
    steps = [
        AgentStep(i, "retrieve_policy", "v2", 500, 100, 200)
        for i in range(1, 5)
    ]

    result = analyze_agent_trace(steps)

    assert result.has_agent_loop is True
    assert result.repeated_tools["retrieve_policy"] == 4


def test_iteration_limit():
    steps = [
        AgentStep(i, f"tool_{i}", "v2", 500, 100, 200)
        for i in range(1, 8)
    ]

    result = analyze_agent_trace(
        steps,
        max_iterations=6,
    )

    assert result.has_agent_loop is True
    assert result.total_iterations == 7


def test_empty_trace():
    result = analyze_agent_trace([])

    assert result.total_iterations == 0
    assert result.total_tokens == 0
    assert result.has_agent_loop is False


def test_invalid_token_count():
    steps = [
        AgentStep(1, "retrieve_policy", "v2", -1, 100, 200),
    ]

    with pytest.raises(ValueError):
        analyze_agent_trace(steps)
