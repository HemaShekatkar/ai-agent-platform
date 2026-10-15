import pytest

from app.evals.agent_models import (
    AgentEvalResult,
    calculate_scorecard,
)


def test_calculate_scorecard():

    results = [
        AgentEvalResult(
            case_id="case-1",
            task_success=True,
            tool_correct=True,
            policy_compliant=True,
            reliable=True,
            latency_ms=100,
            token_count=200,
        ),
        AgentEvalResult(
            case_id="case-2",
            task_success=False,
            tool_correct=False,
            policy_compliant=True,
            reliable=True,
            latency_ms=300,
            token_count=400,
        ),
    ]

    scorecard = calculate_scorecard(results)

    assert scorecard.total_cases == 2
    assert scorecard.task_success_rate == 0.5
    assert scorecard.tool_accuracy == 0.5
    assert scorecard.policy_compliance_rate == 1.0
    assert scorecard.reliability_rate == 1.0
    assert scorecard.avg_latency_ms == 200
    assert scorecard.avg_tokens == 300


def test_empty_results_rejected():
    with pytest.raises(ValueError):
        calculate_scorecard([])
