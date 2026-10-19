import pytest

from app.evals.model_comparison import (
    EvalCase,
    ModelResult,
    RiskLevel,
    evaluate_model,
    compare_models,
)


def make_cases():
    return [
        EvalCase("1", "security", RiskLevel.HIGH, False),
        EvalCase("2", "security", RiskLevel.HIGH, True),
        EvalCase("3", "metadata", RiskLevel.LOW, True),
        EvalCase("4", "metadata", RiskLevel.LOW, False),
    ]


def test_perfect_model():
    cases = make_cases()

    results = [
        ModelResult("1", False, 100, 0.01),
        ModelResult("2", True, 100, 0.01),
        ModelResult("3", True, 100, 0.01),
        ModelResult("4", False, 100, 0.01),
    ]

    metrics = evaluate_model(cases, results)

    assert metrics.accuracy == 1.0
    assert metrics.false_approvals == 0
    assert metrics.false_rejections == 0
    assert metrics.total_cost_usd == pytest.approx(0.04)


def test_false_approval_detected():
    cases = make_cases()

    results = [
        ModelResult("1", True, 100, 0.01),
        ModelResult("2", True, 100, 0.01),
        ModelResult("3", True, 100, 0.01),
        ModelResult("4", False, 100, 0.01),
    ]

    metrics = evaluate_model(cases, results)

    assert metrics.accuracy == 0.75
    assert metrics.false_approvals == 1


def test_false_rejection_detected():
    cases = make_cases()

    results = [
        ModelResult("1", False, 100, 0.01),
        ModelResult("2", False, 100, 0.01),
        ModelResult("3", True, 100, 0.01),
        ModelResult("4", False, 100, 0.01),
    ]

    metrics = evaluate_model(cases, results)

    assert metrics.false_rejections == 1


def test_missing_results_rejected():
    with pytest.raises(ValueError):
        evaluate_model(
            make_cases(),
            [ModelResult("1", False, 100, 0.01)],
        )


def test_duplicate_results_rejected():
    cases = make_cases()

    results = [
        ModelResult("1", False, 100, 0.01),
        ModelResult("1", False, 100, 0.01),
    ]

    with pytest.raises(ValueError):
        evaluate_model(cases, results)


def test_baseline_candidate_comparison():
    cases = make_cases()

    baseline = [
        ModelResult("1", True, 100, 0.01),
        ModelResult("2", True, 100, 0.01),
        ModelResult("3", True, 100, 0.01),
        ModelResult("4", False, 100, 0.01),
    ]

    candidate = [
        ModelResult("1", False, 150, 0.02),
        ModelResult("2", True, 150, 0.02),
        ModelResult("3", True, 150, 0.02),
        ModelResult("4", False, 150, 0.02),
    ]

    comparison = compare_models(
        cases,
        baseline,
        candidate,
    )

    assert comparison["baseline"].accuracy == 0.75
    assert comparison["candidate"].accuracy == 1.0
    assert (
        comparison["candidate"].total_cost_usd
        > comparison["baseline"].total_cost_usd
    )
