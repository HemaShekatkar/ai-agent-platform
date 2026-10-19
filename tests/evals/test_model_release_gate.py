import pytest

from app.evals.model_comparison import (
    EvalCase,
    ModelResult,
    RiskLevel,
    evaluate_segmented,
    percentile,
)
from app.evals.model_release_gate import (
    ReleaseDecision,
    ReleasePolicy,
    evaluate_release,
)


def make_dataset(n=100):
    cases = []
    baseline = []
    candidate = []

    for i in range(n):
        high_risk = i < 30
        expected_pass = i % 2 == 0

        cases.append(
            EvalCase(
                case_id=str(i),
                category="security" if high_risk else "metadata",
                risk_level=(
                    RiskLevel.HIGH
                    if high_risk
                    else RiskLevel.LOW
                ),
                expected_pass=expected_pass,
            )
        )

        baseline.append(
            ModelResult(
                case_id=str(i),
                predicted_pass=expected_pass,
                latency_ms=100,
                cost_usd=0.01,
            )
        )

        candidate.append(
            ModelResult(
                case_id=str(i),
                predicted_pass=expected_pass,
                latency_ms=120,
                cost_usd=0.012,
            )
        )

    return cases, baseline, candidate


def test_percentile():
    assert percentile([10, 20, 30, 40, 50], 95) == 50


def test_segmented_metrics():
    cases, baseline, _ = make_dataset()

    metrics = evaluate_segmented(cases, baseline)

    assert metrics.overall.total_cases == 100
    assert metrics.by_risk[RiskLevel.HIGH].total_cases == 30
    assert metrics.by_risk[RiskLevel.LOW].total_cases == 70
    assert metrics.p95_latency_ms == 100


def test_candidate_passes_release_gate():
    cases, baseline, candidate = make_dataset()

    result = evaluate_release(
        evaluate_segmented(cases, baseline),
        evaluate_segmented(cases, candidate),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.PASS


def test_high_risk_false_approval_blocks_release():
    cases, baseline, candidate = make_dataset()

    # Case 1 is high-risk and expected to fail.
    candidate[1] = ModelResult(
        case_id="1",
        predicted_pass=True,
        latency_ms=120,
        cost_usd=0.012,
    )

    result = evaluate_release(
        evaluate_segmented(cases, baseline),
        evaluate_segmented(cases, candidate),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.BLOCK
    assert "High-risk false approvals increased" in result.reasons


def test_excessive_cost_blocks_release():
    cases, baseline, candidate = make_dataset()

    candidate = [
        ModelResult(
            case_id=r.case_id,
            predicted_pass=r.predicted_pass,
            latency_ms=r.latency_ms,
            cost_usd=0.02,
        )
        for r in candidate
    ]

    result = evaluate_release(
        evaluate_segmented(cases, baseline),
        evaluate_segmented(cases, candidate),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.BLOCK
    assert "Inference cost exceeded budget" in result.reasons


def test_insufficient_evidence():
    cases, baseline, candidate = make_dataset(n=10)

    result = evaluate_release(
        evaluate_segmented(cases, baseline),
        evaluate_segmented(cases, candidate),
        ReleasePolicy(),
    )

    assert result.decision == ReleaseDecision.INSUFFICIENT_EVIDENCE
