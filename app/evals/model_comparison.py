import math
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    risk_level: RiskLevel
    expected_pass: bool


@dataclass(frozen=True)
class ModelResult:
    case_id: str
    predicted_pass: bool
    latency_ms: float
    cost_usd: float


@dataclass(frozen=True)
class EvaluationMetrics:
    total_cases: int
    accuracy: float
    false_approvals: int
    false_rejections: int
    total_cost_usd: float
    average_latency_ms: float


def evaluate_model(
    cases: list[EvalCase],
    results: list[ModelResult],
) -> EvaluationMetrics:

    if not cases:
        raise ValueError("Evaluation dataset cannot be empty")

    case_ids = [case.case_id for case in cases]
    result_ids = [result.case_id for result in results]

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate evaluation case IDs")

    if len(set(result_ids)) != len(result_ids):
        raise ValueError("Duplicate result IDs")

    if set(case_ids) != set(result_ids):
        raise ValueError("Results must match evaluation cases")

    results_by_id = {
        result.case_id: result
        for result in results
    }

    correct = 0
    false_approvals = 0
    false_rejections = 0
    total_cost = 0.0
    total_latency = 0.0

    for case in cases:
        result = results_by_id[case.case_id]

        if result.latency_ms < 0 or result.cost_usd < 0:
            raise ValueError("Cost and latency cannot be negative")

        if result.predicted_pass == case.expected_pass:
            correct += 1

        if result.predicted_pass and not case.expected_pass:
            false_approvals += 1

        if not result.predicted_pass and case.expected_pass:
            false_rejections += 1

        total_cost += result.cost_usd
        total_latency += result.latency_ms

    return EvaluationMetrics(
        total_cases=len(cases),
        accuracy=correct / len(cases),
        false_approvals=false_approvals,
        false_rejections=false_rejections,
        total_cost_usd=total_cost,
        average_latency_ms=total_latency / len(cases),
    )


def compare_models(
    cases: list[EvalCase],
    baseline: list[ModelResult],
    candidate: list[ModelResult],
) -> dict[str, EvaluationMetrics]:

    return {
        "baseline": evaluate_model(cases, baseline),
        "candidate": evaluate_model(cases, candidate),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty data")

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    ordered = sorted(values)
    rank = math.ceil((p / 100) * len(ordered))

    return ordered[max(0, rank - 1)]


@dataclass(frozen=True)
class SegmentedMetrics:
    overall: EvaluationMetrics
    by_risk: dict[RiskLevel, EvaluationMetrics]
    p95_latency_ms: float


def evaluate_segmented(
    cases: list[EvalCase],
    results: list[ModelResult],
) -> SegmentedMetrics:

    overall = evaluate_model(cases, results)

    results_by_id = {
        result.case_id: result
        for result in results
    }

    by_risk = {}

    for risk in RiskLevel:
        segment_cases = [
            case for case in cases
            if case.risk_level == risk
        ]

        if not segment_cases:
            continue

        segment_results = [
            results_by_id[case.case_id]
            for case in segment_cases
        ]

        by_risk[risk] = evaluate_model(
            segment_cases,
            segment_results,
        )

    return SegmentedMetrics(
        overall=overall,
        by_risk=by_risk,
        p95_latency_ms=percentile(
            [result.latency_ms for result in results],
            95,
        ),
    )
