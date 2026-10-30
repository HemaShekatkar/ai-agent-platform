
from dataclasses import dataclass
from collections import defaultdict
from statistics import mean
from typing import Sequence

from app.rag.evaluation import QueryEvaluation


@dataclass(frozen=True)
class QualityGate:
    min_recall: float
    min_mrr: float
    min_ndcg: float


@dataclass(frozen=True)
class CategoryReport:
    category: str
    query_count: int
    recall: float
    mrr: float
    ndcg: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    categories: tuple[CategoryReport, ...]
    passed: bool


DEFAULT_GATES = {
    "GENERAL_FAQ": QualityGate(0.85, 0.80, 0.80),
    "EXACT_POLICY": QualityGate(0.98, 0.95, 0.95),
    "MULTI_DOCUMENT": QualityGate(0.95, 0.90, 0.90),
}


def evaluate_release(
    results: Sequence[QueryEvaluation],
    gates: dict[str, QualityGate] | None = None,
) -> EvaluationReport:
    gates = DEFAULT_GATES if gates is None else gates

    grouped = defaultdict(list)

    for result in results:
        if result.category not in gates:
            raise ValueError(
                f"No release gate for category: {result.category}"
            )
        grouped[result.category].append(result)

    reports = []

    for category, gate in gates.items():
        queries = grouped.get(category, [])

        if not queries:
            reports.append(
                CategoryReport(
                    category=category,
                    query_count=0,
                    recall=0.0,
                    mrr=0.0,
                    ndcg=0.0,
                    passed=False,
                    failures=("No evaluation queries",),
                )
            )
            continue

        recall = mean(q.recall_at_k for q in queries)
        mrr = mean(q.reciprocal_rank for q in queries)
        ndcg = mean(q.ndcg_at_k for q in queries)

        failures = []

        if recall < gate.min_recall:
            failures.append("Recall below threshold")

        if mrr < gate.min_mrr:
            failures.append("MRR below threshold")

        if ndcg < gate.min_ndcg:
            failures.append("nDCG below threshold")

        reports.append(
            CategoryReport(
                category=category,
                query_count=len(queries),
                recall=recall,
                mrr=mrr,
                ndcg=ndcg,
                passed=not failures,
                failures=tuple(failures),
            )
        )

    return EvaluationReport(
        categories=tuple(reports),
        passed=all(report.passed for report in reports),
    )
