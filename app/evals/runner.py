from typing import Any

from app.evals.models import EvalCase, EvalResult, EvalSummary, CategorySummary
from app.evals.evaluators import Evaluator


class EvalRunner:

    def __init__(
        self,
        evaluators: list[Evaluator],
    ):
        self.evaluators = evaluators

    def evaluate(
        self,
        case: EvalCase,
        actual: dict[str, Any],
    ) -> list[EvalResult]:

        results = []

        for evaluator in self.evaluators:
            result = evaluator.evaluate(
                case,
                actual,
            )
            results.append(result)

        return results

    def run_suite(
        self,
        cases: list[EvalCase],
        actuals: dict[str, dict[str, Any]],
    ) -> list[EvalResult]:

        results = []

        for case in cases:
            actual = actuals[case.id]

            results.extend(
                self.evaluate(
                    case,
                    actual,
                )
            )

        return results

    def _aggregate_results(
        self,
        cases: list[EvalCase],
        results: list[EvalResult],
    ) -> CategorySummary:
        case_by_id = {case.id: case for case in cases}

        total_results = len(results)
        passed_results = sum(1 for result in results if result.passed)
        failed_results = total_results - passed_results

        # No results means nothing passed, not a divide-by-zero
        pass_rate = (
            passed_results / total_results
            if total_results > 0
            else 0.0
        )

        # Count each critical case once, however many evaluators failed it
        failed_critical_case_ids = {
            result.case_id
            for result in results
            if not result.passed
            and case_by_id[result.case_id].critical
        }

        return CategorySummary(
            total_cases=len(cases),
            total_results=total_results,
            passed_results=passed_results,
            failed_results=failed_results,
            pass_rate=pass_rate,
            critical_failures=len(failed_critical_case_ids),
        )

    def aggregate(
        self,
        cases: list[EvalCase],
        results: list[EvalResult],
    ) -> EvalSummary:
        overall = self._aggregate_results(cases, results)

        # Group cases by category, then route each result to its case's category
        cases_by_category: dict[str, list[EvalCase]] = {}
        for case in cases:
            cases_by_category.setdefault(case.category, []).append(case)

        category_by_case_id = {case.id: case.category for case in cases}
        results_by_category: dict[str, list[EvalResult]] = {
            category: [] for category in cases_by_category
        }
        for result in results:
            category = category_by_case_id[result.case_id]
            results_by_category[category].append(result)

        categories = {
            category: self._aggregate_results(
                category_cases,
                results_by_category[category],
            )
            for category, category_cases in cases_by_category.items()
        }

        return EvalSummary(
            total_cases=overall.total_cases,
            total_results=overall.total_results,
            passed_results=overall.passed_results,
            failed_results=overall.failed_results,
            pass_rate=overall.pass_rate,
            critical_failures=overall.critical_failures,
            categories=categories,
        )
