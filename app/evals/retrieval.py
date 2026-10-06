from dataclasses import dataclass


def recall_at_k(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    # Set intersection so duplicate retrievals are not counted twice
    hits = relevant_ids & set(retrieved_ids[:k])

    return len(hits) / len(relevant_ids)


def precision_at_k(
    relevant_ids: set[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    return len(set(top_k) & relevant_ids) / len(top_k)


def reciprocal_rank(
    relevant_ids: set[str],
    retrieved_ids: list[str],
) -> float:
    # Ranks are 1-based: a relevant chunk in first place scores 1.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


@dataclass
class RetrievalEvalCase:
    id: str
    query: str
    relevant_ids: set[str]


@dataclass
class RetrievalEvalResult:
    case_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


@dataclass
class RetrievalEvalSummary:
    total_cases: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float


class RetrievalEvalRunner:

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")

        self.k = k

    def evaluate(
        self,
        case: RetrievalEvalCase,
        retrieved_ids: list[str],
    ) -> RetrievalEvalResult:

        return RetrievalEvalResult(
            case_id=case.id,
            recall_at_k=recall_at_k(
                case.relevant_ids,
                retrieved_ids,
                self.k,
            ),
            precision_at_k=precision_at_k(
                case.relevant_ids,
                retrieved_ids,
                self.k,
            ),
            reciprocal_rank=reciprocal_rank(
                case.relevant_ids,
                retrieved_ids,
            ),
        )

    def run_suite(
        self,
        cases: list[RetrievalEvalCase],
        retrieved_by_case: dict[str, list[str]],
    ) -> RetrievalEvalSummary:

        if not cases:
            raise ValueError("cases must not be empty")

        results = [
            self.evaluate(
                case,
                retrieved_by_case[case.id],
            )
            for case in cases
        ]

        return RetrievalEvalSummary(
            total_cases=len(results),
            mean_recall_at_k=_mean(
                [result.recall_at_k for result in results]
            ),
            mean_precision_at_k=_mean(
                [result.precision_at_k for result in results]
            ),
            mean_reciprocal_rank=_mean(
                [result.reciprocal_rank for result in results]
            ),
        )


def _mean(values: list[float]) -> float:
    # No cases means no score, not a divide-by-zero
    if not values:
        return 0.0

    return sum(values) / len(values)


def test_retrieval_eval_suite():

    cases = [
        RetrievalEvalCase(
            id="case-1",
            query="security requirements",
            relevant_ids={"a", "b"},
        ),
        RetrievalEvalCase(
            id="case-2",
            query="metadata requirements",
            relevant_ids={"x"},
        ),
    ]

    retrieved = {
        "case-1": ["a", "c", "d"],
        "case-2": ["z", "x", "y"],
    }

    runner = RetrievalEvalRunner(k=3)

    summary = runner.run_suite(
        cases,
        retrieved,
    )
