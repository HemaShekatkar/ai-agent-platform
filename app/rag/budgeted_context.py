
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceDocument:
    id: str
    requirements: frozenset[str]
    relevance: float
    tokens: int
    authoritative: bool


@dataclass(frozen=True)
class ContextSelection:
    documents: list[EvidenceDocument]
    covered: frozenset[str]
    missing: frozenset[str]
    tokens_used: int
    complete: bool


def select_context(
    candidates: list[EvidenceDocument],
    required: set[str],
    token_budget: int,
) -> ContextSelection:
    if token_budget < 0:
        raise ValueError("Token budget must be nonnegative")

    selected: list[EvidenceDocument] = []
    selected_ids: set[str] = set()
    uncovered = set(required)
    remaining = token_budget

    while uncovered:
        feasible = [
            doc for doc in candidates
            if doc.id not in selected_ids
            and doc.authoritative
            and 0 < doc.tokens <= remaining
            and doc.requirements & uncovered
        ]

        if not feasible:
            break

        def utility(doc: EvidenceDocument) -> float:
            new_coverage = len(doc.requirements & uncovered)
            return new_coverage * doc.relevance / doc.tokens

        best = max(
            feasible,
            key=lambda d: (utility(d), -d.tokens, d.id),
        )

        selected.append(best)
        selected_ids.add(best.id)
        uncovered -= best.requirements
        remaining -= best.tokens

    return ContextSelection(
        documents=selected,
        covered=frozenset(required - uncovered),
        missing=frozenset(uncovered),
        tokens_used=token_budget - remaining,
        complete=not uncovered,
    )
