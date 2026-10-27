
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedDocument:
    id: str
    region: str
    relevance_score: float


@dataclass(frozen=True)
class CoverageResult:
    documents: list[RankedDocument]
    missing_regions: frozenset[str]
    complete: bool


def select_with_coverage(
    candidates: list[RankedDocument],
    required_regions: set[str],
    top_k: int,
) -> CoverageResult:
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")

    if len(required_regions) > top_k:
        raise ValueError(
            "Context budget cannot cover every required region"
        )

    # This function assumes candidates were already
    # authorized and validated against active policies.
    ordered = sorted(
        candidates,
        key=lambda d: (-d.relevance_score, d.id),
    )

    selected: list[RankedDocument] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()

    # Reserve one slot for each required region.
    for region in sorted(required_regions):
        best = next(
            (
                doc for doc in ordered
                if doc.region == region
                and doc.id not in selected_ids
            ),
            None,
        )

        if best is None:
            continue

        selected.append(best)
        selected_ids.add(best.id)
        covered.add(region)

    # Fill remaining slots by global relevance.
    for doc in ordered:
        if len(selected) >= top_k:
            break

        if doc.id not in selected_ids:
            selected.append(doc)
            selected_ids.add(doc.id)

    missing = frozenset(required_regions - covered)

    return CoverageResult(
        documents=selected,
        missing_regions=missing,
        complete=not missing,
    )
