
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvidenceClause:
    clause_id: str
    policy_id: str
    text: str
    token_count: int
    covers_conditions: frozenset[str]
    authoritative: bool = True


@dataclass(frozen=True)
class CompressionResult:
    selected_clauses: tuple[EvidenceClause, ...]
    covered_conditions: frozenset[str]
    missing_conditions: frozenset[str]
    tokens_used: int
    complete: bool


def assemble_context(
    clauses: Sequence[EvidenceClause],
    required_conditions: frozenset[str],
    token_budget: int,
) -> CompressionResult:
    if token_budget < 0:
        raise ValueError("token_budget must be nonnegative")

    selected = []
    covered = set()
    used = 0
    remaining = [
        clause for clause in clauses
        if clause.authoritative
    ]

    # Greedy selection: prioritize newly covered
    # mandatory conditions per token.
    while remaining and covered != required_conditions:
        eligible = [
            clause for clause in remaining
            if used + clause.token_count <= token_budget
            and clause.token_count > 0
        ]

        if not eligible:
            break

        best = max(
            eligible,
            key=lambda clause: (
                len(
                    (clause.covers_conditions & required_conditions)
                    - covered
                ) / clause.token_count,
                -clause.token_count,
            ),
        )

        newly_covered = (
            best.covers_conditions & required_conditions
        ) - covered

        if not newly_covered:
            break

        selected.append(best)
        covered.update(newly_covered)
        used += best.token_count
        remaining.remove(best)

    missing = required_conditions - covered

    return CompressionResult(
        selected_clauses=tuple(selected),
        covered_conditions=frozenset(covered),
        missing_conditions=frozenset(missing),
        tokens_used=used,
        complete=not missing,
    )
