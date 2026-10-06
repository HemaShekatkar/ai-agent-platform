from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    id: str
    category: str
    input: dict[str, Any]
    expected: dict[str, Any]

    critical: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    case_id: str
    evaluator: str

    passed: bool
    score: float | None = None
    reason: str | None = None


@dataclass
class CategorySummary:
    total_cases: int
    total_results: int
    passed_results: int
    failed_results: int
    pass_rate: float
    critical_failures: int


@dataclass
class EvalSummary:
    # overall /grand summary
    total_cases: int
    total_results: int
    passed_results: int
    failed_results: int
    pass_rate: float
    critical_failures: int
    # Slice summaries
    categories: dict[str, CategorySummary]


@dataclass
class ReleaseDecision:
    approved: bool
    reasons: list[str]
