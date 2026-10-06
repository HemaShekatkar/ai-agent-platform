from pydantic import BaseModel
from typing import TYPE_CHECKING, Literal, Protocol
from dataclasses import dataclass, field

from app.retrieval.context import ContextResult
from app.routing.models import DataClassification

if TYPE_CHECKING:
    # Type-only import: app.evals.groundedness imports GroundedDecision
    # from this module, so a runtime import would be circular
    from app.evals.groundedness import GroundednessResult, GroundednessEvaluator


class GroundedDecision(BaseModel):
    decision: Literal[
        "APPROVE",
        "REJECT",
        "HUMAN_REVIEW",
    ]
    reason: str
    citations: list[str]


@dataclass
class Citation:
    reference: str
    classification: DataClassification


def find_invalid_citations(
    citations: list[str],
    included_chunk_ids: list[str],
) -> list[str]:

    valid_ids = set(included_chunk_ids)

    return [
        citation
        for citation in citations
        if citation not in valid_ids
    ]


def requires_citation(decision: str) -> bool:
    return decision in {
        "APPROVE",
        "REJECT",
    }


def has_required_citations(
    decision: GroundedDecision,
) -> bool:

    if not requires_citation(decision.decision):
        return True

    return len(decision.citations) > 0


@dataclass
class CitationValidationResult:
    valid: bool
    missing_required_citations: bool
    invalid_citations: list[str]


def validate_decision_citations(
    decision: GroundedDecision,
    included_chunk_ids: list[str],
) -> CitationValidationResult:

    missing_required = not has_required_citations(
        decision
    )

    invalid = find_invalid_citations(
        decision.citations,
        included_chunk_ids,
    )

    return CitationValidationResult(
        valid=(
            not missing_required
            and not invalid
        ),
        missing_required_citations=missing_required,
        invalid_citations=invalid,
    )


@dataclass
class AgentDecisionResult:
    decision: GroundedDecision
    attempts: int
    recovered: bool


class DecisionGenerator(Protocol):

    def __call__(
        self,
        context: ContextResult,
        validation_feedback: GroundingValidationResult | None = None,
    ) -> GroundedDecision:
        ...


@dataclass
class GroundingValidationResult:
    valid: bool
    citation_validation: CitationValidationResult
    groundedness: "GroundednessResult | None"
    validation_feedback: CitationValidationResult | None = None


def decide_with_grounding_recovery(
    context: ContextResult,
    generate: DecisionGenerator,
    groundedness_evaluator: "GroundednessEvaluator",
) -> AgentDecisionResult:

    decision = generate(context)

    validation = validate_grounding(
        decision,
        context,
        groundedness_evaluator,
    )

    if validation.valid:
        return AgentDecisionResult(
            decision=decision,
            attempts=1,
            recovered=False,
        )

    corrected = generate(
        context,
        validation_feedback=validation,
    )

    validation = validate_grounding(
        corrected,
        context,
        groundedness_evaluator,
    )

    if validation.valid:
        return AgentDecisionResult(
            decision=corrected,
            attempts=2,
            recovered=True,
        )

    return AgentDecisionResult(
        decision=GroundedDecision(
            decision="HUMAN_REVIEW",
            reason=(
                "Unable to produce a decision supported "
                "by valid cited evidence."
            ),
            citations=[],
        ),
        attempts=2,
        recovered=False,
    )


def validate_grounding(
    decision: GroundedDecision,
    context: ContextResult,
    groundedness_evaluator: "GroundednessEvaluator",
) -> GroundingValidationResult:

    citation_validation = validate_decision_citations(
        decision,
        context.included_chunk_ids,
    )

    if not citation_validation.valid:
        return GroundingValidationResult(
            valid=False,
            citation_validation=citation_validation,
            groundedness=None,
        )

    groundedness = groundedness_evaluator.evaluate(
        decision,
        context,
    )

    return GroundingValidationResult(
        valid=groundedness.grounded,
        citation_validation=citation_validation,
        groundedness=groundedness,
    )


def validate_groundedness(
    self,
    decision: GroundedDecision,
    context: ContextResult,
) -> GroundednessResult:

    with self.tracer.start_span("validation.groundedness") as span:
        result = self.groundedness_evaluator.evaluate(
            decision,
            context,
        )

        record_groundedness_attributes(
            span=span,
            result=result,
        )

        return result
