from app.agents.onboarding import (
    GroundedDecision,
    find_invalid_citations,
    has_required_citations,
    requires_citation,
    validate_decision_citations,
    GroundingValidationResult,
    decide_with_grounding_recovery,
    validate_grounding,
)
from app.evals.groundedness import GroundednessResult
from app.retrieval.context import ContextItem, ContextResult


invalid_citation = "security-policy-999"
valid_citation = "security-policy-17"

included_chunk_ids = [
    valid_citation,
    "metadata-policy-8",
    "privacy-policy-42",
]


def test_valid_citations():
    assert find_invalid_citations(
        [valid_citation, "metadata-policy-8"],
        included_chunk_ids,
    ) == []


def test_one_hallucinated_citation():
    assert find_invalid_citations(
        [invalid_citation],
        included_chunk_ids,
    ) == [invalid_citation]


def make_decision(
    decision: str,
    citations: list[str],
) -> GroundedDecision:
    return GroundedDecision(
        decision=decision,
        reason="Test reason",
        citations=citations,
    )


def test_requires_citation_for_approve_and_reject_only():
    assert requires_citation("APPROVE") is True
    assert requires_citation("REJECT") is True
    assert requires_citation("HUMAN_REVIEW") is False


def test_approve_with_citation_is_valid():
    assert has_required_citations(
        make_decision("APPROVE", [valid_citation])
    ) is True


def test_approve_without_citation_is_invalid():
    assert has_required_citations(
        make_decision("APPROVE", [])
    ) is False


def test_reject_with_citation_is_valid():
    assert has_required_citations(
        make_decision("REJECT", [valid_citation])
    ) is True


def test_reject_without_citation_is_invalid():
    assert has_required_citations(
        make_decision("REJECT", [])
    ) is False


def test_human_review_without_citation_is_valid():
    assert has_required_citations(
        make_decision("HUMAN_REVIEW", [])
    ) is True


def test_approve_with_valid_citation_passes_validation():
    result = validate_decision_citations(
        make_decision("APPROVE", [valid_citation]),
        included_chunk_ids,
    )

    assert result.valid is True
    assert result.missing_required_citations is False
    assert result.invalid_citations == []


def test_approve_without_citations_fails_validation():
    result = validate_decision_citations(
        make_decision("APPROVE", []),
        included_chunk_ids,
    )

    assert result.valid is False
    assert result.missing_required_citations is True
    assert result.invalid_citations == []


def test_approve_with_fake_citation_fails_validation():
    result = validate_decision_citations(
        make_decision("APPROVE", ["fake-999"]),
        included_chunk_ids,
    )

    assert result.valid is False
    assert result.missing_required_citations is False
    assert result.invalid_citations == ["fake-999"]


class FakeDecisionGenerator:
    def __init__(self, decisions: list[GroundedDecision]):
        self.decisions = decisions
        self.call_count = 0
        self.contexts = []
        self.feedback = []

    def __call__(
        self,
        context: ContextResult,
        validation_feedback: GroundingValidationResult | None = None,
    ) -> GroundedDecision:
        self.contexts.append(context)
        self.feedback.append(validation_feedback)

        decision = self.decisions[self.call_count]
        self.call_count += 1

        return decision


def make_context() -> ContextResult:
    return ContextResult(
        text="Retrieved policy context",
        items=[
            ContextItem(
                chunk_id=chunk_id,
                text=f"Policy text for {chunk_id}",
            )
            for chunk_id in included_chunk_ids
        ],
        included_chunk_ids=included_chunk_ids,
        excluded_chunks=[],
        budget_tokens=1000,
        used_tokens=100,
    )


class FakeGroundednessEvaluator:

    def __init__(self, result: GroundednessResult):
        self.result = result
        self.call_count = 0

    def evaluate(
        self,
        decision: GroundedDecision,
        context: ContextResult,
    ) -> GroundednessResult:
        self.call_count += 1
        return self.result


def test_invalid_citation_skips_groundedness_evaluation():
    evaluator = FakeGroundednessEvaluator(
        GroundednessResult(
            grounded=True,
            score=1.0,
            reason="Should never be used.",
        )
    )

    result = validate_grounding(
        make_decision("APPROVE", [invalid_citation]),
        make_context(),
        evaluator,
    )

    assert result.valid is False
    assert result.groundedness is None
    assert result.citation_validation.invalid_citations == [
        invalid_citation,
    ]
    assert evaluator.call_count == 0


def test_valid_citation_with_grounded_response_is_valid():
    evaluator = FakeGroundednessEvaluator(
        GroundednessResult(
            grounded=True,
            score=0.95,
            reason="The cited evidence supports the decision.",
        )
    )

    result = validate_grounding(
        make_decision("APPROVE", [valid_citation]),
        make_context(),
        evaluator,
    )

    assert result.valid is True
    assert result.citation_validation.valid is True
    assert result.groundedness is not None
    assert result.groundedness.grounded is True
    assert evaluator.call_count == 1


def test_valid_citation_with_unsupported_response_is_invalid():
    evaluator = FakeGroundednessEvaluator(
        GroundednessResult(
            grounded=False,
            score=0.2,
            reason="The cited evidence contradicts the decision.",
        )
    )

    result = validate_grounding(
        make_decision("APPROVE", [valid_citation]),
        make_context(),
        evaluator,
    )

    # Citations are fine; the failure comes from groundedness alone
    assert result.valid is False
    assert result.citation_validation.valid is True
    assert result.groundedness is not None
    assert result.groundedness.grounded is False
    assert evaluator.call_count == 1


class ScriptedGroundednessEvaluator:

    def __init__(self, results: list[GroundednessResult]):
        self.results = results
        self.call_count = 0
        self.decisions = []

    def evaluate(
        self,
        decision: GroundedDecision,
        context: ContextResult,
    ) -> GroundednessResult:
        self.decisions.append(decision)

        result = self.results[self.call_count]
        self.call_count += 1

        return result


grounded = GroundednessResult(
    grounded=True,
    score=0.95,
    reason="The cited evidence supports the decision.",
)

ungrounded = GroundednessResult(
    grounded=False,
    score=0.2,
    reason="The cited evidence does not support the decision.",
)


def test_grounded_first_decision_does_not_retry():
    first = make_decision("APPROVE", [valid_citation])
    generator = FakeDecisionGenerator([first])
    evaluator = ScriptedGroundednessEvaluator([grounded])

    result = decide_with_grounding_recovery(
        make_context(),
        generator,
        evaluator,
    )

    assert result.decision is first
    assert result.attempts == 1
    assert result.recovered is False
    assert generator.call_count == 1
    assert evaluator.call_count == 1


def test_invalid_citation_recovers_on_grounded_retry():
    retry = make_decision("APPROVE", [valid_citation])
    generator = FakeDecisionGenerator([
        make_decision("APPROVE", ["fake-999"]),
        retry,
    ])
    evaluator = ScriptedGroundednessEvaluator([grounded])

    result = decide_with_grounding_recovery(
        make_context(),
        generator,
        evaluator,
    )

    assert result.decision is retry
    assert result.attempts == 2
    assert result.recovered is True

    # The bad citation is caught before the judge is ever called
    assert evaluator.decisions == [retry]

    feedback = generator.feedback[1]
    assert feedback.citation_validation.invalid_citations == ["fake-999"]
    assert feedback.groundedness is None


def test_ungrounded_decision_recovers_on_grounded_retry():
    retry = make_decision("REJECT", [valid_citation])
    generator = FakeDecisionGenerator([
        make_decision("APPROVE", [valid_citation]),
        retry,
    ])
    evaluator = ScriptedGroundednessEvaluator([ungrounded, grounded])

    result = decide_with_grounding_recovery(
        make_context(),
        generator,
        evaluator,
    )

    assert result.decision is retry
    assert result.attempts == 2
    assert result.recovered is True
    assert evaluator.call_count == 2

    # Citations were fine; the retry is driven by the groundedness verdict
    feedback = generator.feedback[1]
    assert feedback.citation_validation.valid is True
    assert feedback.groundedness is ungrounded


def test_still_ungrounded_after_retry_escalates_to_human_review():
    context = make_context()
    generator = FakeDecisionGenerator([
        make_decision("APPROVE", [valid_citation]),
        make_decision("APPROVE", [valid_citation]),
    ])
    evaluator = ScriptedGroundednessEvaluator([ungrounded, ungrounded])

    result = decide_with_grounding_recovery(
        context,
        generator,
        evaluator,
    )

    assert result.decision.decision == "HUMAN_REVIEW"
    assert result.decision.citations == []
    assert result.attempts == 2
    assert result.recovered is False
    assert generator.call_count == 2
    assert evaluator.call_count == 2
    assert generator.contexts == [context, context]
