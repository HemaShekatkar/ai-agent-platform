from app.agents.onboarding import GroundedDecision
from app.evals.groundedness import (
    GroundednessEvaluator,
    GroundednessResult,
)
from app.models.base import BaseProvider, LLMResponse
from app.retrieval.context import ContextItem, ContextResult


class FakeGroundednessProvider(BaseProvider):

    def __init__(self, result: GroundednessResult):
        super().__init__("fake-groundedness-judge")
        self.result = result
        self.call_count = 0
        self.last_prompt = None
        self.last_system_prompt = None
        self.last_response_model = None

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout_ms: int | None = None,
    ) -> LLMResponse[str]:
        raise NotImplementedError(
            "Groundedness evaluation uses generate_structured"
        )

    def generate_structured(
        self,
        prompt,
        response_model,
        system_prompt=None,
        timeout_ms=None,
    ):
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_response_model = response_model

        return LLMResponse(
            content=self.result,
            model=self.model,
            input_tokens=100,
            output_tokens=20,
            latency_ms=5.0,
        )


def make_context(items: list[ContextItem]) -> ContextResult:
    return ContextResult(
        text="\n\n".join(item.text for item in items),
        items=items,
        included_chunk_ids=[item.chunk_id for item in items],
        excluded_chunks=[],
        budget_tokens=1000,
        used_tokens=100,
    )


def test_cited_evidence_supports_approval():
    decision = GroundedDecision(
        decision="APPROVE",
        reason="Security review passed.",
        citations=["security-policy-17"],
    )

    context = make_context([
        ContextItem(
            chunk_id="security-policy-17",
            text="The partner's security review passed.",
        ),
    ])

    provider = FakeGroundednessProvider(
        GroundednessResult(
            grounded=True,
            score=0.95,
            reason="The cited evidence states the security review passed.",
        )
    )

    result = GroundednessEvaluator(provider).evaluate(
        decision,
        context,
    )

    assert result.grounded is True
    assert result.score == 0.95

    assert provider.call_count == 1
    assert provider.last_response_model is GroundednessResult
    assert "APPROVE" in provider.last_prompt
    assert "Security review passed." in provider.last_prompt
    assert "security-policy-17" in provider.last_prompt
    assert "The partner's security review passed." in provider.last_prompt


def prompt_section(prompt: str, tag: str) -> str:
    start = prompt.index(f"<{tag}>") + len(f"<{tag}>")
    end = prompt.index(f"</{tag}>")
    return prompt[start:end]


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def test_claim_supported_only_by_uncited_context_is_not_grounded():
    decision = GroundedDecision(
        decision="APPROVE",
        reason="Security passed AND metadata is valid.",
        citations=["security-policy-17"],
    )

    context = make_context([
        ContextItem(
            chunk_id="security-policy-17",
            text="The partner's security review passed.",
        ),
        ContextItem(
            chunk_id="metadata-policy-8",
            text="The partner's metadata is valid.",
        ),
    ])

    provider = FakeGroundednessProvider(
        GroundednessResult(
            grounded=False,
            score=0.4,
            reason=(
                "The metadata claim is not supported by the cited "
                "evidence; it appears only in uncited context."
            ),
        )
    )

    result = GroundednessEvaluator(provider).evaluate(
        decision,
        context,
    )

    assert result.grounded is False

    cited_evidence = prompt_section(
        provider.last_prompt,
        "CITED_EVIDENCE",
    )
    full_context = prompt_section(
        provider.last_prompt,
        "FULL_CONTEXT",
    )

    # Only the cited chunk is offered as supporting evidence
    assert "security-policy-17" in cited_evidence
    assert "metadata-policy-8" not in cited_evidence
    assert "The partner's metadata is valid." not in cited_evidence

    # The uncited chunk is still visible to the judge for diagnosis
    assert "security-policy-17" in full_context
    assert "metadata-policy-8" in full_context

    assert "Security passed AND metadata is valid." in provider.last_prompt

    assert (
        "evidence appearing only in the full context but not in the "
        "cited evidence does not make a claim grounded"
        in normalize(provider.last_system_prompt)
    )


def test_cited_evidence_contradicts_decision():
    decision = GroundedDecision(
        decision="APPROVE",
        reason="Security review passed.",
        citations=["security-policy-17"],
    )

    context = make_context([
        ContextItem(
            chunk_id="security-policy-17",
            text=(
                "The partner's security review is incomplete. "
                "Onboarding must not proceed."
            ),
        ),
    ])

    provider = FakeGroundednessProvider(
        GroundednessResult(
            grounded=False,
            score=0.0,
            reason=(
                "The cited evidence contradicts the approval decision."
            ),
        )
    )

    result = GroundednessEvaluator(provider).evaluate(
        decision,
        context,
    )

    assert result.grounded is False
    assert result.score == 0.0

    assert "security-policy-17" in provider.last_prompt
    assert "Onboarding must not proceed." in provider.last_prompt
