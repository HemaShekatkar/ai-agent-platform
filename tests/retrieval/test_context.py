
import pytest

from app.retrieval.retriever import RetrievalRequest, RetrievalResult
from datetime import date

from app.retrieval.models import DocumentChunk
from app.retrieval.context import ContextBudget, ContextBuilder
from app.routing.models import DataClassification


def make_chunk(
    id: str = "chunk-1",
    text: str = "Default policy text",
    data_classification: DataClassification = DataClassification.INTERNAL,
) -> DocumentChunk:
    return DocumentChunk(
        id=id,
        document_id=f"doc-{id}",
        text=text,
        document_type="SECURITY",
        region="US",
        effective_date=date(2026, 1, 1),
        expiration_date=None,
        access_groups={"partner-onboarding"},
        data_classification=data_classification,
    )


def fake_token_counter(text: str) -> int:
    return len(text)


def test_skips_oversized_chunk_and_includes_later_chunk_that_fits():
    chunk_a = make_chunk(
        id="chunk-A",
        text="A" * 60,
    )

    chunk_b = make_chunk(
        id="chunk-B",
        text="B" * 50,
    )

    chunk_c = make_chunk(
        id="chunk-C",
        text="C" * 20,
    )

    results = [
        RetrievalResult(
            chunk=chunk_a,
            score=0.95,
        ),
        RetrievalResult(
            chunk=chunk_b,
            score=0.90,
        ),
        RetrievalResult(
            chunk=chunk_c,
            score=0.80,
        ),
    ]

    budget = ContextBudget(
        model_context_limit=180,
        system_prompt_tokens=20,
        history_tokens=0,
        tool_definition_tokens=0,
        question_tokens=10,
        output_reservation_tokens=20,
    )

    builder = ContextBuilder(
        count_tokens=fake_token_counter,
    )

    result = builder.build(results, budget=budget)

    assert [
        item.chunk_id
        for item in result.items
    ] == [
        "chunk-A",
        "chunk-C",
    ]

    assert len(result.excluded_chunks) == 1
    assert result.budget_tokens == 130
    assert result.excluded_chunks[0].chunk_id == "chunk-B"
    assert result.excluded_chunks[0].reason == "BUDGET_EXCEEDED"

    assert result.used_tokens <= result.budget_tokens
    assert result.items[0].text == "A" * 60
    assert result.items[1].text == "C" * 20


def test_build_rejects_budget_with_no_room_for_retrieval():
    # Fixed costs consume the whole model limit, leaving 0 retrieval tokens
    budget = ContextBudget(
        model_context_limit=100,
        system_prompt_tokens=40,
        history_tokens=20,
        tool_definition_tokens=10,
        question_tokens=10,
        output_reservation_tokens=20,
    )

    builder = ContextBuilder(count_tokens=fake_token_counter)

    with pytest.raises(ValueError, match="No token budget"):
        builder.build(
            [RetrievalResult(chunk=make_chunk(), score=0.9)],
            budget=budget,
        )
