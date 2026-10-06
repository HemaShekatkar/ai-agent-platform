from datetime import date

from app.gateway import GatewayDiagnostics, GatewayResponse, ModelSkip
from app.models.base import LLMResponse
from app.observability.conventions import (
    ContextAttributes,
    ModelAttributes,
    RetrievalAttributes,
    record_context_attributes,
    record_gateway_failure_attributes,
    record_model_attributes,
    record_retrieval_attributes,
)
from app.observability.models import ModelAttempt
from app.observability.tracing import Span
from app.retrieval.context import (
    ContextBudget,
    ContextItem,
    ContextResult,
    ExcludedChunk,
)
from app.retrieval.models import (
    DocumentChunk,
    RetrievalRequest,
    RetrievalResult,
)
from app.retrieval.retriever import RetrievalDiagnostics, RetrievalResponse
from app.routing.models import DataClassification


def make_span() -> Span:
    return Span(
        trace_id="trace-123",
        span_id="span-1",
        parent_span_id=None,
        name="context.build",
        start_time_ms=100,
    )


def make_budget() -> ContextBudget:
    return ContextBudget(
        model_context_limit=180,
        system_prompt_tokens=20,
        history_tokens=5,
        tool_definition_tokens=5,
        question_tokens=10,
        output_reservation_tokens=20,
    )


def make_context(
    included_chunk_ids: list[str] | None = None,
    excluded_chunks: list[ExcludedChunk] | None = None,
    budget_tokens: int = 120,
    used_tokens: int = 110,
) -> ContextResult:
    included_chunk_ids = (
        ["chunk-A", "chunk-C"]
        if included_chunk_ids is None
        else included_chunk_ids
    )
    return ContextResult(
        text="",
        items=[
            ContextItem(chunk_id=chunk_id, text="")
            for chunk_id in included_chunk_ids
        ],
        included_chunk_ids=included_chunk_ids,
        excluded_chunks=(
            [ExcludedChunk(chunk_id="chunk-B", reason="BUDGET_EXCEEDED")]
            if excluded_chunks is None
            else excluded_chunks
        ),
        budget_tokens=budget_tokens,
        used_tokens=used_tokens,
    )


def standard_attribute_names() -> set[str]:
    return {
        value
        for name, value in vars(ContextAttributes).items()
        if not name.startswith("_")
    }


def test_attribute_names_are_stable():
    # Dashboards and alerts query these names; renaming one is a breaking change
    assert standard_attribute_names() == {
        "context.model_limit",
        "context.system_tokens",
        "context.history_tokens",
        "context.tool_tokens",
        "context.question_tokens",
        "context.output_reservation_tokens",
        "context.retrieval_budget_tokens",
        "context.used_tokens",
        "context.retrieved_chunks",
        "context.included_chunks",
        "context.excluded_budget",
    }


def test_records_budget_breakdown():
    span = make_span()

    record_context_attributes(
        span=span,
        budget=make_budget(),
        result=make_context(),
        retrieved_count=3,
    )

    assert span.attributes[ContextAttributes.MODEL_LIMIT] == 180
    assert span.attributes[ContextAttributes.SYSTEM_TOKENS] == 20
    assert span.attributes[ContextAttributes.HISTORY_TOKENS] == 5
    assert span.attributes[ContextAttributes.TOOL_TOKENS] == 5
    assert span.attributes[ContextAttributes.QUESTION_TOKENS] == 10
    assert span.attributes[ContextAttributes.OUTPUT_RESERVATION_TOKENS] == 20


def test_records_context_usage_and_chunk_counts():
    span = make_span()

    record_context_attributes(
        span=span,
        budget=make_budget(),
        result=make_context(budget_tokens=120, used_tokens=110),
        retrieved_count=3,
    )

    assert span.attributes[ContextAttributes.RETRIEVAL_BUDGET_TOKENS] == 120
    assert span.attributes[ContextAttributes.USED_TOKENS] == 110
    assert span.attributes[ContextAttributes.RETRIEVED_CHUNKS] == 3
    assert span.attributes[ContextAttributes.INCLUDED_CHUNKS] == 2
    assert span.attributes[ContextAttributes.EXCLUDED_BUDGET] == 1


def test_records_exactly_the_standard_attribute_set():
    span = make_span()

    record_context_attributes(
        span=span,
        budget=make_budget(),
        result=make_context(),
        retrieved_count=3,
    )

    assert set(span.attributes) == standard_attribute_names()


def test_excluded_budget_counts_only_budget_exceeded_reasons():
    span = make_span()

    record_context_attributes(
        span=span,
        budget=make_budget(),
        result=make_context(
            excluded_chunks=[
                ExcludedChunk(chunk_id="chunk-B", reason="BUDGET_EXCEEDED"),
                ExcludedChunk(chunk_id="chunk-D", reason="DUPLICATE"),
                ExcludedChunk(chunk_id="chunk-E", reason="BUDGET_EXCEEDED"),
            ],
        ),
        retrieved_count=5,
    )

    assert span.attributes[ContextAttributes.EXCLUDED_BUDGET] == 2


def test_records_zero_counts_for_empty_context():
    span = make_span()

    record_context_attributes(
        span=span,
        budget=make_budget(),
        result=make_context(
            included_chunk_ids=[],
            excluded_chunks=[],
            used_tokens=0,
        ),
        retrieved_count=0,
    )

    assert span.attributes[ContextAttributes.RETRIEVED_CHUNKS] == 0
    assert span.attributes[ContextAttributes.INCLUDED_CHUNKS] == 0
    assert span.attributes[ContextAttributes.EXCLUDED_BUDGET] == 0
    assert span.attributes[ContextAttributes.USED_TOKENS] == 0


def test_preserves_existing_span_attributes():
    span = make_span()
    span.attributes["request.id"] = "req-1"

    record_context_attributes(
        span=span,
        budget=make_budget(),
        result=make_context(),
        retrieved_count=3,
    )

    assert span.attributes["request.id"] == "req-1"


def make_request(**overrides) -> RetrievalRequest:
    values = {
        "query": "What security review is required?",
        "region": "EU",
        "document_type": "SECURITY",
        "caller_groups": {"partner-security", "partner-onboarding"},
        "as_of_date": date(2026, 10, 1),
        "top_k": 5,
    }

    values.update(overrides)

    return RetrievalRequest(**values)


def make_results(count: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=DocumentChunk(
                id=f"chunk-{i}",
                document_id="security-policy",
                text="PII partners require an approved security review.",
                document_type="SECURITY",
                region="EU",
                effective_date=date(2026, 1, 1),
                expiration_date=None,
                access_groups={"partner-security"},
                data_classification=DataClassification.PUBLIC,
            ),
            score=1.0 - i / 10,
        )
        for i in range(count)
    ]


def make_response(
    count: int,
    corpus_count: int = 10,
    eligible_count: int = 4,
) -> RetrievalResponse:
    return RetrievalResponse(
        results=make_results(count),
        diagnostics=RetrievalDiagnostics(
            corpus_count=corpus_count,
            eligible_count=eligible_count,
            retrieval_version="lexical-v1",
            index_version="in-memory-v1",
            embedding_model_version="none",
        ),
    )


def retrieval_attribute_names() -> set[str]:
    return {
        value
        for name, value in vars(RetrievalAttributes).items()
        if not name.startswith("_")
    }


def test_retrieval_attribute_names_are_stable():
    assert retrieval_attribute_names() == {
        "retrieval.top_k",
        "retrieval.returned_chunks",
        "retrieval.filter.region",
        "retrieval.filter.document_type",
        "retrieval.filter.caller_group_count",
        "retrieval.corpus_count",
        "retrieval.eligible_count",
        "retrieval.version",
        "retrieval.index.version",
        "retrieval.embedding_model.version",
    }


def test_records_retrieval_request_and_result_attributes():
    span = make_span()

    record_retrieval_attributes(
        span=span,
        request=make_request(),
        response=make_response(3),
    )

    assert span.attributes[RetrievalAttributes.TOP_K] == 5
    assert span.attributes[RetrievalAttributes.RETURNED_CHUNKS] == 3
    assert span.attributes[RetrievalAttributes.REGION] == "EU"
    assert span.attributes[RetrievalAttributes.DOCUMENT_TYPE] == "SECURITY"
    assert span.attributes[RetrievalAttributes.CALLER_GROUP_COUNT] == 2


def test_records_retrieval_diagnostics():
    # corpus -> eligible -> returned shows where chunks were lost:
    # to filters/ACLs (corpus vs eligible) or to top_k (eligible vs returned)
    span = make_span()

    record_retrieval_attributes(
        span=span,
        request=make_request(),
        response=make_response(3, corpus_count=10, eligible_count=3),
    )

    assert span.attributes[RetrievalAttributes.CORPUS_COUNT] == 10
    assert span.attributes[RetrievalAttributes.ELIGIBLE_COUNT] == 3
    assert span.attributes[RetrievalAttributes.RETURNED_CHUNKS] == 3


def test_records_exactly_the_standard_retrieval_attribute_set():
    span = make_span()

    record_retrieval_attributes(
        span=span,
        request=make_request(),
        response=make_response(3),
    )

    assert set(span.attributes) == retrieval_attribute_names()


def test_records_caller_group_count_not_group_names():
    # Group names can identify customers or teams; only the count is telemetry
    span = make_span()

    record_retrieval_attributes(
        span=span,
        request=make_request(caller_groups={"acme-corp-admins"}),
        response=make_response(1),
    )

    assert "acme-corp-admins" not in span.attributes.values()
    assert span.attributes[RetrievalAttributes.CALLER_GROUP_COUNT] == 1


def test_records_zero_returned_chunks_for_empty_results():
    span = make_span()

    record_retrieval_attributes(
        span=span,
        request=make_request(),
        response=make_response(0, corpus_count=10, eligible_count=0),
    )

    assert span.attributes[RetrievalAttributes.RETURNED_CHUNKS] == 0
    assert span.attributes[RetrievalAttributes.ELIGIBLE_COUNT] == 0
    assert span.attributes[RetrievalAttributes.TOP_K] == 5


def test_records_retrieval_provenance():
    span = make_span()

    response = RetrievalResponse(
        results=make_results(3),
        diagnostics=RetrievalDiagnostics(
            corpus_count=100,
            eligible_count=20,
            retrieval_version="lexical-v2",
            index_version="policy-index-17",
            embedding_model_version="none",
        ),
    )

    record_retrieval_attributes(
        span=span,
        request=make_request(),
        response=response,
    )

    assert (
        span.attributes[RetrievalAttributes.RETRIEVAL_VERSION]
        == "lexical-v2"
    )
    assert (
        span.attributes[RetrievalAttributes.INDEX_VERSION]
        == "policy-index-17"
    )
    assert (
        span.attributes[RetrievalAttributes.EMBEDDING_MODEL_VERSION]
        == "none"
    )


def make_llm_response(**overrides) -> LLMResponse[str]:
    values = {
        "content": "Partners handling PII need an approved security review.",
        "model": "claude-sonnet-5-5",
        "input_tokens": 1200,
        "output_tokens": 85,
        "latency_ms": 640.0,
    }

    values.update(overrides)

    return LLMResponse(**values)


def make_gateway_response(
    routing_policy_version: str = "routing-2026-10",
    **overrides,
) -> GatewayResponse[str]:
    return GatewayResponse(
        response=make_llm_response(**overrides),
        diagnostics=GatewayDiagnostics(
            routing_policy_version=routing_policy_version,
            ranked_models=["claude-sonnet-5-5"],
            attempts=[],
            skips=[],
        ),
    )


def model_attribute_names() -> set[str]:
    return {
        value
        for name, value in vars(ModelAttributes).items()
        if not name.startswith("_")
    }


# Success and failure spans record different subsets of ModelAttributes
MODEL_SUCCESS_ATTRIBUTES = {
    ModelAttributes.SERVING_MODEL,
    ModelAttributes.PROMPT_VERSION,
    ModelAttributes.ROUTING_POLICY_VERSION,
    ModelAttributes.INPUT_TOKENS,
    ModelAttributes.OUTPUT_TOKENS,
    ModelAttributes.ATTEMPT_COUNT,
    ModelAttributes.FALLBACK_USED,
    ModelAttributes.SKIP_COUNT,
}

MODEL_FAILURE_ATTRIBUTES = {
    ModelAttributes.PROMPT_VERSION,
    ModelAttributes.ROUTING_POLICY_VERSION,
    ModelAttributes.ATTEMPT_COUNT,
    ModelAttributes.FALLBACK_USED,
    ModelAttributes.SKIP_COUNT,
    ModelAttributes.FAILURE_TYPE,
}


def test_model_attribute_names_are_stable():
    assert model_attribute_names() == {
        "model.serving_model",
        "model.prompt.version",
        "model.routing_policy.version",
        "model.input_tokens",
        "model.output_tokens",
        "model.attempt_count",
        "model.fallback_used",
        "model.skip_count",
        "model.failure_type",
    }


def test_every_model_attribute_is_recorded_on_some_path():
    # Guards against adding a name to ModelAttributes that nothing records
    assert MODEL_SUCCESS_ATTRIBUTES | MODEL_FAILURE_ATTRIBUTES == (
        model_attribute_names()
    )


def test_records_model_response_and_version_attributes():
    span = make_span()

    record_model_attributes(
        span=span,
        response=make_gateway_response(),
        prompt_version="onboarding-prompt-v3",
    )

    assert span.attributes[ModelAttributes.SERVING_MODEL] == "claude-sonnet-5-5"
    assert span.attributes[ModelAttributes.PROMPT_VERSION] == "onboarding-prompt-v3"
    assert span.attributes[ModelAttributes.ROUTING_POLICY_VERSION] == "routing-2026-10"
    assert span.attributes[ModelAttributes.INPUT_TOKENS] == 1200
    assert span.attributes[ModelAttributes.OUTPUT_TOKENS] == 85


def test_records_serving_model_from_response():
    # After fallback the serving model can differ from the one first requested;
    # telemetry must reflect the model that actually answered
    span = make_span()

    record_model_attributes(
        span=span,
        response=make_gateway_response(model="fallback-model"),
        prompt_version="onboarding-prompt-v3",
    )

    assert span.attributes[ModelAttributes.SERVING_MODEL] == "fallback-model"


def test_records_exactly_the_model_success_attribute_set():
    span = make_span()

    record_model_attributes(
        span=span,
        response=make_gateway_response(),
        prompt_version="onboarding-prompt-v3",
    )

    assert set(span.attributes) == MODEL_SUCCESS_ATTRIBUTES


def test_does_not_record_response_content():
    # Generated text can contain customer data; it belongs in evals, not spans
    content = "Acme Corp's SSN list is stored in bucket acme-pii"
    span = make_span()

    record_model_attributes(
        span=span,
        response=make_gateway_response(content=content),
        prompt_version="onboarding-prompt-v3",
    )

    assert content not in span.attributes.values()



def test_routing_policy_version_comes_from_gateway_diagnostics():
    # The span records the policy the gateway actually routed with
    span = make_span()

    record_model_attributes(
        span=span,
        response=make_gateway_response(routing_policy_version="interactive-v1"),
        prompt_version="onboarding-prompt-v3",
    )

    assert span.attributes[ModelAttributes.ROUTING_POLICY_VERSION] == "interactive-v1"


def make_attempt(model: str, success: bool, error_type: str | None = None):
    return ModelAttempt(
        model=model,
        latency_ms=120.0,
        success=success,
        error_type=error_type,
    )


def make_skip(model: str) -> ModelSkip:
    return ModelSkip(
        model=model,
        reason="LATENCY_BUDGET_EXCEEDED",
        remaining_budget_ms=300.0,
        expected_latency_ms=2000.0,
    )


def make_gateway_diagnostics(
    attempts: list[ModelAttempt] | None = None,
    skips: list[ModelSkip] | None = None,
) -> GatewayDiagnostics:
    return GatewayDiagnostics(
        routing_policy_version="interactive-v1",
        ranked_models=["model-x", "model-y", "model-z"],
        attempts=[] if attempts is None else attempts,
        skips=[] if skips is None else skips,
    )


def test_records_gateway_failure_attributes():
    span = make_span()

    record_gateway_failure_attributes(
        span=span,
        diagnostics=make_gateway_diagnostics(
            attempts=[
                make_attempt("model-x", False, "TransientProviderError"),
                make_attempt("model-y", False, "TransientProviderError"),
            ],
            skips=[make_skip("model-z")],
        ),
        prompt_version="onboarding-prompt-v3",
        failure_type="AllProvidersFailedError",
    )

    assert span.attributes[ModelAttributes.PROMPT_VERSION] == "onboarding-prompt-v3"
    assert span.attributes[ModelAttributes.ROUTING_POLICY_VERSION] == "interactive-v1"
    assert span.attributes[ModelAttributes.ATTEMPT_COUNT] == 2
    assert span.attributes[ModelAttributes.FALLBACK_USED] is True
    assert span.attributes[ModelAttributes.SKIP_COUNT] == 1
    assert span.attributes[ModelAttributes.FAILURE_TYPE] == "AllProvidersFailedError"


def test_failure_with_single_attempt_reports_no_fallback():
    span = make_span()

    record_gateway_failure_attributes(
        span=span,
        diagnostics=make_gateway_diagnostics(
            attempts=[
                make_attempt("model-x", False, "PermanentProviderError"),
            ],
        ),
        prompt_version="onboarding-prompt-v3",
        failure_type="PermanentProviderError",
    )

    assert span.attributes[ModelAttributes.ATTEMPT_COUNT] == 1
    assert span.attributes[ModelAttributes.FALLBACK_USED] is False
    assert span.attributes[ModelAttributes.SKIP_COUNT] == 0


def test_failure_when_budget_skipped_every_model():
    # No provider was called, so there is nothing to fall back from
    span = make_span()

    record_gateway_failure_attributes(
        span=span,
        diagnostics=make_gateway_diagnostics(
            skips=[make_skip("model-x"), make_skip("model-y")],
        ),
        prompt_version="onboarding-prompt-v3",
        failure_type="LatencyBudgetExceededError",
    )

    assert span.attributes[ModelAttributes.ATTEMPT_COUNT] == 0
    assert span.attributes[ModelAttributes.FALLBACK_USED] is False
    assert span.attributes[ModelAttributes.SKIP_COUNT] == 2
    assert span.attributes[ModelAttributes.FAILURE_TYPE] == "LatencyBudgetExceededError"


def test_records_exactly_the_model_failure_attribute_set():
    # No model served the request, so serving model and token counts
    # must not appear on a failure span
    span = make_span()

    record_gateway_failure_attributes(
        span=span,
        diagnostics=make_gateway_diagnostics(
            attempts=[
                make_attempt("model-x", False, "TransientProviderError"),
            ],
        ),
        prompt_version="onboarding-prompt-v3",
        failure_type="AllProvidersFailedError",
    )

    assert set(span.attributes) == MODEL_FAILURE_ATTRIBUTES


def test_success_records_fallback_and_skip_counts():
    # A request can succeed after skipping and failing over; that should
    # be visible without opening the trace
    span = make_span()

    response = make_gateway_response()
    response.diagnostics.attempts = [
        make_attempt("model-x", False, "TransientProviderError"),
        make_attempt("model-y", True),
    ]
    response.diagnostics.skips = [make_skip("model-z")]

    record_model_attributes(
        span=span,
        response=response,
        prompt_version="onboarding-prompt-v3",
    )

    assert span.attributes[ModelAttributes.ATTEMPT_COUNT] == 2
    assert span.attributes[ModelAttributes.FALLBACK_USED] is True
    assert span.attributes[ModelAttributes.SKIP_COUNT] == 1
