from app.agents.onboarding import Citation, GroundedDecision
from app.evals.groundedness import GroundednessResult
from app.gateway import GatewayDiagnostics, GatewayResponse
from app.models.base import LLMResponse
from app.observability.tracing import Span
from app.retrieval.context import ContextBudget, ContextResult
from app.retrieval.models import RetrievalRequest
from app.retrieval.retriever import RetrievalResponse
from app.routing.models import DataClassification


class ContextAttributes:
    MODEL_LIMIT = "context.model_limit"
    SYSTEM_TOKENS = "context.system_tokens"
    HISTORY_TOKENS = "context.history_tokens"
    TOOL_TOKENS = "context.tool_tokens"
    QUESTION_TOKENS = "context.question_tokens"
    OUTPUT_RESERVATION_TOKENS = "context.output_reservation_tokens"

    RETRIEVAL_BUDGET_TOKENS = "context.retrieval_budget_tokens"
    USED_TOKENS = "context.used_tokens"

    RETRIEVED_CHUNKS = "context.retrieved_chunks"
    INCLUDED_CHUNKS = "context.included_chunks"
    EXCLUDED_BUDGET = "context.excluded_budget"


def record_context_attributes(
    span: Span,
    budget: ContextBudget,
    result: ContextResult,
    retrieved_count: int,
) -> None:

    span.attributes.update({
        ContextAttributes.MODEL_LIMIT:
            budget.model_context_limit,

        ContextAttributes.SYSTEM_TOKENS:
            budget.system_prompt_tokens,

        ContextAttributes.HISTORY_TOKENS:
            budget.history_tokens,

        ContextAttributes.TOOL_TOKENS:
            budget.tool_definition_tokens,

        ContextAttributes.QUESTION_TOKENS:
            budget.question_tokens,

        ContextAttributes.OUTPUT_RESERVATION_TOKENS:
            budget.output_reservation_tokens,

        ContextAttributes.RETRIEVAL_BUDGET_TOKENS:
            result.budget_tokens,

        ContextAttributes.USED_TOKENS:
            result.used_tokens,

        ContextAttributes.RETRIEVED_CHUNKS:
            retrieved_count,

        ContextAttributes.INCLUDED_CHUNKS:
            len(result.included_chunk_ids),

        ContextAttributes.EXCLUDED_BUDGET:
            sum(
                1
                for item in result.excluded_chunks
                if item.reason == "BUDGET_EXCEEDED"
            ),
    })


class RetrievalAttributes:
    TOP_K = "retrieval.top_k"
    RETURNED_CHUNKS = "retrieval.returned_chunks"

    REGION = "retrieval.filter.region"
    DOCUMENT_TYPE = "retrieval.filter.document_type"

    CALLER_GROUP_COUNT = "retrieval.filter.caller_group_count"

    CORPUS_COUNT = "retrieval.corpus_count"
    ELIGIBLE_COUNT = "retrieval.eligible_count"
    REGION = "retrieval.filter.region"
    DOCUMENT_TYPE = "retrieval.filter.document_type"
    CALLER_GROUP_COUNT = "retrieval.filter.caller_group_count"
    RETRIEVAL_VERSION = "retrieval.version"
    INDEX_VERSION = "retrieval.index.version"
    EMBEDDING_MODEL_VERSION = "retrieval.embedding_model.version"


class CitationAttributes:
    DECISION = "validation.citation.decision"
    CITATION_COUNT = "validation.citation.count"
    PUBLIC_CITATIONS = "validation.citation.public_citations"
    VALID = "validation.citation.valid"
    FAILURE_REASON = "validation.citation.failure_reason"


def record_retrieval_attributes(
    span: Span,
    request: RetrievalRequest,
    response: RetrievalResponse,
) -> None:

    diagnostics = response.diagnostics
    span.attributes.update({
        RetrievalAttributes.TOP_K:
            request.top_k,

        RetrievalAttributes.RETURNED_CHUNKS:
            len(response.results),

        RetrievalAttributes.REGION:
            request.region,

        RetrievalAttributes.DOCUMENT_TYPE:
            request.document_type,

        RetrievalAttributes.CALLER_GROUP_COUNT:
            len(request.caller_groups),

        RetrievalAttributes.CORPUS_COUNT:
            response.diagnostics.corpus_count,

        RetrievalAttributes.ELIGIBLE_COUNT:
            response.diagnostics.eligible_count,

        RetrievalAttributes.REGION:
            request.region,

        RetrievalAttributes.DOCUMENT_TYPE:
            request.document_type,

        RetrievalAttributes.CALLER_GROUP_COUNT:
            len(request.caller_groups),

        RetrievalAttributes.RETRIEVAL_VERSION:
            diagnostics.retrieval_version,

        RetrievalAttributes.INDEX_VERSION:
            diagnostics.index_version,

        RetrievalAttributes.EMBEDDING_MODEL_VERSION:
            diagnostics.embedding_model_version,
    })


class ModelAttributes:
    SERVING_MODEL = "model.serving_model"
    PROMPT_VERSION = "model.prompt.version"
    ROUTING_POLICY_VERSION = "model.routing_policy.version"

    INPUT_TOKENS = "model.input_tokens"
    OUTPUT_TOKENS = "model.output_tokens"


def record_model_attributes(
    span: Span,
    response: LLMResponse,
    prompt_version: str,
    routing_policy_version: str,
) -> None:

    span.attributes.update({
        ModelAttributes.SERVING_MODEL:
            response.model,

        ModelAttributes.PROMPT_VERSION:
            prompt_version,

        ModelAttributes.ROUTING_POLICY_VERSION:
            routing_policy_version,

        ModelAttributes.INPUT_TOKENS:
            response.input_tokens,

        ModelAttributes.OUTPUT_TOKENS:
            response.output_tokens,
    })


class ModelAttributes:
    SERVING_MODEL = "model.serving_model"
    PROMPT_VERSION = "model.prompt.version"
    ROUTING_POLICY_VERSION = "model.routing_policy.version"
    INPUT_TOKENS = "model.input_tokens"
    OUTPUT_TOKENS = "model.output_tokens"
    ATTEMPT_COUNT = "model.attempt_count"
    FALLBACK_USED = "model.fallback_used"
    SKIP_COUNT = "model.skip_count"
    FAILURE_TYPE = "model.failure_type"


def record_model_attributes(
    span: Span,
    response: GatewayResponse,
    prompt_version: str,
) -> None:
    diagnostics = response.diagnostics
    response = response.response
    span.attributes.update({
        ModelAttributes.SERVING_MODEL:
        response.model,

        ModelAttributes.PROMPT_VERSION:
        prompt_version,

        ModelAttributes.ROUTING_POLICY_VERSION:
        diagnostics.routing_policy_version,

        ModelAttributes.INPUT_TOKENS:
        response.input_tokens,

        ModelAttributes.OUTPUT_TOKENS:
        response.output_tokens,

        ModelAttributes.ATTEMPT_COUNT:
        len(diagnostics.attempts),

        ModelAttributes.FALLBACK_USED:
        len(diagnostics.attempts) > 1,

        ModelAttributes.SKIP_COUNT:
        len(diagnostics.skips),
    })


def record_gateway_failure_attributes(
    span: Span,
    diagnostics: GatewayDiagnostics,
    prompt_version: str,
    failure_type: str,
) -> None:
    span.attributes.update({
        ModelAttributes.PROMPT_VERSION:
            prompt_version,

        ModelAttributes.ROUTING_POLICY_VERSION:
            diagnostics.routing_policy_version,

        ModelAttributes.ATTEMPT_COUNT:
            len(diagnostics.attempts),

        ModelAttributes.FALLBACK_USED:
            len(diagnostics.attempts) > 1,

        ModelAttributes.SKIP_COUNT:
            len(diagnostics.skips),

        ModelAttributes.FAILURE_TYPE:
            failure_type,
    })


def record_citation_validation_attributes(
    span: Span,
    decision: GroundedDecision,
    citations: list[Citation],
    valid: bool,
    failure_reason: str | None = None,
    invalid_citations: list[str] | None = None,
) -> None:

    # Invalid references were not in the model's context and may be
    # fabricated, so they never go into telemetry even if marked PUBLIC
    excluded = set(invalid_citations or [])

    public_citations = [
        citation.reference
        for citation in citations
        if citation.classification == DataClassification.PUBLIC
        and citation.reference not in excluded
    ]

    span.attributes.update({
        CitationAttributes.DECISION:
            decision.decision,

        CitationAttributes.CITATION_COUNT:
            len(citations),

        CitationAttributes.PUBLIC_CITATIONS:
            public_citations,

        CitationAttributes.VALID:
            valid,
    })

    if failure_reason is not None:
        span.attributes[
            CitationAttributes.FAILURE_REASON
        ] = failure_reason


class GroundednessAttributes:
    GROUNDED = "validation.groundedness.grounded"


def record_groundedness_attributes(
    span: Span,
    result: GroundednessResult,
) -> None:
    span.attributes[
        GroundednessAttributes.GROUNDED
    ] = result.grounded
