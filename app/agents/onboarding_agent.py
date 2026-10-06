from collections.abc import Callable
from typing import TypeVar

from app.agents.onboarding import (
    Citation,
    CitationValidationResult,
    GroundedDecision,
    GroundingValidationResult,
    validate_decision_citations,
)
from app.evals.groundedness import GroundednessEvaluator, GroundednessResult
from app.gateway import GatewayResponse, ModelGateway
from app.observability.conventions import (
    record_citation_validation_attributes,
    record_context_attributes,
    record_gateway_failure_attributes,
    record_groundedness_attributes,
    record_model_attributes,
    record_retrieval_attributes,
)
from app.observability.tracing import Tracer
from app.retrieval.context import ContextBudget, ContextBuilder, ContextResult
from app.retrieval.models import RetrievalResult, RetrievalRequest
from app.retrieval.retriever import Retriever, RetrievalResponse
from app.routing.exceptions import (
    AllProvidersFailedError,
    LatencyBudgetExceededError,
)
from app.routing.models import ModelRequest

T = TypeVar("T")


class OnboardingAgent:

    PROMPT_VERSION = "onboarding-prompt-v3"

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        gateway: ModelGateway,
        groundedness_evaluator: GroundednessEvaluator,
        tracer: Tracer,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.gateway = gateway
        self.groundedness_evaluator = groundedness_evaluator
        self.tracer = tracer

    def generate(
        self,
        prompt: str,
        request: ModelRequest,
    ) -> GatewayResponse[str]:
        return self._traced_generation(
            lambda: self.gateway.generate(
                prompt=prompt,
                request=request,
            )
        )

    def generate_decision(
        self,
        prompt: str,
        request: ModelRequest,
    ) -> GatewayResponse[GroundedDecision]:
        return self._traced_generation(
            lambda: self.gateway.generate_structured(
                prompt=prompt,
                request=request,
                response_model=GroundedDecision,
            )
        )

    def _traced_generation(
        self,
        call_gateway: Callable[[], GatewayResponse[T]],
        span_name: str = "model.generate",
    ) -> GatewayResponse[T]:
        # Every generation path shares one span contract, so model
        # telemetry is identical regardless of output type or retry
        with self.tracer.start_span(span_name) as span:
            try:
                result = call_gateway()

            except (AllProvidersFailedError, LatencyBudgetExceededError) as exc:
                # diagnostics is optional on LatencyBudgetExceededError
                if exc.diagnostics is not None:
                    record_gateway_failure_attributes(
                        span=span,
                        diagnostics=exc.diagnostics,
                        prompt_version=self.PROMPT_VERSION,
                        failure_type=type(exc).__name__,
                    )
                raise

            record_model_attributes(
                span=span,
                response=result,
                prompt_version=self.PROMPT_VERSION,
            )

            return result

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:

        with self.tracer.start_span("retrieval.search") as span:

            response = self.retriever.retrieve(request)

            record_retrieval_attributes(
                span=span,
                request=request,
                response=response,
            )

            return response

    def build_context(
        self,
        results: list[RetrievalResult],
        budget: ContextBudget,
    ) -> ContextResult:

        with self.tracer.start_span("context.build") as span:

            result = self.context_builder.build(
                results=results,
                budget=budget,
            )

            record_context_attributes(
                span=span,
                budget=budget,
                result=result,
                retrieved_count=len(results),
            )

            return result

    def run(
        self,
        retrieval_request: RetrievalRequest,
        model_request: ModelRequest,
        context_budget: ContextBudget,
    ) -> GroundedDecision:

        with self.tracer.start_span("agent.run"):
            retrieval_response = self.retrieve(retrieval_request)

            context_result = self.build_context(
                retrieval_response.results,
                budget=context_budget,
            )

            prompt = self.build_prompt(
                question=retrieval_request.query,
                context=context_result.text,
            )

            generation_result = self.generate_decision(
                prompt=prompt,
                request=model_request,
            )

            decision = generation_result.response.content

            citations = self.resolve_citations(
                decision=decision,
                results=retrieval_response.results,
                context=context_result,
            )

            validation_result = self.validate_decision(
                decision=decision,
                context=context_result,
                citations=citations,
            )
            if validation_result.valid:
                return decision

            failure_reason = self._validation_failure_reason(
                validation_result
            )

            retry_prompt = self.build_retry_prompt(
                question=retrieval_request.query,
                context=context_result.text,
                failure_reason=failure_reason,
            )

            retry_result = self.generate_retry(
                prompt=retry_prompt,
                request=model_request,
            )

            retry_decision = retry_result.response.content

            retry_citations = self.resolve_citations(
                decision=retry_decision,
                results=retrieval_response.results,
                context=context_result,
            )

            retry_validation = self.validate_decision(
                decision=retry_decision,
                context=context_result,
                citations=retry_citations,
            )

            if retry_validation.valid:
                return retry_decision

            return GroundedDecision(
                decision="HUMAN_REVIEW",
                reason="Decision failed validation after retry.",
                citations=[],
            )

    @staticmethod
    def _validation_failure_reason(
        result: GroundingValidationResult,
    ) -> str:

        if result.citation_validation.missing_required_citations:
            return "MISSING_REQUIRED_CITATIONS"

        if result.citation_validation.invalid_citations:
            return "INVALID_CITATION"

        if (
            result.groundedness is not None
            and not result.groundedness.grounded
        ):
            return "UNGROUNDED"

        return "VALIDATION_FAILED"

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        return f"""
Use the supplied evidence to answer the onboarding question.

Evidence:
{context}

Question:
{question}
""".strip()

    def validate_citations(
        self,
        decision: GroundedDecision,
        included_chunk_ids: list[str],
        citations: list[Citation],
    ) -> CitationValidationResult:

        with self.tracer.start_span("validation.citation") as span:
            result = validate_decision_citations(
                decision=decision,
                included_chunk_ids=included_chunk_ids,
            )

            record_citation_validation_attributes(
                span=span,
                decision=decision,
                citations=citations,
                valid=result.valid,
                failure_reason=self._citation_failure_reason(result),
                invalid_citations=result.invalid_citations,
            )

            return result

    @staticmethod
    def _citation_failure_reason(
        result: CitationValidationResult,
    ) -> str | None:

        if result.missing_required_citations:
            return "MISSING_REQUIRED_CITATIONS"

        if result.invalid_citations:
            return "INVALID_CITATION"

        return None

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

    def validate_decision(
        self,
        decision: GroundedDecision,
        context: ContextResult,
        citations: list[Citation],
    ) -> GroundingValidationResult:

        citation_result = self.validate_citations(
            decision=decision,
            included_chunk_ids=context.included_chunk_ids,
            citations=citations,
        )

        if not citation_result.valid:
            return GroundingValidationResult(
                valid=False,
                citation_validation=citation_result,
                groundedness=None,
            )

        groundedness = self.validate_groundedness(
            decision=decision,
            context=context,
        )

        return GroundingValidationResult(
            valid=groundedness.grounded,
            citation_validation=citation_result,
            groundedness=groundedness,
        )

    def resolve_citations(
        self,
        decision: GroundedDecision,
        results: list[RetrievalResult],
        context: ContextResult,
    ) -> list[Citation]:

        included_ids = set(context.included_chunk_ids)

        chunks_by_id = {
            result.chunk.id: result.chunk
            for result in results
            if result.chunk.id in included_ids
        }

        return [
            Citation(
                reference=reference,
                classification=chunks_by_id[reference].data_classification,
            )
            for reference in decision.citations
            if reference in chunks_by_id
        ]

    def generate_retry(
        self,
        prompt: str,
        request: ModelRequest,
    ) -> GatewayResponse[GroundedDecision]:
        return self._traced_generation(
            lambda: self.gateway.generate_structured(
                prompt=prompt,
                request=request,
                response_model=GroundedDecision,
            ),
            span_name="model.generate.retry",
        )

    def build_retry_prompt(
        self,
        question: str,
        context: str,
        failure_reason: str,
    ) -> str:
        return f"""
Your previous decision failed validation.

Validation failure:
{failure_reason}

Re-evaluate the decision using ONLY the supplied evidence.

Evidence:
{context}

Question:
{question}

Do not cite evidence that is not present in the supplied context.
""".strip()
