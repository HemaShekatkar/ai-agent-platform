from dataclasses import replace
from datetime import date

import pytest

from app.agents.onboarding import Citation, GroundedDecision
from app.agents.onboarding_agent import OnboardingAgent
from app.evals.groundedness import GroundednessResult
from app.gateway import GatewayDiagnostics, GatewayResponse, ModelSkip
from app.models.base import LLMResponse
from app.observability.conventions import (
    CitationAttributes,
    ContextAttributes,
    GroundednessAttributes,
    ModelAttributes,
    RetrievalAttributes,
)
from app.observability.models import ModelAttempt
from app.observability.tracing import Trace, Tracer
from app.retrieval.context import (
    ContextBudget,
    ContextBuilder,
    ContextItem,
    ContextResult,
)
from app.retrieval.models import DocumentChunk, RetrievalRequest, RetrievalResult
from app.retrieval.retriever import (
    RetrievalDiagnostics,
    InMemoryRetriever,
    RetrievalResponse,
    Retriever,
)
from app.routing.exceptions import AllProvidersFailedError
from app.routing.models import (
    Capability,
    CostTier,
    DataClassification,
    LatencyTier,
    ModelRequest,
)


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


def make_budget() -> ContextBudget:
    # Retrieval budget = 180 - 20 - 5 - 5 - 10 - 20 = 120
    return ContextBudget(
        model_context_limit=180,
        system_prompt_tokens=20,
        history_tokens=5,
        tool_definition_tokens=5,
        question_tokens=10,
        output_reservation_tokens=20,
    )


def make_results() -> list[RetrievalResult]:
    # Each section is "[chunk_id=chunk-X]\n" (19 chars) + text
    return [
        RetrievalResult(chunk=make_chunk("chunk-A", "A" * 60), score=0.95),
        RetrievalResult(chunk=make_chunk("chunk-B", "B" * 50), score=0.90),
        RetrievalResult(chunk=make_chunk("chunk-C", "C" * 10), score=0.80),
    ]


def make_request() -> RetrievalRequest:
    return RetrievalRequest(
        query="What security review is required?",
        region="US",
        document_type="SECURITY",
        caller_groups={"partner-security", "partner-onboarding"},
        as_of_date=date(2026, 10, 1),
        top_k=3,
    )


class StubRetriever(Retriever):

    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request):
        self.requests.append(request)
        return RetrievalResponse(
            results=self.results,
            diagnostics=RetrievalDiagnostics(
                corpus_count=10,
                eligible_count=len(self.results),
                retrieval_version="stub-v1",
                index_version="stub-index-v1",
                embedding_model_version="none",
            ),
        )


class FailingRetriever(Retriever):

    def retrieve(self, request):
        raise RuntimeError("vector store unavailable")


class UnusedGateway:
    # Retrieval and context tests never reach model generation
    def generate(self, request, prompt, system_prompt=None):
        raise AssertionError("gateway should not be called")


class UnusedGroundednessEvaluator:
    # Only grounding tests reach groundedness evaluation
    def evaluate(self, decision, context):
        raise AssertionError("groundedness evaluator should not be called")


def make_agent(
    retriever: Retriever | None = None,
    gateway=None,
    groundedness_evaluator=None,
) -> tuple[OnboardingAgent, Trace]:
    trace = Trace(trace_id="trace-123")
    builder = ContextBuilder(
        count_tokens=fake_token_counter,
    )
    agent = OnboardingAgent(
        retriever=retriever or StubRetriever(make_results()),
        context_builder=builder,
        gateway=gateway or UnusedGateway(),
        groundedness_evaluator=(
            groundedness_evaluator or UnusedGroundednessEvaluator()
        ),
        tracer=Tracer(trace),
    )
    return agent, trace


def span_named(trace: Trace, name: str):
    return next(span for span in trace.spans if span.name == name)


def test_retrieve_returns_retriever_response():
    results = make_results()
    retriever = StubRetriever(results)
    agent, _ = make_agent(retriever)
    request = make_request()

    response = agent.retrieve(request)

    assert response.results == results
    assert response.diagnostics.corpus_count == 10
    assert response.diagnostics.eligible_count == 3
    assert retriever.requests == [request]


def test_retrieve_records_retrieval_span():
    agent, trace = make_agent()

    agent.retrieve(make_request())

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "retrieval.search"
    assert span.status == "OK"
    assert span.latency_ms is not None


def test_retrieve_records_standard_retrieval_telemetry():
    agent, trace = make_agent()

    agent.retrieve(make_request())

    attributes = span_named(trace, "retrieval.search").attributes

    # The full attribute contract lives in tests/observability/test_conventions.py;
    # here we only check the agent wires the request and response into it
    assert attributes[RetrievalAttributes.TOP_K] == 3
    assert attributes[RetrievalAttributes.RETURNED_CHUNKS] == 3
    assert attributes[RetrievalAttributes.REGION] == "US"
    assert attributes[RetrievalAttributes.DOCUMENT_TYPE] == "SECURITY"
    assert attributes[RetrievalAttributes.CORPUS_COUNT] == 10
    assert attributes[RetrievalAttributes.ELIGIBLE_COUNT] == 3
    assert attributes[RetrievalAttributes.RETRIEVAL_VERSION] == "stub-v1"
    assert attributes[RetrievalAttributes.INDEX_VERSION] == "stub-index-v1"
    assert attributes[RetrievalAttributes.CALLER_GROUP_COUNT] == 2


def test_retrieve_records_error_span_and_reraises():
    agent, trace = make_agent(FailingRetriever())

    with pytest.raises(RuntimeError, match="vector store unavailable"):
        agent.retrieve(make_request())

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "retrieval.search"
    assert span.status == "ERROR"
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.latency_ms is not None


def test_build_context_returns_builder_result():
    agent, _ = make_agent()

    context = agent.build_context(make_results(), budget=make_budget())

    # A (79) fits; A+B (79 + 2 + 69 = 150) exceeds 120; A+C (79 + 2 + 29 = 110) fits
    assert context.included_chunk_ids == ["chunk-A", "chunk-C"]
    assert [item.chunk_id for item in context.excluded_chunks] == ["chunk-B"]
    assert context.budget_tokens == 120
    assert context.used_tokens == 110


def test_build_context_records_single_context_span():
    agent, trace = make_agent()

    agent.build_context(make_results(), budget=make_budget())

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "context.build"
    assert span.status == "OK"
    assert span.trace_id == "trace-123"
    assert span.latency_ms is not None
    assert span.latency_ms >= 0


def test_build_context_records_standard_context_telemetry():
    agent, trace = make_agent()

    context = agent.build_context(make_results(), budget=make_budget())

    attributes = trace.spans[0].attributes

    # The full attribute contract lives in tests/observability/test_conventions.py;
    # here we only check the agent wires the builder's real outputs into it
    assert attributes[ContextAttributes.RETRIEVAL_BUDGET_TOKENS] == context.budget_tokens
    assert attributes[ContextAttributes.USED_TOKENS] == context.used_tokens
    assert attributes[ContextAttributes.RETRIEVED_CHUNKS] == 3


class FailingContextBuilder:

    def build(self, results, budget):
        raise RuntimeError("context build failed")


def test_build_context_records_error_span_and_reraises():
    trace = Trace(trace_id="trace-123")
    agent = OnboardingAgent(
        retriever=StubRetriever(make_results()),
        context_builder=FailingContextBuilder(),
        gateway=UnusedGateway(),
        groundedness_evaluator=UnusedGroundednessEvaluator(),
        tracer=Tracer(trace),
    )

    with pytest.raises(RuntimeError, match="context build failed"):
        agent.build_context(make_results(), budget=make_budget())

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "context.build"
    assert span.status == "ERROR"
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.latency_ms is not None


def test_retrieval_and_context_spans_are_siblings_under_agent_run():
    agent, trace = make_agent()

    with agent.tracer.start_span("agent.run") as parent:
        response = agent.retrieve(make_request())
        agent.build_context(response.results, budget=make_budget())

    retrieval_span = span_named(trace, "retrieval.search")
    context_span = span_named(trace, "context.build")

    # context.build consumes retrieval's output but is not an operation
    # inside retrieval; both are stages orchestrated by the agent
    assert retrieval_span.parent_span_id == parent.span_id
    assert context_span.parent_span_id == parent.span_id
    assert context_span.parent_span_id != retrieval_span.span_id
    assert parent.parent_span_id is None


def test_retrieve_with_real_retriever():
    # Guards the Retriever contract end to end; the stub alone can drift
    # from what real retrievers return
    retriever = InMemoryRetriever([
        make_chunk("chunk-A", "security review required"),
        replace(make_chunk("chunk-EU", "security review required"), region="EU"),
    ])
    agent, trace = make_agent(retriever)

    response = agent.retrieve(
        RetrievalRequest(
            query="security review",
            region="US",
            document_type="SECURITY",
            caller_groups={"partner-onboarding"},
            as_of_date=date(2026, 10, 1),
        )
    )

    assert [result.chunk.id for result in response.results] == ["chunk-A"]

    attributes = span_named(trace, "retrieval.search").attributes

    assert attributes[RetrievalAttributes.CORPUS_COUNT] == 2
    assert attributes[RetrievalAttributes.ELIGIBLE_COUNT] == 1
    assert attributes[RetrievalAttributes.RETURNED_CHUNKS] == 1
    assert attributes[RetrievalAttributes.RETRIEVAL_VERSION] == "lexical-v1"
    assert attributes[RetrievalAttributes.INDEX_VERSION] == "in-memory-v1"
    assert attributes[RetrievalAttributes.EMBEDDING_MODEL_VERSION] == "none"


class StubGateway:

    def __init__(
        self,
        result: GatewayResponse | None = None,
        error: Exception | None = None,
    ):
        self.result = result
        self.error = error
        self.calls: list[tuple[ModelRequest, str]] = []

    def generate(self, request, prompt, system_prompt=None):
        self.calls.append((request, prompt))
        if self.error is not None:
            raise self.error
        return self.result

    def generate_structured(
        self,
        prompt,
        request,
        response_model,
        system_prompt=None,
    ):
        self.calls.append((request, prompt))
        self.received_response_model = response_model
        if self.error is not None:
            raise self.error
        return self.result


def make_model_request() -> ModelRequest:
    return ModelRequest(
        task="partner_onboarding",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10_000,
        data_classification=DataClassification.PII,
    )


def make_fallback_diagnostics() -> GatewayDiagnostics:
    # model-z skipped for budget, model-x failed over, model-y served
    return GatewayDiagnostics(
        routing_policy_version="interactive-v1",
        ranked_models=["model-z", "model-x", "model-y"],
        attempts=[
            ModelAttempt(
                model="model-x",
                latency_ms=40.0,
                success=False,
                error_type="TransientProviderError",
            ),
            ModelAttempt(model="model-y", latency_ms=90.0, success=True),
        ],
        skips=[
            ModelSkip(
                model="model-z",
                reason="LATENCY_BUDGET_EXCEEDED",
                remaining_budget_ms=300.0,
                expected_latency_ms=5000.0,
            ),
        ],
    )


def make_gateway_result() -> GatewayResponse[str]:
    return GatewayResponse(
        response=LLMResponse(
            content="Partners handling PII need an approved security review.",
            model="model-y",
            input_tokens=1200,
            output_tokens=85,
            latency_ms=90.0,
        ),
        diagnostics=make_fallback_diagnostics(),
    )


def test_generate_returns_gateway_response():
    result = make_gateway_result()
    gateway = StubGateway(result=result)
    agent, _ = make_agent(gateway=gateway)
    request = make_model_request()

    returned = agent.generate(prompt="assess partner", request=request)

    assert returned is result
    assert gateway.calls == [(request, "assess partner")]


def test_generate_records_model_span_with_routing_telemetry():
    agent, trace = make_agent(
        gateway=StubGateway(result=make_gateway_result()))

    agent.generate(prompt="assess partner", request=make_model_request())

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "model.generate"
    assert span.status == "OK"

    attributes = span.attributes

    assert attributes[ModelAttributes.SERVING_MODEL] == "model-y"
    assert attributes[ModelAttributes.PROMPT_VERSION] == OnboardingAgent.PROMPT_VERSION
    assert attributes[ModelAttributes.ROUTING_POLICY_VERSION] == "interactive-v1"
    assert attributes[ModelAttributes.INPUT_TOKENS] == 1200
    assert attributes[ModelAttributes.OUTPUT_TOKENS] == 85
    assert attributes[ModelAttributes.ATTEMPT_COUNT] == 2
    assert attributes[ModelAttributes.FALLBACK_USED] is True
    assert attributes[ModelAttributes.SKIP_COUNT] == 1


def test_generate_records_failure_diagnostics_and_reraises():
    error = AllProvidersFailedError(
        "All eligible providers failed.",
        diagnostics=GatewayDiagnostics(
            routing_policy_version="interactive-v1",
            ranked_models=["model-x", "model-y"],
            attempts=[
                ModelAttempt(
                    model="model-x",
                    latency_ms=40.0,
                    success=False,
                    error_type="TransientProviderError",
                ),
                ModelAttempt(
                    model="model-y",
                    latency_ms=55.0,
                    success=False,
                    error_type="TransientProviderError",
                ),
            ],
            skips=[],
        ),
    )
    agent, trace = make_agent(gateway=StubGateway(error=error))

    with pytest.raises(AllProvidersFailedError) as exc_info:
        agent.generate(prompt="assess partner", request=make_model_request())

    assert exc_info.value is error

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "model.generate"
    assert span.status == "ERROR"
    assert span.attributes["error.type"] == "AllProvidersFailedError"

    attributes = span.attributes

    assert attributes[ModelAttributes.PROMPT_VERSION] == OnboardingAgent.PROMPT_VERSION
    assert attributes[ModelAttributes.ROUTING_POLICY_VERSION] == "interactive-v1"
    assert attributes[ModelAttributes.ATTEMPT_COUNT] == 2
    assert attributes[ModelAttributes.FALLBACK_USED] is True
    assert attributes[ModelAttributes.SKIP_COUNT] == 0
    assert attributes[ModelAttributes.FAILURE_TYPE] == "AllProvidersFailedError"

    # Nothing served the request
    assert ModelAttributes.SERVING_MODEL not in attributes


def test_model_generate_span_is_child_of_agent_run():
    agent, trace = make_agent(
        gateway=StubGateway(result=make_gateway_result()))

    with agent.tracer.start_span("agent.run") as parent:
        agent.generate(prompt="assess partner", request=make_model_request())

    model_span = span_named(trace, "model.generate")

    assert model_span.parent_span_id == parent.span_id


def make_run_decision_result(
    citations: list[str] | None = None,
) -> GatewayResponse[GroundedDecision]:
    # chunk-A survives the context budget, so it is a valid citation in run()
    return GatewayResponse(
        response=LLMResponse(
            content=GroundedDecision(
                decision="APPROVE",
                reason="Security review is on file.",
                citations=["chunk-A"] if citations is None else citations,
            ),
            model="model-y",
            input_tokens=1200,
            output_tokens=85,
            latency_ms=90.0,
        ),
        diagnostics=make_fallback_diagnostics(),
    )


def make_run_agent(gateway, grounded_result: bool = True):
    return make_agent(
        gateway=gateway,
        groundedness_evaluator=StubGroundednessEvaluator(
            result=grounded(grounded_result),
        ),
    )


def test_run_creates_end_to_end_trace():
    agent, _ = make_run_agent(StubGateway(result=make_run_decision_result()))

    agent.run(
        retrieval_request=make_request(),
        model_request=make_model_request(),
        context_budget=make_budget(),
    )

    spans = agent.tracer.trace.spans
    by_name = {span.name: span for span in spans}

    root = by_name["agent.run"]

    assert len(spans) == 6
    assert root.parent_span_id is None
    assert root.status == "OK"

    assert set(by_name) == {
        "agent.run",
        "retrieval.search",
        "context.build",
        "model.generate",
        "validation.citation",
        "validation.groundedness",
    }

    # Each stage is orchestrated by the agent, not nested in the previous one
    for name in set(by_name) - {"agent.run"}:
        assert by_name[name].parent_span_id == root.span_id


def test_run_returns_validated_decision():
    result = make_run_decision_result()
    agent, _ = make_run_agent(StubGateway(result=result))

    decision = agent.run(
        retrieval_request=make_request(),
        model_request=make_model_request(),
        context_budget=make_budget(),
    )

    assert decision is result.response.content


def test_run_falls_back_to_human_review_when_validation_fails():
    # chunk-B was cut by the context budget, so citing it is invalid
    agent, _ = make_run_agent(
        StubGateway(result=make_run_decision_result(citations=["chunk-B"])),
    )

    decision = agent.run(
        retrieval_request=make_request(),
        model_request=make_model_request(),
        context_budget=make_budget(),
    )

    assert decision.decision == "HUMAN_REVIEW"
    assert decision.citations == []


def test_run_sends_budgeted_context_and_question_to_gateway():
    gateway = StubGateway(result=make_run_decision_result())
    agent, _ = make_run_agent(gateway)
    model_request = make_model_request()

    agent.run(
        retrieval_request=make_request(),
        model_request=model_request,
        context_budget=make_budget(),
    )

    assert len(gateway.calls) == 1
    assert gateway.received_response_model is GroundedDecision

    request, prompt = gateway.calls[0]

    assert request is model_request
    assert "What security review is required?" in prompt

    # chunk-B exceeds the 120-token budget; A and C fit
    assert "A" * 60 in prompt
    assert "C" * 10 in prompt
    assert "B" * 50 not in prompt


def test_run_marks_agent_run_failed_when_generation_fails():
    error = AllProvidersFailedError(
        "All eligible providers failed.",
        diagnostics=GatewayDiagnostics(
            routing_policy_version="interactive-v1",
            ranked_models=["model-x"],
            attempts=[
                ModelAttempt(
                    model="model-x",
                    latency_ms=40.0,
                    success=False,
                    error_type="TransientProviderError",
                ),
            ],
            skips=[],
        ),
    )
    agent, trace = make_agent(gateway=StubGateway(error=error))

    with pytest.raises(AllProvidersFailedError):
        agent.run(
            retrieval_request=make_request(),
            model_request=make_model_request(),
            context_budget=make_budget(),
        )

    assert span_named(trace, "retrieval.search").status == "OK"
    assert span_named(trace, "context.build").status == "OK"
    assert span_named(trace, "model.generate").status == "ERROR"
    assert span_named(trace, "agent.run").status == "ERROR"
    assert (
        span_named(trace, "agent.run").attributes["error.type"]
        == "AllProvidersFailedError"
    )


def test_build_prompt_places_evidence_before_question():
    agent, _ = make_agent()

    prompt = agent.build_prompt(
        question="Is a security review required?",
        context="[chunk_id=chunk-A]\nPII partners need a review.",
    )

    assert prompt.startswith("Use the supplied evidence")
    assert prompt.index("Evidence:") < prompt.index("[chunk_id=chunk-A]")
    assert prompt.index("[chunk_id=chunk-A]") < prompt.index("Question:")
    assert prompt.endswith("Is a security review required?")


INCLUDED_CHUNK_IDS = ["security-policy-17", "privacy-policy-42"]


def attribute_strings(attributes: dict) -> set[str]:
    # Flatten list-valued attributes so leaks inside lists are caught too
    strings = set()
    for value in attributes.values():
        if isinstance(value, (list, tuple, set)):
            strings.update(str(item) for item in value)
        else:
            strings.add(str(value))
    return strings


def test_validate_citations_accepts_valid_citations():
    agent, trace = make_agent()
    decision = GroundedDecision(
        decision="APPROVE",
        reason="Security review is on file.",
        citations=["security-policy-17", "privacy-policy-42"],
    )

    result = agent.validate_citations(
        decision=decision,
        included_chunk_ids=INCLUDED_CHUNK_IDS,
        citations=[
            Citation("security-policy-17", DataClassification.PUBLIC),
            Citation("privacy-policy-42", DataClassification.INTERNAL),
        ],
    )

    assert result.valid is True

    span = span_named(trace, "validation.citation")
    attributes = span.attributes

    assert span.status == "OK"
    assert attributes[CitationAttributes.VALID] is True
    assert attributes[CitationAttributes.DECISION] == "APPROVE"
    assert attributes[CitationAttributes.CITATION_COUNT] == 2
    assert attributes[CitationAttributes.PUBLIC_CITATIONS] == ["security-policy-17"]
    assert CitationAttributes.FAILURE_REASON not in attributes

    # Internal references are evidence for the decision, not telemetry
    assert "privacy-policy-42" not in attribute_strings(attributes)


def test_validate_citations_flags_missing_required_citations():
    agent, trace = make_agent()
    decision = GroundedDecision(
        decision="REJECT",
        reason="Partner lacks a security review.",
        citations=[],
    )

    result = agent.validate_citations(
        decision=decision,
        included_chunk_ids=INCLUDED_CHUNK_IDS,
        citations=[],
    )

    assert result.valid is False
    assert result.missing_required_citations is True

    attributes = span_named(trace, "validation.citation").attributes

    assert attributes[CitationAttributes.VALID] is False
    assert (
        attributes[CitationAttributes.FAILURE_REASON]
        == "MISSING_REQUIRED_CITATIONS"
    )
    assert attributes[CitationAttributes.CITATION_COUNT] == 0


def test_validate_citations_flags_unknown_citation():
    agent, trace = make_agent()
    decision = GroundedDecision(
        decision="APPROVE",
        reason="Security review is on file.",
        citations=["security-policy-17", "security-policy-999"],
    )

    result = agent.validate_citations(
        decision=decision,
        included_chunk_ids=INCLUDED_CHUNK_IDS,
        citations=[
            Citation("security-policy-17", DataClassification.PUBLIC),
            # Classified PUBLIC, but not in the context the model was given
            Citation("security-policy-999", DataClassification.PUBLIC),
        ],
    )

    assert result.valid is False
    assert result.invalid_citations == ["security-policy-999"]

    attributes = span_named(trace, "validation.citation").attributes

    assert attributes[CitationAttributes.VALID] is False
    assert attributes[CitationAttributes.FAILURE_REASON] == "INVALID_CITATION"

    # Invalid IDs can be model-fabricated; keep them out of telemetry
    assert "security-policy-999" not in attribute_strings(attributes)
    assert attributes[CitationAttributes.PUBLIC_CITATIONS] == ["security-policy-17"]


class StubGroundednessEvaluator:

    def __init__(
        self,
        result: GroundednessResult | None = None,
        error: Exception | None = None,
    ):
        self.result = result
        self.error = error
        self.calls: list[tuple[GroundedDecision, ContextResult]] = []

    def evaluate(self, decision, context):
        self.calls.append((decision, context))
        if self.error is not None:
            raise self.error
        return self.result


def make_grounding_context() -> ContextResult:
    return ContextResult(
        text="",
        items=[
            ContextItem(
                chunk_id="security-policy-17",
                text="PII partners require an approved security review.",
            ),
        ],
        included_chunk_ids=["security-policy-17"],
        excluded_chunks=[],
        budget_tokens=120,
        used_tokens=60,
    )


def make_grounded_decision(**overrides) -> GroundedDecision:
    values = {
        "decision": "APPROVE",
        "reason": "Security review is on file.",
        "citations": ["security-policy-17"],
    }
    values.update(overrides)
    return GroundedDecision(**values)


GROUNDING_CITATIONS = [
    Citation("security-policy-17", DataClassification.PUBLIC),
]


def grounded(value: bool) -> GroundednessResult:
    return GroundednessResult(
        grounded=value,
        score=0.9 if value else 0.2,
        reason="Evaluator rationale may quote evidence text.",
    )


def test_validate_groundedness_returns_evaluator_result():
    result = grounded(True)
    evaluator = StubGroundednessEvaluator(result=result)
    agent, _ = make_agent(groundedness_evaluator=evaluator)
    decision = make_grounded_decision()
    context = make_grounding_context()

    returned = agent.validate_groundedness(decision=decision, context=context)

    assert returned is result
    assert evaluator.calls == [(decision, context)]


def test_validate_groundedness_records_span():
    agent, trace = make_agent(
        groundedness_evaluator=StubGroundednessEvaluator(result=grounded(False)),
    )

    agent.validate_groundedness(
        decision=make_grounded_decision(),
        context=make_grounding_context(),
    )

    assert len(trace.spans) == 1

    span = trace.spans[0]

    assert span.name == "validation.groundedness"
    assert span.status == "OK"
    assert span.attributes[GroundednessAttributes.GROUNDED] is False

    # The evaluator's free-text reason can quote evidence; keep it out
    assert "Evaluator rationale may quote evidence text." not in (
        attribute_strings(span.attributes)
    )


def test_validate_groundedness_records_error_span_and_reraises():
    agent, trace = make_agent(
        groundedness_evaluator=StubGroundednessEvaluator(
            error=RuntimeError("evaluator unavailable"),
        ),
    )

    with pytest.raises(RuntimeError, match="evaluator unavailable"):
        agent.validate_groundedness(
            decision=make_grounded_decision(),
            context=make_grounding_context(),
        )

    span = span_named(trace, "validation.groundedness")

    assert span.status == "ERROR"
    assert span.attributes["error.type"] == "RuntimeError"


def test_validate_decision_skips_groundedness_when_citations_invalid():
    evaluator = StubGroundednessEvaluator(result=grounded(True))
    agent, trace = make_agent(groundedness_evaluator=evaluator)

    result = agent.validate_decision(
        decision=make_grounded_decision(citations=["security-policy-999"]),
        context=make_grounding_context(),
        citations=[],
    )

    assert result.valid is False
    assert result.groundedness is None
    assert result.citation_validation.invalid_citations == ["security-policy-999"]

    # Ungrounded citations fail fast; no evaluator (LLM) call is spent
    assert evaluator.calls == []
    assert [span.name for span in trace.spans] == ["validation.citation"]


def test_validate_decision_is_valid_when_cited_and_grounded():
    agent, trace = make_agent(
        groundedness_evaluator=StubGroundednessEvaluator(result=grounded(True)),
    )

    result = agent.validate_decision(
        decision=make_grounded_decision(),
        context=make_grounding_context(),
        citations=GROUNDING_CITATIONS,
    )

    assert result.valid is True
    assert result.citation_validation.valid is True
    assert result.groundedness.grounded is True
    assert sorted(span.name for span in trace.spans) == [
        "validation.citation",
        "validation.groundedness",
    ]


def test_validate_decision_is_invalid_when_cited_but_not_grounded():
    agent, _ = make_agent(
        groundedness_evaluator=StubGroundednessEvaluator(result=grounded(False)),
    )

    result = agent.validate_decision(
        decision=make_grounded_decision(),
        context=make_grounding_context(),
        citations=GROUNDING_CITATIONS,
    )

    assert result.valid is False
    assert result.citation_validation.valid is True
    assert result.groundedness.grounded is False


def test_validation_spans_are_siblings_under_parent():
    agent, trace = make_agent(
        groundedness_evaluator=StubGroundednessEvaluator(result=grounded(True)),
    )

    with agent.tracer.start_span("agent.run") as parent:
        agent.validate_decision(
            decision=make_grounded_decision(),
            context=make_grounding_context(),
            citations=GROUNDING_CITATIONS,
        )

    # Groundedness runs after citation validation, not inside it
    assert span_named(trace, "validation.citation").parent_span_id == parent.span_id
    assert span_named(trace, "validation.groundedness").parent_span_id == parent.span_id


def make_decision_result() -> GatewayResponse[GroundedDecision]:
    # Same serving model, tokens and diagnostics as make_gateway_result(),
    # so the two generation modes can be compared directly
    text_result = make_gateway_result()

    return GatewayResponse(
        response=LLMResponse(
            content=make_grounded_decision(),
            model=text_result.response.model,
            input_tokens=text_result.response.input_tokens,
            output_tokens=text_result.response.output_tokens,
            latency_ms=text_result.response.latency_ms,
        ),
        diagnostics=make_fallback_diagnostics(),
    )


def test_generate_decision_returns_typed_grounded_decision():
    gateway = StubGateway(result=make_decision_result())
    agent, _ = make_agent(gateway=gateway)
    request = make_model_request()

    result = agent.generate_decision(prompt="assess partner", request=request)

    assert isinstance(result.response.content, GroundedDecision)
    assert result.response.content.decision == "APPROVE"
    assert result.response.content.citations == ["security-policy-17"]

    assert gateway.received_response_model is GroundedDecision
    assert gateway.calls == [(request, "assess partner")]


def test_generate_decision_span_matches_text_generation_span():
    text_agent, text_trace = make_agent(
        gateway=StubGateway(result=make_gateway_result()),
    )
    decision_agent, decision_trace = make_agent(
        gateway=StubGateway(result=make_decision_result()),
    )

    text_agent.generate(prompt="assess partner", request=make_model_request())
    decision_agent.generate_decision(
        prompt="assess partner",
        request=make_model_request(),
    )

    text_span = span_named(text_trace, "model.generate")
    decision_span = span_named(decision_trace, "model.generate")

    assert len(decision_trace.spans) == 1
    assert decision_span.status == "OK"

    # Output type must not change the model/routing telemetry contract
    assert decision_span.attributes == text_span.attributes
    assert decision_span.attributes[ModelAttributes.SERVING_MODEL] == "model-y"
    assert (
        decision_span.attributes[ModelAttributes.ROUTING_POLICY_VERSION]
        == "interactive-v1"
    )
    assert decision_span.attributes[ModelAttributes.FALLBACK_USED] is True


def test_generate_retry_creates_retry_span():
    agent, trace = make_agent(gateway=StubGateway(result=make_decision_result()))

    agent.generate_retry(prompt="retry prompt", request=make_model_request())

    assert len(trace.spans) == 1

    span = trace.spans[0]

    # Distinct name so retries are countable separately from first attempts
    assert span.name == "model.generate.retry"
    assert span.status == "OK"


def test_generate_retry_returns_typed_grounded_decision():
    gateway = StubGateway(result=make_decision_result())
    agent, _ = make_agent(gateway=gateway)
    request = make_model_request()

    result = agent.generate_retry(prompt="retry prompt", request=request)

    assert isinstance(result.response.content, GroundedDecision)
    assert gateway.received_response_model is GroundedDecision
    assert gateway.calls == [(request, "retry prompt")]


def test_generate_retry_records_normal_model_attributes():
    first_agent, first_trace = make_agent(
        gateway=StubGateway(result=make_decision_result()),
    )
    retry_agent, retry_trace = make_agent(
        gateway=StubGateway(result=make_decision_result()),
    )

    first_agent.generate_decision(prompt="prompt", request=make_model_request())
    retry_agent.generate_retry(prompt="prompt", request=make_model_request())

    first_span = span_named(first_trace, "model.generate")
    retry_span = span_named(retry_trace, "model.generate.retry")

    # Only the span name differs; the model/routing contract is identical
    assert retry_span.attributes == first_span.attributes
    assert retry_span.attributes[ModelAttributes.SERVING_MODEL] == "model-y"
    assert (
        retry_span.attributes[ModelAttributes.ROUTING_POLICY_VERSION]
        == "interactive-v1"
    )


def test_generate_retry_preserves_failure_diagnostics_and_reraises():
    error = AllProvidersFailedError(
        "All eligible providers failed.",
        diagnostics=GatewayDiagnostics(
            routing_policy_version="interactive-v1",
            ranked_models=["model-x", "model-y"],
            attempts=[
                ModelAttempt(
                    model="model-x",
                    latency_ms=40.0,
                    success=False,
                    error_type="TransientProviderError",
                ),
                ModelAttempt(
                    model="model-y",
                    latency_ms=55.0,
                    success=False,
                    error_type="TransientProviderError",
                ),
            ],
            skips=[],
        ),
    )
    agent, trace = make_agent(gateway=StubGateway(error=error))

    with pytest.raises(AllProvidersFailedError) as exc_info:
        agent.generate_retry(prompt="retry prompt", request=make_model_request())

    assert exc_info.value is error

    span = span_named(trace, "model.generate.retry")
    attributes = span.attributes

    assert span.status == "ERROR"
    assert attributes["error.type"] == "AllProvidersFailedError"
    assert attributes[ModelAttributes.FAILURE_TYPE] == "AllProvidersFailedError"
    assert attributes[ModelAttributes.ROUTING_POLICY_VERSION] == "interactive-v1"
    assert attributes[ModelAttributes.ATTEMPT_COUNT] == 2
    assert attributes[ModelAttributes.FALLBACK_USED] is True
    assert ModelAttributes.SERVING_MODEL not in attributes


class SequencedGateway:
    # Returns one queued result per structured call: first attempt, then retry

    def __init__(self, results: list[GatewayResponse]):
        self.results = list(results)
        self.prompts: list[str] = []
        self.issued: list[GroundedDecision] = []

    def generate_structured(
        self,
        prompt,
        request,
        response_model,
        system_prompt=None,
    ):
        self.prompts.append(prompt)
        result = self.results.pop(0)
        self.issued.append(result.response.content)
        return result


class SequencedGroundednessEvaluator:

    def __init__(self, results: list[GroundednessResult]):
        self.results = list(results)
        self.calls: list[GroundedDecision] = []

    def evaluate(self, decision, context):
        self.calls.append(decision)
        return self.results.pop(0)


class CountingContextBuilder(ContextBuilder):

    def __init__(self):
        super().__init__(count_tokens=fake_token_counter)
        self.build_count = 0

    def build(self, results, budget):
        self.build_count += 1
        return super().build(results, budget)


def make_retry_agent(
    decisions: list[list[str]],
    grounded_results: list[bool],
):
    # Each entry in decisions is the citation list for one generation;
    # chunk-A survives the context budget, chunk-B does not
    trace = Trace(trace_id="trace-123")
    retriever = StubRetriever(make_results())
    builder = CountingContextBuilder()
    gateway = SequencedGateway(
        [make_run_decision_result(citations=citations) for citations in decisions]
    )
    evaluator = SequencedGroundednessEvaluator(
        [grounded(value) for value in grounded_results]
    )
    agent = OnboardingAgent(
        retriever=retriever,
        context_builder=builder,
        gateway=gateway,
        groundedness_evaluator=evaluator,
        tracer=Tracer(trace),
    )
    return agent, trace, retriever, builder, gateway, evaluator


def run_agent(agent: OnboardingAgent) -> GroundedDecision:
    return agent.run(
        retrieval_request=make_request(),
        model_request=make_model_request(),
        context_budget=make_budget(),
    )


def span_count(trace: Trace, name: str) -> int:
    return sum(1 for span in trace.spans if span.name == name)


def test_retry_reuses_retrieval_and_context_but_generates_twice():
    agent, trace, retriever, builder, gateway, _ = make_retry_agent(
        decisions=[["chunk-B"], ["chunk-A"]],
        grounded_results=[True],
    )

    run_agent(agent)

    # The retry corrects the model's output, not its evidence: retrieval
    # and context are computed once and the same context is reused
    assert len(retriever.requests) == 1
    assert builder.build_count == 1
    assert len(gateway.prompts) == 2

    assert span_count(trace, "retrieval.search") == 1
    assert span_count(trace, "context.build") == 1
    assert span_count(trace, "model.generate") == 1
    assert span_count(trace, "model.generate.retry") == 1

    # Both prompts carry the same budgeted evidence
    for prompt in gateway.prompts:
        assert "A" * 60 in prompt
        assert "B" * 50 not in prompt

    run_span = span_named(trace, "agent.run")
    assert span_named(trace, "model.generate.retry").parent_span_id == (
        run_span.span_id
    )


def test_first_valid_decision_is_returned_without_retry():
    agent, trace, _, _, gateway, _ = make_retry_agent(
        decisions=[["chunk-A"]],
        grounded_results=[True],
    )

    decision = run_agent(agent)

    assert decision is gateway.issued[0]
    assert decision.citations == ["chunk-A"]
    assert len(gateway.prompts) == 1
    assert span_count(trace, "model.generate.retry") == 0


def test_invalid_first_decision_returns_valid_retry_decision():
    agent, trace, _, _, gateway, _ = make_retry_agent(
        decisions=[["chunk-B"], ["chunk-A"]],
        grounded_results=[True],
    )

    decision = run_agent(agent)

    assert decision is gateway.issued[1]
    assert decision.citations == ["chunk-A"]
    assert len(gateway.prompts) == 2
    assert span_count(trace, "model.generate.retry") == 1


def test_invalid_retry_falls_back_to_human_review_after_one_retry():
    agent, trace, _, _, gateway, _ = make_retry_agent(
        decisions=[["chunk-B"], ["chunk-B"]],
        grounded_results=[],
    )

    decision = run_agent(agent)

    assert decision.decision == "HUMAN_REVIEW"
    assert decision.citations == []

    # Exactly one retry: no further generation after the retry fails
    assert len(gateway.prompts) == 2
    assert span_count(trace, "model.generate.retry") == 1


def test_invalid_citations_skip_groundedness_and_trigger_retry():
    agent, trace, _, _, gateway, evaluator = make_retry_agent(
        decisions=[["chunk-B"], ["chunk-A"]],
        grounded_results=[True],
    )

    decision = run_agent(agent)

    # The judge only saw the retry decision; the first never reached it
    assert evaluator.calls == [decision]
    assert span_count(trace, "validation.citation") == 2
    assert span_count(trace, "validation.groundedness") == 1

    assert "INVALID_CITATION" in gateway.prompts[1]
    assert "INVALID_CITATION" not in gateway.prompts[0]


def test_ungrounded_decision_retries_with_ungrounded_feedback():
    agent, trace, _, _, gateway, evaluator = make_retry_agent(
        decisions=[["chunk-A"], ["chunk-A"]],
        grounded_results=[False, True],
    )

    decision = run_agent(agent)

    # Both attempts cite chunk-A, so check identity: the retry's decision wins
    assert decision is gateway.issued[1]
    assert evaluator.calls == gateway.issued
    assert span_count(trace, "validation.groundedness") == 2
    assert span_count(trace, "model.generate.retry") == 1

    assert "UNGROUNDED" in gateway.prompts[1]
