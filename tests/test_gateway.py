from app.gateway import ModelGateway
from app.models.base import BaseProvider, LLMResponse
from app.models.exceptions import TransientProviderError, PermanentProviderError
from app.routing.router import rank_models
from app.routing.exceptions import LatencyBudgetExceededError, AllProvidersFailedError
from app.routing.models import (
    Capability,
    CostTier,
    DataClassification,
    LatencyTier,
    ModelConfig,
    ModelRequest,
    RoutingPolicy,
)
from app.observability.sink import TraceSink, GatewayTrace

import pytest
import trace
from pydantic import BaseModel


class FakeProvider(BaseProvider):

    def __init__(
        self,
        model: str,
        failure_type: str | None = None,
    ):
        super().__init__(model)
        self.failure_type = failure_type
        self.call_count = 0
        self.received_timeout_ms = None
        self.received_system_prompt = None
        self.received_response_model = None

    def _record_call_and_maybe_fail(self, timeout_ms, system_prompt):
        self.call_count += 1
        self.received_timeout_ms = timeout_ms
        self.received_system_prompt = system_prompt

        if self.failure_type == "transient":
            raise TransientProviderError(
                f"{self.model} transient failure"
            )

        if self.failure_type == "permanent":
            raise PermanentProviderError(
                f"{self.model} permanent failure"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout_ms: int | None = None
    ) -> LLMResponse[str]:
        self._record_call_and_maybe_fail(timeout_ms, system_prompt)

        return LLMResponse(
            content=f"fake response: {prompt}",
            model=self.model,
            input_tokens=10,
            output_tokens=5,
            latency_ms=25.0,
        )

    def generate_structured(
        self,
        prompt,
        response_model,
        system_prompt=None,
        timeout_ms=None,
    ):
        self._record_call_and_maybe_fail(timeout_ms, system_prompt)
        self.received_response_model = response_model

        return LLMResponse(
            content=response_model(
                risk_level="LOW",
                assessed_by=self.model,
            ),
            model=self.model,
            input_tokens=12,
            output_tokens=7,
            latency_ms=30.0,
        )


model_x = ModelConfig(
    name="model-x",
    capability=Capability.HIGH,
    latency=LatencyTier.LOW,
    cost=CostTier.HIGH,
    expected_latency_ms=5000,
    allowed_data={
        DataClassification.PUBLIC,
        DataClassification.PII,
    },
)

model_y = ModelConfig(
    name="model-y",
    capability=Capability.HIGH,
    latency=LatencyTier.HIGH,
    cost=CostTier.LOW,
    expected_latency_ms=5000,
    allowed_data={
        DataClassification.PUBLIC,
        DataClassification.PII,
    },
)

provider_fakeX = FakeProvider("provider-x")
provider_fakeY = FakeProvider("provider-y")


def test_gateway_invokes_provider_ranked_first():
    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10000,
        data_classification=DataClassification.PII,
    )

    # Latency-heavy policy: model-x scores 2 (cost penalty 2 x 1.0),
    # model-y scores 8 (latency penalty 2 x 4.0), so model-x wins
    latency_first_policy = RoutingPolicy(
        latency_weight=4.0,
        cost_weight=1.0,
    )

    gateway = ModelGateway(
        models=[model_y, model_x],
        providers={
            "model-x": FakeProvider("provider-x"),
            "model-y": provider_fakeY,
        },
        policy=latency_first_policy,
    )

    result = gateway.generate(
        request,
        prompt="assess partner risk",
    )

    # FakeProvider stamps its own name (not the ModelConfig name) on
    # response.model, so this proves provider-x was actually called
    assert result.response.model == "provider-x"
    assert result.response.content == "fake response: assess partner risk"


def test_gateway_falls_back_when_primary_fails_transiently():
    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        # Room for model-x's attempt plus the model-y fallback
        timeout_ms=10000,
        data_classification=DataClassification.PII,
    )

    # Latency-heavy policy: model-x scores 2 (cost penalty 2 x 1.0),
    # model-y scores 8 (latency penalty 2 x 4.0), so model-x wins
    latency_first_policy = RoutingPolicy(
        latency_weight=4.0,
        cost_weight=1.0,
    )

    provider_x = FakeProvider(
        "provider-x",
        failure_type="transient",
    )

    # Y would succeed if the gateway called it.
    provider_y = FakeProvider(
        "provider-y",
    )
    gateway = ModelGateway(
        models=[model_y, model_x],
        providers={
            "model-x": provider_x,
            "model-y": provider_y,
        },
        policy=latency_first_policy,
    )

    result = gateway.generate(
        request,
        prompt="assess partner risk",
    )
    # model-x ranked first but failed transiently, so the gateway
    # fell back to model-y
    assert result.response.model == "provider-y"


def test_gateway_does_not_fallback_on_permanent_failure():

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10000,
        data_classification=DataClassification.PII,
    )

    latency_first_policy = RoutingPolicy(
        latency_weight=4.0,
        cost_weight=1.0,
    )

    # X will be selected first, but permanently fails.
    provider_x = FakeProvider(
        "provider-x",
        failure_type="permanent",
    )

    # Y would succeed if the gateway called it.
    provider_y = FakeProvider(
        "provider-y",
    )

    gateway = ModelGateway(
        models=[model_y, model_x],
        providers={
            "model-x": provider_x,
            "model-y": provider_y,
        },
        policy=latency_first_policy,
    )

    with pytest.raises(PermanentProviderError):
        gateway.generate(
            request,
            prompt="assess partner risk",
        )

    assert provider_x.call_count == 1
    assert provider_y.call_count == 0


def test_transient_failure_with_insufficient_latency_budget():
    model_x_budget = ModelConfig(
        name="model-x",
        capability=Capability.HIGH,
        latency=LatencyTier.LOW,
        cost=CostTier.HIGH,
        allowed_data={
            DataClassification.PUBLIC,
            DataClassification.PII,
        },
        expected_latency_ms=50,
    )

    model_y_budget = ModelConfig(
        name="model-y",
        capability=Capability.HIGH,
        latency=LatencyTier.HIGH,
        cost=CostTier.LOW,
        allowed_data={
            DataClassification.PUBLIC,
            DataClassification.PII,
        },
        expected_latency_ms=2000,
    )

    provider_x = FakeProvider(
        "provider-x",
        failure_type="transient"
    )

    provider_y = FakeProvider(
        "provider-y",
    )
    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=500,
        data_classification=DataClassification.PII,
    )

    latency_first_policy = RoutingPolicy(
        latency_weight=4.0,
        cost_weight=1.0,
    )

    gateway = ModelGateway(
        models=[model_x_budget, model_y_budget],
        providers={
            "model-x": provider_x,
            "model-y": provider_y,
        },
        policy=latency_first_policy,
    )

    with pytest.raises(
        LatencyBudgetExceededError,
        match="Latency budget",
    ) as exc_info:
        gateway.generate(
            request,
            prompt="assess partner risk",
        )

    diagnostics = exc_info.value.diagnostics

    # model-x was attempted and failed; model-y never got a call
    assert [attempt.model for attempt in diagnostics.attempts] == ["model-x"]
    assert [skip.model for skip in diagnostics.skips] == ["model-y"]

    skip = diagnostics.skips[0]

    assert skip.reason == "LATENCY_BUDGET_EXCEEDED"
    assert skip.expected_latency_ms == 2000
    # 500ms request budget minus the provider timeout buffer, less elapsed time
    assert 0 <= skip.remaining_budget_ms <= 400
    assert skip.expected_latency_ms > skip.remaining_budget_ms
    assert provider_y.call_count == 0
    assert provider_x.call_count == 1
    assert provider_y.call_count == 0


def test_all_providers_failed_error():
    trace_sink = FakeTraceSink()
    model_x_budget = ModelConfig(
        name="model-x",
        capability=Capability.HIGH,
        latency=LatencyTier.LOW,
        cost=CostTier.HIGH,
        allowed_data={
            DataClassification.PUBLIC,
            DataClassification.PII,
        },
        expected_latency_ms=50,
    )

    model_y_budget = ModelConfig(
        name="model-y",
        capability=Capability.HIGH,
        latency=LatencyTier.HIGH,
        cost=CostTier.LOW,
        allowed_data={
            DataClassification.PUBLIC,
            DataClassification.PII,
        },
        expected_latency_ms=1000,
    )

    provider_x = FakeProvider(
        "provider-x",
        failure_type="transient",
    )

    provider_y = FakeProvider(
        "provider-y",
        failure_type="transient",
    )

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10000,
        data_classification=DataClassification.PII,
    )

    latency_first_policy = RoutingPolicy(
        latency_weight=4.0,
        cost_weight=1.0,
    )

    gateway = ModelGateway(
        models=[model_x_budget, model_y_budget],
        providers={
            "model-x": provider_x,
            "model-y": provider_y,
        },
        policy=latency_first_policy,
        trace_sink=trace_sink,
    )

    with pytest.raises(AllProvidersFailedError, match="All eligible providers failed."):
        gateway.generate(
            request,
            prompt="assess partner risk",
        )
    assert provider_x.call_count == 1
    assert provider_y.call_count == 1
    assert len(trace_sink.traces) == 1

    trace = trace_sink.traces[0]

    assert trace.success is False
    assert trace.error_type == "AllProvidersFailedError"
    assert trace.serving_model is None

    assert len(trace.attempts) == 2
    assert all(
        attempt.success is False
        for attempt in trace.attempts
    )

    assert trace.total_latency_ms > 0


def test_gateway_propagates_remaining_timeout_to_provider():
    provider_x = FakeProvider("provider-x")

    model_x = ModelConfig(
        name="model-x",
        capability=Capability.HIGH,
        latency=LatencyTier.LOW,
        cost=CostTier.LOW,
        allowed_data={DataClassification.PII},
        expected_latency_ms=100,
    )

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        data_classification=DataClassification.PII,
        timeout_ms=500,
    )

    gateway = ModelGateway(
        models=[model_x],
        providers={"model-x": provider_x},
        policy=RoutingPolicy(
            latency_weight=4.0,
            cost_weight=1.0,
        ),
    )

    gateway.generate(
        request,
        prompt="assess partner risk",
    )

    assert provider_x.received_timeout_ms is not None
    assert 0 < provider_x.received_timeout_ms <= 4900


class FakeTraceSink(TraceSink):
    def __init__(self):
        self.traces = []

    def emit(self, trace: GatewayTrace) -> None:
        self.traces.append(trace)


INTERACTIVE_POLICY = RoutingPolicy(
    latency_weight=4.0,
    cost_weight=1.0,
    version="interactive-v1",
)


def test_gateway_emits_trace_for_fallback():
    provider_x = FakeProvider(
        "provider-x",
        failure_type="transient",
    )

    provider_y = FakeProvider("provider-y")

    trace_sink = FakeTraceSink()

    gateway = ModelGateway(
        models=[model_x, model_y],
        providers={
            "model-x": provider_x,
            "model-y": provider_y,
        },
        policy=INTERACTIVE_POLICY,
        trace_sink=trace_sink,
    )

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        data_classification=DataClassification.PII,
        timeout_ms=10000,
    )

    result = gateway.generate(
        request,
        prompt="assess partner risk",
    )

    assert result.response.model == "provider-y"

    # Exactly one trace for the gateway request
    assert len(trace_sink.traces) == 1

    trace = trace_sink.traces[0]

    assert trace.success is True
    assert trace.serving_model == "model-y"
    assert trace.routing_policy_version == "interactive-v1"

    assert len(trace.attempts) == 2

    assert trace.attempts[0].model == "model-x"
    assert trace.attempts[0].success is False
    assert (
        trace.attempts[0].error_type
        == "TransientProviderError"
    )

    assert trace.attempts[1].model == "model-y"
    assert trace.attempts[1].success is True

    assert trace.total_latency_ms > 0


def test_gateway_returns_routing_diagnostics_with_response():
    provider_x = FakeProvider(
        "provider-x",
        failure_type="transient",
    )

    gateway = ModelGateway(
        models=[model_x, model_y],
        providers={
            "model-x": provider_x,
            "model-y": FakeProvider("provider-y"),
        },
        policy=INTERACTIVE_POLICY,
    )

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        data_classification=DataClassification.PII,
        timeout_ms=10000,
    )

    result = gateway.generate(
        request,
        prompt="assess partner risk",
    )

    diagnostics = result.diagnostics

    assert diagnostics.routing_policy_version == "interactive-v1"
    assert diagnostics.ranked_models == ["model-x", "model-y"]

    # Callers can see the fallback without a trace sink
    assert [attempt.model for attempt in diagnostics.attempts] == [
        "model-x",
        "model-y",
    ]
    assert [attempt.success for attempt in diagnostics.attempts] == [
        False,
        True,
    ]
    assert diagnostics.attempts[0].error_type == "TransientProviderError"


def test_gateway_records_no_skips_when_budget_allows_every_model():
    gateway = ModelGateway(
        models=[model_x, model_y],
        providers={
            "model-x": FakeProvider("provider-x"),
            "model-y": FakeProvider("provider-y"),
        },
        policy=INTERACTIVE_POLICY,
    )

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        data_classification=DataClassification.PII,
        timeout_ms=10000,
    )

    result = gateway.generate(
        request,
        prompt="assess partner risk",
    )

    assert result.diagnostics.skips == []


def test_gateway_reports_skipped_model_in_successful_response():
    # The top-ranked model is too slow for the budget, so the gateway skips it
    # and serves from the next model; the skip must still be visible
    slow_model = ModelConfig(
        name="slow-model",
        capability=Capability.HIGH,
        latency=LatencyTier.LOW,
        cost=CostTier.LOW,
        allowed_data={DataClassification.PII},
        expected_latency_ms=5000,
    )

    fast_model = ModelConfig(
        name="fast-model",
        capability=Capability.HIGH,
        latency=LatencyTier.MEDIUM,
        cost=CostTier.MEDIUM,
        allowed_data={DataClassification.PII},
        expected_latency_ms=100,
    )

    slow_provider = FakeProvider("slow-provider")

    gateway = ModelGateway(
        models=[slow_model, fast_model],
        providers={
            "slow-model": slow_provider,
            "fast-model": FakeProvider("fast-provider"),
        },
        policy=INTERACTIVE_POLICY,
    )

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.MEDIUM,
        cost_tier=CostTier.MEDIUM,
        data_classification=DataClassification.PII,
        timeout_ms=1000,
    )

    result = gateway.generate(
        request,
        prompt="assess partner risk",
    )

    assert result.diagnostics.ranked_models[0] == "slow-model"
    assert result.response.model == "fast-provider"
    assert slow_provider.call_count == 0

    assert len(result.diagnostics.skips) == 1

    skip = result.diagnostics.skips[0]

    assert skip.model == "slow-model"
    assert skip.reason == "LATENCY_BUDGET_EXCEEDED"
    assert skip.expected_latency_ms == 5000
    assert skip.remaining_budget_ms <= 900

    assert [attempt.model for attempt in result.diagnostics.attempts] == [
        "fast-model",
    ]


class RiskAssessment(BaseModel):
    risk_level: str
    assessed_by: str


def make_structured_request(timeout_ms: int = 10000) -> ModelRequest:
    return ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        data_classification=DataClassification.PII,
        timeout_ms=timeout_ms,
    )


def test_generate_structured_returns_parsed_content_from_top_ranked_model():
    provider_x = FakeProvider("provider-x")
    provider_y = FakeProvider("provider-y")

    gateway = ModelGateway(
        models=[model_y, model_x],
        providers={"model-x": provider_x, "model-y": provider_y},
        policy=INTERACTIVE_POLICY,
    )

    result = gateway.generate_structured(
        prompt="assess partner risk",
        request=make_structured_request(),
        response_model=RiskAssessment,
    )

    assert isinstance(result.response.content, RiskAssessment)
    assert result.response.content.assessed_by == "provider-x"
    assert provider_y.call_count == 0

    assert result.diagnostics.routing_policy_version == "interactive-v1"
    assert result.diagnostics.ranked_models == ["model-x", "model-y"]
    assert [attempt.model for attempt in result.diagnostics.attempts] == [
        "model-x",
    ]


def test_generate_structured_passes_schema_system_prompt_and_timeout():
    provider_x = FakeProvider("provider-x")

    gateway = ModelGateway(
        models=[model_x],
        providers={"model-x": provider_x},
        policy=INTERACTIVE_POLICY,
    )

    gateway.generate_structured(
        prompt="assess partner risk",
        request=make_structured_request(timeout_ms=10000),
        response_model=RiskAssessment,
        system_prompt="Return a risk assessment.",
    )

    assert provider_x.received_response_model is RiskAssessment
    assert provider_x.received_system_prompt == "Return a risk assessment."
    assert 0 < provider_x.received_timeout_ms <= 9900


def test_generate_structured_falls_back_on_transient_failure():
    provider_x = FakeProvider("provider-x", failure_type="transient")
    provider_y = FakeProvider("provider-y")

    gateway = ModelGateway(
        models=[model_x, model_y],
        providers={"model-x": provider_x, "model-y": provider_y},
        policy=INTERACTIVE_POLICY,
    )

    result = gateway.generate_structured(
        prompt="assess partner risk",
        request=make_structured_request(),
        response_model=RiskAssessment,
    )

    assert result.response.content.assessed_by == "provider-y"
    assert result.diagnostics.routing_policy_version == "interactive-v1"
    assert result.diagnostics.ranked_models == ["model-x", "model-y"]
    assert [
        (attempt.model, attempt.success, attempt.error_type)
        for attempt in result.diagnostics.attempts
    ] == [
        ("model-x", False, "TransientProviderError"),
        ("model-y", True, None),
    ]


def test_generate_structured_does_not_fall_back_on_permanent_failure():
    provider_x = FakeProvider("provider-x", failure_type="permanent")
    provider_y = FakeProvider("provider-y")

    gateway = ModelGateway(
        models=[model_x, model_y],
        providers={"model-x": provider_x, "model-y": provider_y},
        policy=INTERACTIVE_POLICY,
    )

    with pytest.raises(PermanentProviderError):
        gateway.generate_structured(
            prompt="assess partner risk",
            request=make_structured_request(),
            response_model=RiskAssessment,
        )

    assert provider_y.call_count == 0


def test_generate_structured_reports_budget_skips():
    provider_x = FakeProvider("provider-x")

    gateway = ModelGateway(
        models=[model_x],
        providers={"model-x": provider_x},
        policy=INTERACTIVE_POLICY,
    )

    # model-x expects 5000ms; a 1000ms request leaves no room for it
    with pytest.raises(LatencyBudgetExceededError) as exc_info:
        gateway.generate_structured(
            prompt="assess partner risk",
            request=make_structured_request(timeout_ms=1000),
            response_model=RiskAssessment,
        )

    assert provider_x.call_count == 0

    assert exc_info.value.diagnostics.routing_policy_version == "interactive-v1"

    skips = exc_info.value.diagnostics.skips

    assert [skip.model for skip in skips] == ["model-x"]
    assert skips[0].reason == "LATENCY_BUDGET_EXCEEDED"


def test_generate_structured_emits_trace():
    trace_sink = FakeTraceSink()

    gateway = ModelGateway(
        models=[model_x, model_y],
        providers={
            "model-x": FakeProvider("provider-x", failure_type="transient"),
            "model-y": FakeProvider("provider-y"),
        },
        policy=INTERACTIVE_POLICY,
        trace_sink=trace_sink,
    )

    gateway.generate_structured(
        prompt="assess partner risk",
        request=make_structured_request(),
        response_model=RiskAssessment,
    )

    assert len(trace_sink.traces) == 1

    emitted = trace_sink.traces[0]

    assert emitted.success is True
    assert emitted.serving_model == "model-y"
    assert emitted.routing_policy_version == "interactive-v1"
    assert emitted.input_tokens == 12
    assert emitted.output_tokens == 7
    assert [attempt.model for attempt in emitted.attempts] == [
        "model-x",
        "model-y",
    ]
