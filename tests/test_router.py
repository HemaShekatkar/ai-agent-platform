from app.routing.models import (
    Capability,
    CostTier,
    DataClassification,
    LatencyTier,
    ModelConfig,
    ModelRequest,
    RoutingPolicy,
)

from app.routing.router import rank_models
import pytest
from app.routing.exceptions import NoEligibleModelError

model_a = ModelConfig(
    name="model-a",
    capability=Capability.HIGH,
    latency=LatencyTier.HIGH,
    cost=CostTier.HIGH,
    expected_latency_ms=5000,
    allowed_data={
        DataClassification.PUBLIC,
        DataClassification.PII,
    },
)

model_b = ModelConfig(
    name="model-b",
    capability=Capability.HIGH,
    latency=LatencyTier.LOW,
    cost=CostTier.LOW,
    expected_latency_ms=5000,
    allowed_data={
        DataClassification.PUBLIC,
    },
)

model_e = ModelConfig(
    name="model-e",
    capability=Capability.HIGH,
    latency=LatencyTier.LOW,
    cost=CostTier.MEDIUM,
    expected_latency_ms=5000,
    allowed_data={
        DataClassification.PUBLIC,
        DataClassification.PII,
    },
)
model_fast_cheap = ModelConfig(
    name="fast-cheap",
    capability=Capability.LOW,
    latency=LatencyTier.LOW,
    cost=CostTier.LOW,
    expected_latency_ms=5000,
    allowed_data={
        DataClassification.PUBLIC,
        DataClassification.PII,
    },
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

MODELS = [model_a, model_b, model_e, model_fast_cheap]

INTERACTIVE_POLICY = RoutingPolicy(
    version="interactive-v1",
    latency_weight=4.0,
    cost_weight=1.0,
)

BATCH_POLICY = RoutingPolicy(
    version="batch-v1",
    latency_weight=1.0,
    cost_weight=4.0,
)


def test_pii_model_is_never_routed_to_unapproved_model():
    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10_000,
        data_classification=DataClassification.PII,
    )

    ranked = rank_models(
        MODELS,
        request,
        INTERACTIVE_POLICY,
    )

    # model-b is the fastest and cheapest, but isn't approved for PII
    assert model_b not in ranked

    assert ranked

    assert all(
        DataClassification.PII in model.allowed_data
        for model in ranked
    )

    assert ranked[0] == model_e


def test_models_below_minimum_capability_are_never_routed():
    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10_000,
        data_classification=DataClassification.PII,
    )

    models = [
        model_a,
        model_e,
        model_fast_cheap,
    ]

    ranked = rank_models(
        models,
        request,
        INTERACTIVE_POLICY,
    )

    assert ranked

    assert all(
        model.capability.value >= request.minimum_capability.value
        for model in ranked
    )


def test_raises_when_no_model_satisfies_hard_constraints():
    # construct a request for which none of the models qualify

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10_000,
        data_classification=DataClassification.PII,
    )

    # model-b isn't approved for PII; fast-cheap is below HIGH capability
    models = [
        model_b,
        model_fast_cheap,
    ]

    with pytest.raises(NoEligibleModelError):
        rank_models(
            models,
            request,
            INTERACTIVE_POLICY,
        )


def test_routing_policy_changes_model_preference():

    request = ModelRequest(
        task="partner_risk_assessment",
        minimum_capability=Capability.HIGH,
        latency_requirement=LatencyTier.LOW,
        cost_tier=CostTier.LOW,
        timeout_ms=10_000,
        data_classification=DataClassification.PII,
    )

    models = [
        model_x,
        model_y,
    ]

    interactive_ranked = rank_models(
        models,
        request,
        INTERACTIVE_POLICY,
    )

    batch_ranked = rank_models(
        models,
        request,
        BATCH_POLICY,
    )

    assert interactive_ranked[0] == model_x
    assert batch_ranked[0] == model_y


def test_policies_with_different_weights_have_distinct_versions():
    # Traces record only the policy version, so each version must map to
    # exactly one weighting or routing decisions become unattributable
    assert INTERACTIVE_POLICY.version == "interactive-v1"
    assert BATCH_POLICY.version == "batch-v1"

    assert (
        INTERACTIVE_POLICY.latency_weight,
        INTERACTIVE_POLICY.cost_weight,
    ) != (
        BATCH_POLICY.latency_weight,
        BATCH_POLICY.cost_weight,
    )
    assert INTERACTIVE_POLICY.version != BATCH_POLICY.version
