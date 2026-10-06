from app.routing.models import ModelConfig, ModelRequest, RoutingPolicy

from app.routing.exceptions import NoEligibleModelError


def allows_data(model: ModelConfig, request: ModelRequest) -> bool:
    """Is request.data_classification allowed by the model?"""
    return request.data_classification in model.allowed_data


def meets_capability(model: ModelConfig, request: ModelRequest) -> bool:
    """Is model.capability >= request.minimum_capability?"""
    # Enum members aren't orderable, so compare their underlying values.
    return model.capability.value >= request.minimum_capability.value


def is_eligible(model: ModelConfig, request: ModelRequest) -> bool:
    return (
        allows_data(model, request)
        and meets_capability(model, request)

    )


def rank_models(models: list[ModelConfig],
                request: ModelRequest,
                policy: RoutingPolicy) -> list[ModelConfig]:
    """Among eligible models, prefer lower latency first, then lower cost."""
    eligible = [m for m in models if is_eligible(m, request)]

    if not eligible:
        raise NoEligibleModelError(
            f"No eligible model found for task '{request.task}'"
        )

    def routing_score(model: ModelConfig):
        latency_penalty = max(
            0,
            model.latency.value - request.latency_requirement.value
        )

        cost_penalty = max(
            0,
            model.cost.value - request.cost_tier.value
        )

        score = (
            policy.latency_weight * latency_penalty
            + policy.cost_weight * cost_penalty
        )
        return score

    return sorted(
        eligible,
        key=lambda model: (
            routing_score(model),  # First: satisfy request
            model.cost.value,      # Tie: prefer cheaper
            model.latency.value,   # Tie again: prefer faster
        )
    )
