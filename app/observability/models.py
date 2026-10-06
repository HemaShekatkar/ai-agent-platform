from dataclasses import dataclass, field


@dataclass
class ModelAttempt:
    model: str
    latency_ms: float
    success: bool
    error_type: str | None = None


@dataclass
class GatewayTrace:
    trace_id: str
    task: str
    routing_policy_version: str

    # Filled in as the request runs; a trace starts as not-yet-successful
    success: bool = False
    error_type: str | None = None

    ranked_models: list[str] = field(default_factory=list)
    attempts: list[ModelAttempt] = field(default_factory=list)

    serving_model: str | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0
