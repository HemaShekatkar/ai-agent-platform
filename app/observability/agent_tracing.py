from dataclasses import dataclass
from collections import Counter


@dataclass(frozen=True)
class AgentStep:
    iteration: int
    tool_name: str
    policy_version: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class TraceAnalysis:
    total_iterations: int
    total_tokens: int
    total_latency_ms: float
    repeated_tools: dict[str, int]
    policy_versions: set[str]
    has_policy_conflict: bool
    has_agent_loop: bool


def analyze_agent_trace(
    steps: list[AgentStep],
    max_iterations: int = 6,
    repeat_threshold: int = 3,
) -> TraceAnalysis:

    if max_iterations < 1 or repeat_threshold < 1:
        raise ValueError("Thresholds must be positive")

    for step in steps:
        if (
            step.iteration < 1
            or step.input_tokens < 0
            or step.output_tokens < 0
            or step.latency_ms < 0
        ):
            raise ValueError("Invalid trace data")

    tool_counts = Counter(
        step.tool_name for step in steps
    )

    repeated_tools = {
        tool: count
        for tool, count in tool_counts.items()
        if count >= repeat_threshold
    }

    policy_versions = {
        step.policy_version
        for step in steps
        if step.policy_version is not None
    }

    return TraceAnalysis(
        total_iterations=max(
            (step.iteration for step in steps),
            default=0,
        ),
        total_tokens=sum(
            step.input_tokens + step.output_tokens
            for step in steps
        ),
        total_latency_ms=sum(
            step.latency_ms for step in steps
        ),
        repeated_tools=repeated_tools,
        policy_versions=policy_versions,
        has_policy_conflict=len(policy_versions) > 1,
        has_agent_loop=(
            len(steps) >= max_iterations
            or bool(repeated_tools)
        ),
    )
