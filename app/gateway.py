import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from app.models.base import BaseProvider, LLMResponse, StructuredT
from app.models.exceptions import TransientProviderError, PermanentProviderError
from app.routing.exceptions import ProviderNotFoundError, LatencyBudgetExceededError, AllProvidersFailedError
from app.routing.models import (
    ModelConfig,
    ModelRequest,
    RoutingPolicy,
)
from app.routing.router import rank_models
from app.observability.sink import TraceSink, NoOpTraceSink
from app.observability.models import GatewayTrace, ModelAttempt

T = TypeVar("T")

PROVIDER_TIMEOUT_BUFFER_MS = 100


class ModelGateway:

    def __init__(
        self,
        models: list[ModelConfig],
        providers: dict[str, BaseProvider],  # injected, keyed by model name
        policy: RoutingPolicy,
        trace_sink: TraceSink | None = None,
    ):
        self.models = models
        self.providers = providers
        self.policy = policy
        self.trace_sink = trace_sink or NoOpTraceSink()

    def generate(
        self,
        request: ModelRequest,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GatewayResponse[str]:
        return self._generate_with_fallback(
            request,
            lambda provider, timeout_ms: provider.generate(
                prompt,
                system_prompt=system_prompt,
                timeout_ms=timeout_ms,
            ),
        )

    def generate_structured(
        self,
        prompt: str,
        request: ModelRequest,
        response_model: type[StructuredT],
        system_prompt: str | None = None,
    ) -> GatewayResponse[StructuredT]:
        return self._generate_with_fallback(
            request,
            lambda provider, timeout_ms: provider.generate_structured(
                prompt,
                response_model=response_model,
                system_prompt=system_prompt,
                timeout_ms=timeout_ms,
            ),
        )

    def _generate_with_fallback(
        self,
        request: ModelRequest,
        call_provider: Callable[[BaseProvider, int], LLMResponse[T]],
    ) -> GatewayResponse[T]:
        # Routing, latency budget, fallback and tracing are shared by every
        # generation mode; only the provider call differs
        start = time.perf_counter()
        trace = GatewayTrace(
            trace_id=str(uuid.uuid4()),
            task=request.task,
            routing_policy_version=self.policy.version,
        )
        try:
            # Rank models (raises NoEligibleModelError if none qualify)
            ranked = rank_models(
                self.models,
                request,
                self.policy,
            )

            trace.ranked_models = [
                model.name for model in ranked
            ]

            last_error = None
            skipped_for_budget = 0

            # Create diagnostics BEFORE attempting providers
            diagnostics = GatewayDiagnostics(
                routing_policy_version=self.policy.version,
                ranked_models=[
                    model.name for model in ranked
                ],
                attempts=[],
                skips=[],
            )

            # Try each model in ranked order until one succeeds
            for model in ranked:

                # Time budget left for this attempt
                elapsed_ms = (time.perf_counter() - start) * 1000
                remaining_ms = request.timeout_ms - elapsed_ms
                provider_timeout_ms = max(
                    0,
                    int(remaining_ms - PROVIDER_TIMEOUT_BUFFER_MS),
                )

                if (
                    provider_timeout_ms <= 0
                    or model.expected_latency_ms > provider_timeout_ms
                ):
                    skipped_for_budget += 1
                    diagnostics.skips.append(
                        ModelSkip(
                            model=model.name,
                            reason="LATENCY_BUDGET_EXCEEDED",
                            # The budget the skip decision was made against,
                            # after reserving PROVIDER_TIMEOUT_BUFFER_MS
                            remaining_budget_ms=provider_timeout_ms,
                            expected_latency_ms=model.expected_latency_ms,
                        )
                    )
                    continue

                # Look up its provider in the registry
                provider = self.providers.get(model.name)
                if provider is None:
                    raise ProviderNotFoundError(
                        f"No provider registered for model '{model.name}'"
                    )

                # Call that provider and return its LLMResponse
                try:
                    attempt_start = time.perf_counter()
                    response = call_provider(provider, provider_timeout_ms)

                    attempt_latency_ms = (
                        time.perf_counter() - attempt_start
                    ) * 1000

                    trace.attempts.append(
                        ModelAttempt(
                            model=model.name,
                            latency_ms=attempt_latency_ms,
                            success=True,
                        )
                    )
                    trace.serving_model = model.name
                    trace.input_tokens = response.input_tokens
                    trace.output_tokens = response.output_tokens
                    trace.success = True
                    diagnostics.attempts.append(
                        ModelAttempt(
                            model=model.name,
                            success=True,
                            latency_ms=attempt_latency_ms,
                        )
                    )

                    return GatewayResponse(
                        response=response,
                        diagnostics=diagnostics,
                    )

                except TransientProviderError as e:
                    # Temporary failure: fall through to the next-ranked model
                    attempt_latency_ms = (
                        time.perf_counter() - attempt_start
                    ) * 1000

                    trace.attempts.append(
                        ModelAttempt(
                            model=model.name,
                            latency_ms=attempt_latency_ms,
                            success=False,
                            error_type=type(e).__name__,
                        )
                    )
                    last_error = e
                    diagnostics.attempts.append(
                        ModelAttempt(
                            model=model.name,
                            success=False,
                            latency_ms=attempt_latency_ms,
                            error_type=type(e).__name__,
                        )
                    )
                    continue

                except PermanentProviderError as e:
                    attempt_latency_ms = (
                        time.perf_counter() - attempt_start
                    ) * 1000

                    trace.attempts.append(
                        ModelAttempt(
                            model=model.name,
                            latency_ms=attempt_latency_ms,
                            success=False,
                            error_type=type(e).__name__,
                        )
                    )
                    raise

            # Every attempted model failed with a transient error, or the
            # latency budget ruled out the rest
            if skipped_for_budget > 0:
                raise LatencyBudgetExceededError(
                    f"Latency budget prevented attempting "
                    f"{skipped_for_budget} eligible model(s)",
                    diagnostics=diagnostics,
                ) from last_error

            raise AllProvidersFailedError(
                "All eligible providers failed.",
                diagnostics=diagnostics,
            )

        except Exception as e:
            # Record why the request as a whole failed, then re-raise
            trace.error_type = type(e).__name__
            raise

        finally:
            trace.total_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            try:
                self.trace_sink.emit(trace)
            except Exception:
                pass


@dataclass
class ModelSkip:
    model: str
    reason: str
    remaining_budget_ms: float
    expected_latency_ms: float


@dataclass
class GatewayDiagnostics:
    routing_policy_version: str
    ranked_models: list[str]
    attempts: list[ModelAttempt]
    skips: list[ModelSkip]


@dataclass
class GatewayResponse(Generic[T]):
    response: LLMResponse[T]
    diagnostics: GatewayDiagnostics
