import time

from openai import OpenAI

from app.models.base import BaseProvider, LLMResponse, StructuredT

from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
)

from app.models.exceptions import (
    TransientProviderError,
    PermanentProviderError,
)

class OpenAIProvider(BaseProvider):

    def __init__(self, model: str, client: OpenAI | None = None):
        super().__init__(model)
        # Injectable so tests can pass a fake client without credentials;
        # the default reads OPENAI_API_KEY from the environment
        self.client = client or OpenAI()

    def generate(self, prompt: str, system_prompt: str | None = None, timeout_ms: int | None = None) -> LLMResponse[str]:
        timeout_seconds = (
            timeout_ms / 1000
            if timeout_ms is not None
            else None
        )

        start = time.perf_counter()

        try:
            response = self.client.with_options(
                timeout=timeout_seconds, max_retries=0
            ).responses.create(
                model=self.model,
                instructions=system_prompt,
                input=prompt,
            )
        except (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
        ) as e:
            # Temporary failures: another model/provider may succeed
            raise TransientProviderError(
                f"{self.model}: {type(e).__name__}: {e}"
            ) from e
        except (
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
        ) as e:
            # Request or credentials are wrong: retrying won't help
            raise PermanentProviderError(
                f"{self.model}: {type(e).__name__}: {e}"
            ) from e

        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=response.output_text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredT],
        system_prompt: str | None = None
    ) -> LLMResponse[StructuredT]:
        start = time.perf_counter()

        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=prompt,
            text_format=response_model,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=response.output_parsed,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
