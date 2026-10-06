from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

# Response content: plain text by default, or a parsed BaseModel subclass
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")
StructuredT = TypeVar("StructuredT", bound=BaseModel)


@dataclass
class LLMResponse(Generic[T]):
    content: T
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class BaseProvider(ABC):

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, timeout_ms: int | None = None) -> LLMResponse[str]:
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_model: type[StructuredT],
        system_prompt: str | None = None,
        timeout_ms: int | None = None
    ) -> LLMResponse[StructuredT]:
        ...
