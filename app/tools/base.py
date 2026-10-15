from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.tools.models import ToolDefinition, ToolRuntimeContext


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")


class BaseTool(ABC, Generic[InputT, OutputT]):

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Trusted metadata describing this tool."""
        ...

    @abstractmethod
    def execute(
        self,
        input: InputT,
        runtime: ToolRuntimeContext,
    ) -> OutputT:
        """Perform one attempt of the tool's business operation."""
        ...
