
from abc import ABC, abstractmethod

from app.observability.models import GatewayTrace


class TraceSink(ABC):

    @abstractmethod
    def emit(self, trace: GatewayTrace) -> None:
        ...


class NoOpTraceSink(TraceSink):

    def emit(self, trace: GatewayTrace) -> None:
        pass
