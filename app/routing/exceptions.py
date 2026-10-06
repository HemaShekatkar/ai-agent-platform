from typing import TYPE_CHECKING


class NoEligibleModelError(Exception):
    """Raised when no model satisfies the request's hard constraints."""
    pass


class ProviderNotFoundError(Exception):
    """Raised when the selected model has no provider in the registry."""
    pass


if TYPE_CHECKING:
    from app.gateway import GatewayDiagnostics


class AllProvidersFailedError(Exception):
    def __init__(
        self,
        message: str,
        diagnostics: "GatewayDiagnostics",
    ):
        super().__init__(message)
        self.diagnostics = diagnostics


class LatencyBudgetExceededError(Exception):
    """Raised when no remaining model can fit within the request latency budget."""

    def __init__(
        self,
        message: str,
        diagnostics: "GatewayDiagnostics | None" = None,
    ):
        super().__init__(message)
        self.diagnostics = diagnostics
