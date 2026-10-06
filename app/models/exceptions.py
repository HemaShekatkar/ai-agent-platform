class ProviderError(Exception):
    """Base exception for provider failures."""
    pass


class TransientProviderError(ProviderError):
    """Temporary failure where fallback may succeed."""
    pass


class PermanentProviderError(ProviderError):
    """Failure where retry/fallback should not be attempted."""
    pass
