
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class TokenReservation:
    reservation_id: str
    tenant_id: str
    reserved_tokens: int


class TokenBudgetManager:
    """
    Simplified single-process token budget.

    Models a fixed budget window, not a production
    rolling-minute limiter.
    """

    def __init__(self, tenant_limits: dict[str, int]):
        if any(limit <= 0 for limit in tenant_limits.values()):
            raise ValueError("Token limits must be positive")

        self.limits = dict(tenant_limits)
        self.consumed = {
            tenant_id: 0 for tenant_id in tenant_limits
        }
        self.reserved = {
            tenant_id: 0 for tenant_id in tenant_limits
        }
        self.active: dict[str, TokenReservation] = {}

    def available(self, tenant_id: str) -> int:
        if tenant_id not in self.limits:
            raise ValueError("Unknown tenant")

        return max(
            0,
            self.limits[tenant_id]
            - self.consumed[tenant_id]
            - self.reserved[tenant_id],
        )

    def reserve(
        self,
        tenant_id: str,
        estimated_tokens: int,
    ) -> TokenReservation | None:
        if estimated_tokens <= 0:
            raise ValueError("Estimated tokens must be positive")

        if self.available(tenant_id) < estimated_tokens:
            return None

        reservation = TokenReservation(
            reservation_id=str(uuid4()),
            tenant_id=tenant_id,
            reserved_tokens=estimated_tokens,
        )

        self.reserved[tenant_id] += estimated_tokens
        self.active[reservation.reservation_id] = reservation

        return reservation

    def reconcile(
        self,
        reservation_id: str,
        actual_tokens: int,
    ) -> None:
        if actual_tokens < 0:
            raise ValueError("Actual tokens cannot be negative")

        reservation = self.active[reservation_id]

        if actual_tokens > reservation.reserved_tokens:
            raise ValueError(
                "Actual usage exceeds reservation"
            )

        tenant_id = reservation.tenant_id

        self.reserved[tenant_id] -= reservation.reserved_tokens
        self.consumed[tenant_id] += actual_tokens
        del self.active[reservation_id]

    def cancel(self, reservation_id: str) -> None:
        reservation = self.active.pop(reservation_id)

        self.reserved[
            reservation.tenant_id
        ] -= reservation.reserved_tokens
