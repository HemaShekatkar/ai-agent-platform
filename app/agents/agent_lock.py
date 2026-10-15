from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta, timezone


class AgentRunLock(ABC):

    @abstractmethod
    async def acquire(
        self,
        agent_run_id: str,
        owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        ...

    @abstractmethod
    async def release(
        self,
        agent_run_id: str,
        owner: str,
    ) -> bool:
        ...


class InMemoryAgentRunLock(AgentRunLock):
    # Same semantics as PostgresAgentRunLock, for unit tests: one owner per
    # agent run until released or the lease expires

    def __init__(
        self,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.leases: dict[str, tuple[str, datetime]] = {}
        self._clock = clock

    async def acquire(
        self,
        agent_run_id: str,
        owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        now = self._clock()
        current = self.leases.get(agent_run_id)

        if current is not None and current[1] >= now:
            return False

        self.leases[agent_run_id] = (
            owner,
            now + timedelta(seconds=lease_seconds),
        )
        return True

    async def release(
        self,
        agent_run_id: str,
        owner: str,
    ) -> bool:
        current = self.leases.get(agent_run_id)

        if current is None or current[0] != owner:
            return False

        del self.leases[agent_run_id]
        return True
