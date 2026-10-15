from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum


class EventProcessingStatus(str, Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


@dataclass
class EventRecord:
    event_id: str
    status: EventProcessingStatus
    attempts: int = 0
    last_error: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None


class EventInbox(ABC):

    @abstractmethod
    async def receive(self, event_id: str) -> None:
        ...

    @abstractmethod
    async def claim(
        self,
        event_id: str,
        owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        ...

    @abstractmethod
    async def mark_processed(
        self,
        event_id: str,
        owner: str,
    ) -> bool:
        ...

    @abstractmethod
    async def mark_failed(
        self,
        event_id: str,
        owner: str,
        error: str,
    ) -> bool:
        ...


class InMemoryEventInbox(EventInbox):
    # Same lease semantics as PostgresEventInbox, for unit tests. The clock
    # is injectable so tests can expire a lease without sleeping.

    def __init__(
        self,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.records: dict[str, EventRecord] = {}
        self._clock = clock

    async def receive(self, event_id: str) -> None:
        if event_id not in self.records:
            self.records[event_id] = EventRecord(
                event_id=event_id,
                status=EventProcessingStatus.RECEIVED,
            )

    async def claim(
        self,
        event_id: str,
        owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        record = self.records.get(event_id)
        if record is None:
            return False

        now = self._clock()
        claimable = (
            record.status in {
                EventProcessingStatus.RECEIVED,
                EventProcessingStatus.FAILED,
            }
            or (
                record.status == EventProcessingStatus.PROCESSING
                and record.lease_expires_at is not None
                and record.lease_expires_at < now
            )
        )
        if not claimable:
            return False

        record.status = EventProcessingStatus.PROCESSING
        record.attempts += 1
        record.lease_owner = owner
        record.lease_expires_at = now + timedelta(seconds=lease_seconds)
        return True

    def _holds_lease(self, record: EventRecord | None, owner: str) -> bool:
        return (
            record is not None
            and record.status == EventProcessingStatus.PROCESSING
            and record.lease_owner == owner
            and record.lease_expires_at is not None
            and record.lease_expires_at > self._clock()
        )

    async def mark_processed(self, event_id: str, owner: str) -> bool:
        record = self.records.get(event_id)
        if not self._holds_lease(record, owner):
            return False

        record.status = EventProcessingStatus.PROCESSED
        record.lease_owner = None
        record.lease_expires_at = None
        return True

    async def mark_failed(
        self,
        event_id: str,
        owner: str,
        error: str,
    ) -> bool:
        record = self.records.get(event_id)
        if not self._holds_lease(record, owner):
            return False

        record.status = EventProcessingStatus.FAILED
        record.last_error = error
        record.lease_owner = None
        record.lease_expires_at = None
        return True
