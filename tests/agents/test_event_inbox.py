from datetime import datetime, timedelta, timezone

import pytest

from app.agents.event_inbox import (
    EventProcessingStatus,
    InMemoryEventInbox,
)


class FakeClock:
    # Lets tests expire a lease by advancing time instead of sleeping

    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def inbox(clock) -> InMemoryEventInbox:
    return InMemoryEventInbox(clock=clock)


@pytest.mark.asyncio
async def test_receive_new_event(inbox):
    await inbox.receive("event-123")

    record = inbox.records["event-123"]
    assert record.status == EventProcessingStatus.RECEIVED
    assert record.attempts == 0
    assert record.lease_owner is None


@pytest.mark.asyncio
async def test_receive_duplicate_event(inbox):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")

    await inbox.receive("event-123")

    # Redelivery leaves the existing record untouched
    assert len(inbox.records) == 1
    assert inbox.records["event-123"].status == EventProcessingStatus.PROCESSING
    assert inbox.records["event-123"].attempts == 1


@pytest.mark.asyncio
async def test_claim_event(inbox, clock):
    await inbox.receive("event-123")

    assert await inbox.claim("event-123", owner="worker-a", lease_seconds=30)

    record = inbox.records["event-123"]
    assert record.status == EventProcessingStatus.PROCESSING
    assert record.attempts == 1
    assert record.lease_owner == "worker-a"
    assert record.lease_expires_at == clock.now + timedelta(seconds=30)


@pytest.mark.asyncio
async def test_unknown_event_cannot_be_claimed(inbox):
    assert await inbox.claim("event-missing", owner="worker-a") is False


@pytest.mark.asyncio
async def test_processing_event_cannot_be_claimed_twice(inbox):
    await inbox.receive("event-123")

    assert await inbox.claim("event-123", owner="worker-a") is True
    assert await inbox.claim("event-123", owner="worker-b") is False

    record = inbox.records["event-123"]
    assert record.lease_owner == "worker-a"
    assert record.attempts == 1


@pytest.mark.asyncio
async def test_expired_lease_can_be_reclaimed(inbox, clock):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a", lease_seconds=60)

    # worker-a crashed; its lease runs out
    clock.advance(61)

    assert await inbox.claim("event-123", owner="worker-b") is True
    assert inbox.records["event-123"].lease_owner == "worker-b"
    assert inbox.records["event-123"].attempts == 2


@pytest.mark.asyncio
async def test_stale_owner_cannot_acknowledge(inbox, clock):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a", lease_seconds=60)
    clock.advance(61)
    await inbox.claim("event-123", owner="worker-b")

    # worker-a finishes late; its result must not count
    assert await inbox.mark_processed("event-123", owner="worker-a") is False
    assert await inbox.mark_failed("event-123", "worker-a", "late") is False

    record = inbox.records["event-123"]
    assert record.status == EventProcessingStatus.PROCESSING
    assert record.lease_owner == "worker-b"


@pytest.mark.asyncio
async def test_owner_cannot_acknowledge_after_own_lease_expires(inbox, clock):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a", lease_seconds=60)
    clock.advance(61)

    # Nobody reclaimed it, but the lease is gone: the work may be redone
    assert await inbox.mark_processed("event-123", owner="worker-a") is False


@pytest.mark.asyncio
async def test_processed_event_cannot_be_claimed_again(inbox, clock):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")

    assert await inbox.mark_processed("event-123", owner="worker-a") is True
    clock.advance(3600)

    assert await inbox.claim("event-123", owner="worker-b") is False
    assert inbox.records["event-123"].status == EventProcessingStatus.PROCESSED
    assert inbox.records["event-123"].lease_owner is None


@pytest.mark.asyncio
async def test_failed_event_can_be_retried(inbox):
    await inbox.receive("event-123")
    assert await inbox.claim("event-123", owner="worker-a") is True

    assert await inbox.mark_failed(
        "event-123", "worker-a", "Temporary error"
    ) is True

    record = inbox.records["event-123"]
    assert record.status == EventProcessingStatus.FAILED
    assert record.last_error == "Temporary error"

    assert await inbox.claim("event-123", owner="worker-b") is True
    assert record.attempts == 2
    assert record.status == EventProcessingStatus.PROCESSING
