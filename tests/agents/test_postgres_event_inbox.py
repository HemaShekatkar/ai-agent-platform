"""Integration tests for PostgresEventInbox against a real PostgreSQL.

Set TEST_DATABASE_URL to a disposable database, e.g.

    TEST_DATABASE_URL=postgresql://localhost:5432/agent_platform_test

The table is created from db/001_create_event_inbox.sql and emptied before
every test. Without TEST_DATABASE_URL (or if it is unreachable) these tests
are skipped, so the default suite never needs a database.
"""

import asyncio

import pytest

from app.agents.postgres_event_inbox import PostgresEventInbox


@pytest.fixture
def pool(postgres_pool):
    return postgres_pool


@pytest.fixture
def inbox(pool) -> PostgresEventInbox:
    return PostgresEventInbox(pool)


async def fetch_row(pool, event_id: str) -> dict | None:
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT status, attempts, lease_owner, last_error,
                   lease_expires_at > NOW() AS lease_active
            FROM agent_event_inbox
            WHERE event_id = %s
            """,
            (event_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        return None

    status, attempts, lease_owner, last_error, lease_active = row
    return {
        "status": status,
        "attempts": attempts,
        "lease_owner": lease_owner,
        "last_error": last_error,
        "lease_active": lease_active,
    }


async def expire_lease(pool, event_id: str) -> None:
    # Simulate an expired lease without sleeping
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE agent_event_inbox
            SET lease_expires_at = NOW() - INTERVAL '1 minute'
            WHERE event_id = %s
            """,
            (event_id,),
        )


@pytest.mark.asyncio
async def test_receive_is_idempotent(inbox, pool):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")

    await inbox.receive("event-123")

    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM agent_event_inbox WHERE event_id = %s",
            ("event-123",),
        )
        (count,) = await cursor.fetchone()

    assert count == 1

    # The duplicate did not reset the existing record
    row = await fetch_row(pool, "event-123")
    assert row["status"] == "PROCESSING"
    assert row["attempts"] == 1


@pytest.mark.asyncio
async def test_only_one_worker_can_claim(inbox, pool):
    await inbox.receive("event-123")

    # Many workers race for the same event on separate connections
    owners = [f"worker-{n}" for n in range(8)]
    results = await asyncio.gather(
        *(inbox.claim("event-123", owner=owner) for owner in owners)
    )

    assert results.count(True) == 1

    winner = owners[results.index(True)]
    row = await fetch_row(pool, "event-123")
    assert row["lease_owner"] == winner
    assert row["attempts"] == 1


@pytest.mark.asyncio
async def test_active_lease_blocks_other_claims(inbox, pool):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")

    assert await inbox.claim("event-123", owner="worker-b") is False
    assert (await fetch_row(pool, "event-123"))["lease_owner"] == "worker-a"


@pytest.mark.asyncio
async def test_expired_lease_can_be_reclaimed(inbox, pool):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")
    await expire_lease(pool, "event-123")

    assert await inbox.claim("event-123", owner="worker-b") is True

    row = await fetch_row(pool, "event-123")
    assert row["lease_owner"] == "worker-b"
    assert row["lease_active"] is True
    assert row["attempts"] == 2


@pytest.mark.asyncio
async def test_stale_owner_cannot_acknowledge(inbox, pool):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")
    await expire_lease(pool, "event-123")
    await inbox.claim("event-123", owner="worker-b")

    # worker-a finishes late; its outcome must not count
    assert await inbox.mark_processed("event-123", owner="worker-a") is False
    assert await inbox.mark_failed("event-123", "worker-a", "late") is False

    row = await fetch_row(pool, "event-123")
    assert row["status"] == "PROCESSING"
    assert row["lease_owner"] == "worker-b"

    # The current owner can still acknowledge
    assert await inbox.mark_processed("event-123", owner="worker-b") is True


@pytest.mark.asyncio
async def test_processed_event_cannot_be_reclaimed(inbox, pool):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")
    assert await inbox.mark_processed("event-123", owner="worker-a") is True

    # Even with a lease timestamp in the past, PROCESSED stays terminal
    await expire_lease(pool, "event-123")

    assert await inbox.claim("event-123", owner="worker-b") is False

    row = await fetch_row(pool, "event-123")
    assert row["status"] == "PROCESSED"
    assert row["lease_owner"] is None
    assert row["attempts"] == 1


@pytest.mark.asyncio
async def test_failed_event_can_be_retried(inbox, pool):
    await inbox.receive("event-123")
    await inbox.claim("event-123", owner="worker-a")

    assert await inbox.mark_failed(
        "event-123", "worker-a", "Temporary error"
    ) is True

    row = await fetch_row(pool, "event-123")
    assert row["status"] == "FAILED"
    assert row["last_error"] == "Temporary error"
    assert row["lease_owner"] is None

    assert await inbox.claim("event-123", owner="worker-b") is True

    row = await fetch_row(pool, "event-123")
    assert row["status"] == "PROCESSING"
    assert row["lease_owner"] == "worker-b"
    assert row["attempts"] == 2
