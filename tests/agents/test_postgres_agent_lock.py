"""Integration tests for PostgresAgentRunLock against a real PostgreSQL.

Uses the postgres_pool fixture from conftest.py: skipped unless
TEST_DATABASE_URL points at a disposable database.
"""

import asyncio

import pytest

from app.agents.postgres_agent_lock import PostgresAgentRunLock


@pytest.fixture
def lock(postgres_pool) -> PostgresAgentRunLock:
    return PostgresAgentRunLock(postgres_pool)


async def lease_owner(pool, agent_run_id: str) -> str | None:
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT lease_owner FROM agent_run_locks WHERE agent_run_id = %s",
            (agent_run_id,),
        )
        row = await cursor.fetchone()
    return row[0] if row else None


async def expire_lease(pool, agent_run_id: str) -> None:
    # Simulate an expired lease without sleeping
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE agent_run_locks
            SET lease_expires_at = NOW() - INTERVAL '1 minute'
            WHERE agent_run_id = %s
            """,
            (agent_run_id,),
        )


@pytest.mark.asyncio
async def test_first_acquisition_succeeds(lock, postgres_pool):
    assert await lock.acquire("agent-run-123", owner="worker-a") is True
    assert await lease_owner(postgres_pool, "agent-run-123") == "worker-a"


@pytest.mark.asyncio
async def test_second_owner_cannot_acquire_while_lease_valid(
    lock, postgres_pool,
):
    await lock.acquire("agent-run-123", owner="worker-a")

    assert await lock.acquire("agent-run-123", owner="worker-b") is False
    assert await lease_owner(postgres_pool, "agent-run-123") == "worker-a"


@pytest.mark.asyncio
async def test_different_agent_runs_can_both_acquire(lock, postgres_pool):
    # The lock is per run: unrelated runs proceed in parallel
    assert await lock.acquire("agent-run-123", owner="worker-a") is True
    assert await lock.acquire("agent-run-456", owner="worker-b") is True

    assert await lease_owner(postgres_pool, "agent-run-123") == "worker-a"
    assert await lease_owner(postgres_pool, "agent-run-456") == "worker-b"


@pytest.mark.asyncio
async def test_expired_lease_can_be_acquired_by_new_owner(lock, postgres_pool):
    await lock.acquire("agent-run-123", owner="worker-a")
    await expire_lease(postgres_pool, "agent-run-123")

    assert await lock.acquire("agent-run-123", owner="worker-b") is True
    assert await lease_owner(postgres_pool, "agent-run-123") == "worker-b"


@pytest.mark.asyncio
async def test_stale_owner_release_fails(lock, postgres_pool):
    await lock.acquire("agent-run-123", owner="worker-a")
    await expire_lease(postgres_pool, "agent-run-123")
    await lock.acquire("agent-run-123", owner="worker-b")

    # worker-a finishes late; it must not free worker-b's lock
    assert await lock.release("agent-run-123", owner="worker-a") is False
    assert await lease_owner(postgres_pool, "agent-run-123") == "worker-b"


@pytest.mark.asyncio
async def test_current_owner_release_succeeds(lock, postgres_pool):
    await lock.acquire("agent-run-123", owner="worker-a")

    assert await lock.release("agent-run-123", owner="worker-a") is True
    assert await lease_owner(postgres_pool, "agent-run-123") is None

    # Released runs can be acquired immediately by someone else
    assert await lock.acquire("agent-run-123", owner="worker-b") is True


@pytest.mark.asyncio
async def test_concurrent_acquisition_has_one_winner(lock, postgres_pool):
    owners = [f"worker-{n}" for n in range(8)]

    results = await asyncio.gather(
        *(lock.acquire("agent-run-123", owner=owner) for owner in owners)
    )

    assert results.count(True) == 1
    winner = owners[results.index(True)]
    assert await lease_owner(postgres_pool, "agent-run-123") == winner
