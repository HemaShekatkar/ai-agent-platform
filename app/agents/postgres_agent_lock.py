from psycopg_pool import AsyncConnectionPool

from app.agents.agent_lock import AgentRunLock


class PostgresAgentRunLock(AgentRunLock):

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def acquire(
        self,
        agent_run_id: str,
        owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO agent_run_locks (
                    agent_run_id,
                    lease_owner,
                    lease_expires_at
                )
                VALUES (
                    %s,
                    %s,
                    NOW() + (%s * INTERVAL '1 second')
                )
                ON CONFLICT (agent_run_id)
                DO UPDATE SET
                    lease_owner = EXCLUDED.lease_owner,
                    lease_expires_at = EXCLUDED.lease_expires_at,
                    updated_at = NOW()
                WHERE agent_run_locks.lease_expires_at < NOW()
                RETURNING agent_run_id
                """,
                (agent_run_id, owner, lease_seconds),
            )

            return await cursor.fetchone() is not None

    async def release(
        self,
        agent_run_id: str,
        owner: str,
    ) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                DELETE FROM agent_run_locks
                WHERE agent_run_id = %s
                  AND lease_owner = %s
                RETURNING agent_run_id
                """,
                (agent_run_id, owner),
            )

            return await cursor.fetchone() is not None
