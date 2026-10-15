from psycopg_pool import AsyncConnectionPool

from app.agents.event_inbox import EventInbox


class PostgresEventInbox(EventInbox):

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def receive(self, event_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO agent_event_inbox (event_id)
                VALUES (%s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (event_id,),
            )

    async def claim(
        self,
        event_id: str,
        owner: str,
        lease_seconds: int = 60,
    ) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE agent_event_inbox
                SET status = 'PROCESSING',
                    attempts = attempts + 1,
                    lease_owner = %s,
                    lease_expires_at =
                        NOW() + (%s * INTERVAL '1 second'),
                    updated_at = NOW()
                WHERE event_id = %s
                  AND (
                      status IN ('RECEIVED', 'FAILED')
                      OR (
                          status = 'PROCESSING'
                          AND lease_expires_at < NOW()
                      )
                  )
                RETURNING event_id
                """,
                (owner, lease_seconds, event_id),
            )

            return await cursor.fetchone() is not None

    async def mark_processed(
        self,
        event_id: str,
        owner: str,
    ) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE agent_event_inbox
                SET status = 'PROCESSED',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW()
                WHERE event_id = %s
                  AND status = 'PROCESSING'
                  AND lease_owner = %s
                  AND lease_expires_at > NOW()
                RETURNING event_id
                """,
                (event_id, owner),
            )

            return await cursor.fetchone() is not None

    async def mark_failed(
        self,
        event_id: str,
        owner: str,
        error: str,
    ) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE agent_event_inbox
                SET status = 'FAILED',
                    last_error = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = NOW()
                WHERE event_id = %s
                  AND status = 'PROCESSING'
                  AND lease_owner = %s
                  AND lease_expires_at > NOW()
                RETURNING event_id
                """,
                (error, event_id, owner),
            )

            return await cursor.fetchone() is not None
