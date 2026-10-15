import os
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).parents[2]
MIGRATIONS = sorted((PROJECT_ROOT / "db").glob("*.sql"))
TABLES = ("agent_event_inbox", "agent_run_locks")


def test_database_url() -> str | None:
    # An exported variable wins; otherwise read just this one key from the
    # git-ignored .env, so the tests run from any terminal or IDE. Other
    # .env values (e.g. API keys) are deliberately not loaded into tests.
    return (
        os.environ.get("TEST_DATABASE_URL")
        or dotenv_values(PROJECT_ROOT / ".env").get("TEST_DATABASE_URL")
    )


@pytest_asyncio.fixture
async def postgres_pool():
    """A pool on a disposable PostgreSQL database, with every migration in
    db/ applied and all tables emptied. Skips unless TEST_DATABASE_URL is set
    and reachable, so the default suite never needs a database."""

    database_url = test_database_url()
    if not database_url:
        pytest.skip("TEST_DATABASE_URL not set; skipping PostgreSQL tests")

    psycopg_pool = pytest.importorskip("psycopg_pool")

    pool = psycopg_pool.AsyncConnectionPool(
        database_url,
        min_size=2,
        max_size=10,
        open=False,
    )
    try:
        await pool.open(wait=True, timeout=5)
    except Exception as exc:
        pytest.skip(f"PostgreSQL at TEST_DATABASE_URL unreachable: {exc}")

    async with pool.connection() as conn:
        for migration in MIGRATIONS:
            await conn.execute(migration.read_text())
        await conn.execute(f"TRUNCATE {', '.join(TABLES)}")

    yield pool

    await pool.close()
