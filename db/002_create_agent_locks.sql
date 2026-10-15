CREATE TABLE IF NOT EXISTS agent_run_locks (
    agent_run_id TEXT PRIMARY KEY,
    lease_owner TEXT NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);