CREATE TABLE IF NOT EXISTS agent_event_inbox (
    event_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'RECEIVED',
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_event_status CHECK (
        status IN ('RECEIVED', 'PROCESSING', 'PROCESSED', 'FAILED')
    )
);

CREATE INDEX IF NOT EXISTS idx_agent_event_inbox_recovery
ON agent_event_inbox (lease_expires_at)
WHERE status = 'PROCESSING';