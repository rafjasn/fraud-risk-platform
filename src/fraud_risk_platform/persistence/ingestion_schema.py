"""Additive ingestion ledger schema; accepted data lives in immutable S3 objects."""

INGESTION_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_streams (
    stream_id text PRIMARY KEY,
    configuration jsonb NOT NULL,
    accepted_cutoff timestamp
);
CREATE TABLE IF NOT EXISTS ingestion_attempts (
    attempt_id text PRIMARY KEY,
    stream_id text NOT NULL REFERENCES ingestion_streams(stream_id),
    as_of timestamp NOT NULL,
    status text NOT NULL CHECK (status IN ('running','accepted','rejected','failed')),
    report_key text,
    error text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id text PRIMARY KEY,
    stream_id text NOT NULL REFERENCES ingestion_streams(stream_id),
    previous_cutoff timestamp,
    as_of timestamp NOT NULL,
    manifest jsonb NOT NULL,
    UNIQUE (stream_id, as_of)
);
CREATE TABLE IF NOT EXISTS monthly_snapshot_progress (
    workflow_id text PRIMARY KEY,
    simulation_id text NOT NULL,
    initial_snapshot_id text,
    last_snapshot_id text,
    last_cutoff timestamp,
    updated_at timestamptz NOT NULL DEFAULT now()
);
"""
