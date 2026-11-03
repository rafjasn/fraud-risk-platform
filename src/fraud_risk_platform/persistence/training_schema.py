"""Training job ledger schema; models and reports live in MLflow and the state directory."""

TRAINING_DDL = """
-- One row per distinct training request; the ID is the hash of the request.
CREATE TABLE IF NOT EXISTS training_jobs (
    job_id text PRIMARY KEY,
    request jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
-- Every attempt stays visible. A job runs at most once at a time and succeeds at most once.
CREATE TABLE IF NOT EXISTS training_attempts (
    job_id text NOT NULL REFERENCES training_jobs(job_id),
    attempt integer NOT NULL CHECK (attempt > 0),
    status text NOT NULL CHECK (status IN ('running','succeeded','failed','abandoned')),
    run_id text,
    recommendation text,
    release_id text,
    error text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    PRIMARY KEY (job_id, attempt)
);
CREATE UNIQUE INDEX IF NOT EXISTS training_attempts_one_running
    ON training_attempts(job_id) WHERE status = 'running';
CREATE UNIQUE INDEX IF NOT EXISTS training_attempts_one_success
    ON training_attempts(job_id) WHERE status = 'succeeded';
"""
