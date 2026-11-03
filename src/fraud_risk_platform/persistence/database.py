"""PostgreSQL connections and schema initialization."""

import os

import psycopg
from psycopg.rows import dict_row

from fraud_risk_platform.persistence.ibm_schema import IBM_SOURCE_DDL
from fraud_risk_platform.persistence.ingestion_schema import INGESTION_DDL
from fraud_risk_platform.persistence.training_schema import TRAINING_DDL

PLATFORM_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id uuid PRIMARY KEY,
    transaction_id bigint NOT NULL,
    source_import_id text NOT NULL,
    tx_datetime timestamp NOT NULL,
    amount double precision NOT NULL,
    fraud_score double precision NOT NULL CHECK (fraud_score BETWEEN 0 AND 1),
    flagged_for_review boolean NOT NULL,
    threshold double precision NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    release_id text NOT NULL,
    source_run_id text NOT NULL,
    feature_version text NOT NULL,
    predicted_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS source_import_id text;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_inputs jsonb NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS predictions_transaction_idx ON predictions(transaction_id);
-- One replayed month of traffic per simulation, recorded at the month's start.
CREATE TABLE IF NOT EXISTS replays (
    replay_id uuid PRIMARY KEY,
    simulation_id text NOT NULL,
    source_import_id text NOT NULL,
    release_id text NOT NULL,
    window_start timestamp NOT NULL,
    window_end timestamp NOT NULL CHECK (window_end > window_start),
    clock timestamp NOT NULL CHECK (clock <= window_start),
    transactions integer NOT NULL,
    rejected jsonb NOT NULL,
    flagged integer NOT NULL,
    scoring_seconds double precision NOT NULL,
    completed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (simulation_id, window_start)
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS replay_id uuid REFERENCES replays(replay_id);
CREATE INDEX IF NOT EXISTS predictions_replay_idx ON predictions(replay_id);
"""


def connect(url: str | None = None):
    return psycopg.connect(url or os.environ["DATABASE_URL"], row_factory=dict_row)


def connect_source(url: str | None = None):
    return connect(url or os.environ.get("SOURCE_DATABASE_URL"))


def initialize(url: str | None = None, *, source_url: str | None = None):
    with connect_source(source_url or url) as conn:
        conn.execute(IBM_SOURCE_DDL)
    with connect(url) as conn:
        conn.execute(PLATFORM_DDL)
        conn.execute(INGESTION_DDL)
        conn.execute(TRAINING_DDL)
