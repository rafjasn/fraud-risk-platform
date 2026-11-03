"""IBM source imports, transaction identities, and completed-import visibility."""

IBM_SOURCE_DDL = """
CREATE TABLE IF NOT EXISTS ibm_source_imports (
    import_id text PRIMARY KEY,
    source_manifest jsonb NOT NULL,
    adapter_version integer NOT NULL,
    customer_limit integer CHECK (customer_limit > 0),
    label_delay_days integer NOT NULL CHECK (label_delay_days >= 0),
    status text NOT NULL CHECK (status IN ('loading', 'failed', 'complete')),
    scanned_rows bigint NOT NULL DEFAULT 0 CHECK (scanned_rows >= 0),
    imported_rows bigint NOT NULL DEFAULT 0 CHECK (imported_rows >= 0),
    last_error text,
    started_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    CHECK (imported_rows <= scanned_rows)
);
CREATE TABLE IF NOT EXISTS ibm_transactions (
    import_id text NOT NULL REFERENCES ibm_source_imports(import_id),
    source_row bigint NOT NULL CHECK (source_row > 0),
    user_id integer NOT NULL CHECK (user_id >= 0),
    card_id smallint NOT NULL CHECK (card_id >= 0),
    tx_datetime timestamp NOT NULL,
    amount numeric(18,2) NOT NULL CHECK (amount > '-Infinity' AND amount < 'Infinity'),
    use_chip text NOT NULL CHECK (use_chip IN ('Swipe Transaction', 'Chip Transaction', 'Online Transaction')),
    merchant_id bigint NOT NULL,
    merchant_city text NOT NULL,
    merchant_state text,
    postal_code text,
    mcc text NOT NULL CHECK (mcc ~ '^[0-9]{4}$'),
    errors text,
    is_fraud boolean NOT NULL,
    label_available_at timestamp NOT NULL,
    PRIMARY KEY (import_id, source_row),
    CHECK (label_available_at >= tx_datetime)
);
CREATE INDEX IF NOT EXISTS ibm_transactions_time_idx
    ON ibm_transactions(import_id, tx_datetime, source_row);
-- Serving reads one card's recent history per prediction.
CREATE INDEX IF NOT EXISTS ibm_transactions_card_idx
    ON ibm_transactions(import_id, user_id, card_id, tx_datetime);
CREATE OR REPLACE VIEW ibm_complete_transactions AS
    SELECT t.* FROM ibm_transactions t
    JOIN ibm_source_imports i ON i.import_id = t.import_id
    WHERE i.status = 'complete';
"""
