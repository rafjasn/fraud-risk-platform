"""Prediction records and the per-request card history the API reads.

The functions take connection factories: calling one returns a context manager that
yields a database connection, such as a checkout from one of the API's pools, or
`lambda: connect(url)` in a job or test.
"""

from decimal import Decimal

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

# Connections per database. The API runs one worker, so this bounds its connections.
POOL_SIZE = 8
# Seconds a request waits for a connection, so an outage answers HTTP 503 quickly.
POOL_TIMEOUT = 2.0


def pool(url: str, *, name: str) -> ConnectionPool:
    """Connections the API reuses across requests; its lifespan opens and closes them.

    Opening does not wait for the database, so the API starts during an outage and
    reports itself unready. Each checkout first checks its connection, so one broken
    while idle, for example by a database restart, is replaced instead of failing
    a request.
    """
    return ConnectionPool(
        url,
        kwargs={"row_factory": dict_row},
        min_size=1,
        max_size=POOL_SIZE,
        timeout=POOL_TIMEOUT,
        check=ConnectionPool.check_connection,
        name=name,
        open=False,
    )


def unconfigured():
    """The connection factory for a database without a configured URL."""
    raise psycopg.OperationalError("No database URL is configured")


def record_prediction(payload, result: dict, *, platform, source, feature_inputs=None):
    """Store a prediction in the platform database after checking the source.

    `platform` and `source` are connection factories for those databases.
    """
    feature_inputs = feature_inputs or {}
    values = {
        **result,
        "transaction_id": payload.transaction_id,
        "tx_datetime": payload.timestamp,
        "amount": payload.amount,
        "feature_inputs": Jsonb(feature_inputs),
    }
    with source() as conn:
        # If history already exists, prevent an ambiguous later outcome join.
        known = conn.execute(
            """SELECT tx_datetime, amount, use_chip, mcc, merchant_id, merchant_city,
            merchant_state, postal_code, user_id, card_id
            FROM ibm_complete_transactions WHERE import_id=%s AND source_row=%s""",
            (result["source_import_id"], payload.transaction_id),
        ).fetchone()
        if known and (
            known["tx_datetime"] != payload.timestamp
            or known["amount"] != Decimal(str(payload.amount))
            or any(known.get(name) != value for name, value in feature_inputs.items())
        ):
            raise ValueError(
                "transaction_id already exists with different amount, timestamp, or feature inputs"
            )
    with platform() as conn:
        conn.execute(
            """INSERT INTO predictions
            (prediction_id, source_import_id, transaction_id, tx_datetime, amount, fraud_score,
             flagged_for_review, threshold, release_id, source_run_id, feature_version, feature_inputs)
            VALUES (%(prediction_id)s, %(source_import_id)s, %(transaction_id)s, %(tx_datetime)s, %(amount)s,
             %(fraud_score)s, %(flagged_for_review)s, %(threshold)s, %(release_id)s,
             %(source_run_id)s, %(feature_version)s, %(feature_inputs)s)""",
            values,
        )


def card_history(
    connection, *, import_id: str, user_id: int, card_id: int, start, end
) -> list[dict]:
    """One card's transactions in [start, end), with labels masked as of end.

    `connection` is a connection factory for the source database.
    """
    with connection() as conn:
        return conn.execute(
            """SELECT user_id, card_id, tx_datetime, amount::float8 AS amount, merchant_id,
            mcc, CASE WHEN label_available_at <= %(end)s THEN is_fraud END AS is_fraud,
            label_available_at FROM ibm_complete_transactions
            WHERE import_id=%(import_id)s AND user_id=%(user_id)s AND card_id=%(card_id)s
            AND tx_datetime >= %(start)s AND tx_datetime < %(end)s
            ORDER BY tx_datetime, source_row""",
            {
                "import_id": import_id,
                "user_id": user_id,
                "card_id": card_id,
                "start": start,
                "end": end,
            },
        ).fetchall()
