"""Recorded predictions joined to the outcomes released by a given clock."""

from decimal import Decimal

from fraud_risk_platform.persistence.database import connect, connect_source


def query_outcomes(
    clock, url: str | None = None, *, source_url: str | None = None
) -> list[dict]:
    """Return only labels available at the replay clock; keep pending outcomes null."""
    results = []
    # PostgreSQL joins cannot span databases. Fetch source outcomes in bounded
    # batches, keeping the exact identity and label-availability checks.
    with connect(url) as platform, connect_source(source_url or url) as source:
        source.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        with platform.cursor(name="outcome_predictions") as cursor:
            cursor.execute(
                """SELECT prediction_id::text,source_import_id,transaction_id,release_id,fraud_score,
                flagged_for_review,amount,tx_datetime,feature_inputs FROM predictions
                WHERE tx_datetime < %s AND source_import_id IS NOT NULL ORDER BY predicted_at,prediction_id""",
                (clock,),
            )
            while batch := cursor.fetchmany(1000):
                outcomes = source.execute(
                    """SELECT t.import_id,t.source_row,t.amount,t.tx_datetime,t.is_fraud,
                    t.label_available_at::text,t.use_chip,t.mcc,t.merchant_id,t.merchant_city,
                    t.merchant_state,t.postal_code,t.user_id,t.card_id FROM ibm_complete_transactions t
                    JOIN unnest(%s::text[],%s::bigint[]) AS requested(import_id,source_row)
                    ON t.import_id=requested.import_id AND t.source_row=requested.source_row
                    WHERE t.label_available_at <= %s AND t.tx_datetime < %s""",
                    (
                        [row["source_import_id"] for row in batch],
                        [row["transaction_id"] for row in batch],
                        clock,
                        clock,
                    ),
                ).fetchall()
                by_id = {(row["import_id"], row["source_row"]): row for row in outcomes}
                for prediction in batch:
                    outcome = by_id.get(
                        (prediction["source_import_id"], prediction["transaction_id"])
                    )
                    amount = prediction.pop("amount")
                    timestamp = prediction.pop("tx_datetime")
                    feature_inputs = prediction.pop("feature_inputs")
                    matched = (
                        outcome is not None
                        and outcome["amount"] == Decimal(str(amount))
                        and outcome["tx_datetime"] == timestamp
                        and all(
                            outcome.get(name) == value
                            for name, value in feature_inputs.items()
                        )
                    )
                    prediction["fraud"] = int(outcome["is_fraud"]) if matched else None
                    prediction["label_available_at"] = (
                        outcome["label_available_at"] if matched else None
                    )
                    results.append(prediction)
    return results
