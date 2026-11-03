"""Monthly replay of live traffic through the deployed release.

A replay scores every transaction of the month after the simulation's current cutoff
as the API would have scored it at its own time: the same release, feature code, and
point-in-time card history, with labels visible only once released. It runs while the
simulation clock is still at the month's start, before the next snapshot releases any
of the month's outcomes, and it never reads those outcomes. Predictions are recorded
in one transaction, so a failed replay leaves nothing behind and can simply be rerun.
"""

import logging
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from fraud_risk_platform.contracts.features import HISTORY_DAYS
from fraud_risk_platform.dataops.workflow import simulation_clock
from fraud_risk_platform.features.history import HISTORY_COLUMNS
from fraud_risk_platform.features.transaction import (
    artifact_feature_spec,
    build_transaction_features,
)
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.database import connect, connect_source

logger = logging.getLogger(__name__)
# Raw request fields the API accepts as null; any other null input is rejected (HTTP 422).
NULLABLE_INPUTS = {"merchant_state", "postal_code"}
SOURCE_INPUTS = {
    "user_id",
    "card_id",
    "tx_datetime",
    "amount",
    "use_chip",
    "merchant_id",
    "merchant_city",
    "merchant_state",
    "postal_code",
    "mcc",
    "errors",
}


def _traffic(conn, spec, import_id, start, end) -> list[dict]:
    """The window's transactions with model inputs only; no labels are selected."""
    columns = sorted(set(spec.inputs) | {"tx_datetime", "amount"})
    if not set(columns) <= SOURCE_INPUTS:
        raise ValueError("Release inputs are not source transaction fields")
    fields = [
        sql.SQL("amount::float8 AS amount")
        if name == "amount"
        else sql.Identifier(name)
        for name in columns
    ]
    return conn.execute(
        sql.SQL(
            "SELECT source_row, {} FROM ibm_complete_transactions "
            "WHERE import_id=%s AND tx_datetime >= %s AND tx_datetime < %s "
            "ORDER BY tx_datetime, source_row"
        ).format(sql.SQL(", ").join(fields)),
        (import_id, start, end),
    ).fetchall()


def _history(conn, import_id, start, end) -> pd.DataFrame:
    """Card history for the window; feature code masks labels per transaction time."""
    rows = conn.execute(
        """SELECT user_id, card_id, tx_datetime, amount::float8 AS amount, merchant_id, mcc,
        CASE WHEN label_available_at <= %(end)s THEN is_fraud END AS is_fraud,
        label_available_at FROM ibm_complete_transactions
        WHERE import_id=%(import_id)s AND tx_datetime >= %(start)s AND tx_datetime < %(end)s""",
        {
            "import_id": import_id,
            "start": start - timedelta(days=HISTORY_DAYS),
            "end": end,
        },
    ).fetchall()
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def replayed(simulation_id: str, window_start, platform_url=None) -> bool:
    with connect(platform_url) as conn:
        return bool(
            conn.execute(
                "SELECT 1 FROM replays WHERE simulation_id=%s AND window_start=%s",
                (simulation_id, window_start),
            ).fetchone()
        )


def any_replays(simulation_id: str, platform_url=None) -> bool:
    with connect(platform_url) as conn:
        return bool(
            conn.execute(
                "SELECT 1 FROM replays WHERE simulation_id=%s LIMIT 1",
                (simulation_id,),
            ).fetchone()
        )


def replay(root: Path, settings, *, source_url=None, platform_url=None) -> dict:
    clock = simulation_clock(settings, source_url=source_url, platform_url=platform_url)
    release_id = deployment(root)["current"]
    if not release_id:
        raise ValueError("Promote a release before replaying traffic")
    manifest, model = load_release(root, release_id)
    import_id = manifest["source_import_id"]
    if import_id != clock["import_id"]:
        raise ValueError("The deployed release uses a different source import")
    start, end = clock["clock"], clock["next_cutoff"]
    if start >= end:
        raise ValueError("The simulation has reached the end of the source data")
    if replayed(clock["simulation_id"], start, platform_url):
        raise ValueError(
            f"Traffic from {start:%Y-%m-%d} is already replayed; run make next-snapshot"
        )
    spec = artifact_feature_spec(manifest)
    started = perf_counter()
    with connect_source(source_url) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        rows = _traffic(conn, spec, import_id, start, end)
        history = (
            _history(conn, import_id, start, end) if spec.history_features else None
        )
    rejected = {}
    accepted = []
    for row in rows:
        missing = [
            name
            for name in spec.inputs
            if name not in NULLABLE_INPUTS and row[name] is None
        ]
        for name in missing:
            rejected[name] = rejected.get(name, 0) + 1
        if not missing:
            accepted.append(row)
    scores = np.empty(0)
    if accepted:
        features = build_transaction_features(
            pd.DataFrame(accepted), spec, history=history
        )
        scores = adapter_for(manifest["recipe"]["estimator"]).scores(model, features)
    threshold = manifest["threshold"]
    flagged = scores >= threshold
    seconds = perf_counter() - started
    replay_id = uuid4()
    inputs = [name for name in spec.inputs if name not in {"amount", "tx_datetime"}]
    try:
        with connect(platform_url) as conn:
            conn.execute(
                """INSERT INTO replays (replay_id, simulation_id, source_import_id, release_id,
                window_start, window_end, clock, transactions, rejected, flagged, scoring_seconds)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    replay_id,
                    clock["simulation_id"],
                    import_id,
                    release_id,
                    start,
                    end,
                    clock["clock"],
                    len(accepted),
                    Jsonb(rejected),
                    int(flagged.sum()),
                    seconds,
                ),
            )
            with conn.cursor().copy(
                """COPY predictions (prediction_id, source_import_id, transaction_id,
                tx_datetime, amount, fraud_score, flagged_for_review, threshold, release_id,
                source_run_id, feature_version, feature_inputs, replay_id) FROM STDIN"""
            ) as copy:
                for row, score, flag in zip(accepted, scores, flagged):
                    copy.write_row(
                        (
                            uuid4(),
                            import_id,
                            row["source_row"],
                            row["tx_datetime"],
                            row["amount"],
                            float(score),
                            bool(flag),
                            threshold,
                            release_id,
                            manifest["source_run_id"],
                            spec.version,
                            Jsonb({name: row[name] for name in inputs}),
                            replay_id,
                        )
                    )
    except psycopg.errors.UniqueViolation:
        raise ValueError(
            f"Traffic from {start:%Y-%m-%d} is already replayed; run make next-snapshot"
        ) from None
    result = {
        "replay_id": str(replay_id),
        "release_id": release_id,
        "window": [start.isoformat(), end.isoformat()],
        "simulation_clock": clock["clock"].isoformat(),
        "transactions": len(accepted),
        "rejected": rejected,
        "flagged": int(flagged.sum()),
        "alert_rate": float(flagged.mean()) if accepted else 0.0,
        "scoring_seconds": round(seconds, 1),
    }
    logger.info(
        "Replayed %d transactions from %s to %s with release %s: %d flagged (%.3f%%)",
        len(accepted),
        start.date(),
        end.date(),
        release_id[:12],
        result["flagged"],
        result["alert_rate"] * 100,
    )
    return result
