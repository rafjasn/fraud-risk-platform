"""Training feature matrices built from a verified snapshot's partitions."""

import logging
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from fraud_risk_platform.contracts.features import HISTORY_DAYS, basic_feature_set
from fraud_risk_platform.features.history import HISTORY_COLUMNS
from fraud_risk_platform.features.transaction import build_transaction_features

logger = logging.getLogger(__name__)


def card_history(reader):
    """A snapshot's accepted context transactions, the history source for card features."""
    if reader.profile.context_days < HISTORY_DAYS:
        raise ValueError(
            f"Card-history features need {HISTORY_DAYS} days of snapshot context; "
            f"this snapshot has {reader.profile.context_days}"
        )
    started = perf_counter()
    table = pa.Table.from_batches(
        list(reader.iter_batches("context", columns=HISTORY_COLUMNS))
    )
    table = table.set_column(
        table.schema.get_field_index("amount"),
        "amount",
        pc.cast(table["amount"], pa.float64()),
    )
    history = table.to_pandas(strings_to_categorical=True)
    logger.info(
        "Loaded card history: %d context rows in %.1fs",
        len(history),
        perf_counter() - started,
    )
    return history


def feature_matrices(
    reader, splits=("train", "validation", "evaluation"), *, feature_spec=None
):
    """Build requested typed features; category encoders are fitted later on training only."""
    spec = feature_spec or basic_feature_set()
    history = card_history(reader) if spec.history_features else None
    matrices, labels = {}, {}
    for split in splits:
        started = perf_counter()
        count = reader.manifest["splits"][split]["rows"]
        features = {
            name: np.empty(count, dtype=np.float64 if name in spec.numeric else object)
            for name in spec.columns
        }
        targets = np.empty(count, dtype=np.int64)
        offset = 0
        columns = [*spec.inputs, "is_fraud"]
        # History features index the whole history once per split, not per batch.
        batches = (
            [pa.Table.from_batches(list(reader.iter_batches(split, columns=columns)))]
            if history is not None
            else reader.iter_batches(split, columns=columns)
        )
        for batch in batches:
            if "amount" in batch.schema.names:
                # Features use float amounts; skip per-row Decimal objects.
                batch = batch.set_column(
                    batch.schema.get_field_index("amount"),
                    "amount",
                    pc.cast(batch["amount"], pa.float64()),
                )
            frame = batch.to_pandas()
            end = offset + len(frame)
            if end > count:
                raise ValueError("Feature rows exceed snapshot split count")
            prepared = build_transaction_features(frame, spec, history=history)
            for name in spec.columns:
                features[name][offset:end] = prepared[name].to_numpy()
            targets[offset:end] = frame.is_fraud.to_numpy(dtype=np.int64)
            offset = end
        if offset != count:
            raise ValueError("Feature rows disagree with snapshot split count")
        matrices[split] = pd.DataFrame(features, columns=spec.columns, copy=False)
        labels[split] = pd.Series(targets, copy=False)
        logger.info(
            "Loaded %s features: %d rows x %d columns, %d frauds in %.1fs",
            split,
            count,
            len(spec.columns),
            int(targets.sum()),
            perf_counter() - started,
        )
    return matrices, labels


def window_features(reader, window, *, feature_spec, history=None):
    """Features and labels for the context rows in [start, end), in (time, row) order.

    For a period that no partition covers, such as the final holdout. Card history
    comes from the whole context, as in training; pass `history` to reuse it across
    feature sets. Every row must have a released label.
    """
    start, end = (
        pa.scalar(datetime.fromisoformat(value), pa.timestamp("us")) for value in window
    )
    columns = list(dict.fromkeys([*feature_spec.inputs, "tx_datetime", "is_fraud"]))
    tables = []
    for batch in reader.iter_batches("context", columns=columns):
        times = batch.column("tx_datetime")
        inside = pc.and_(pc.greater_equal(times, start), pc.less(times, end))
        tables.append(pa.Table.from_batches([batch.filter(inside)]))
    table = pa.concat_tables(tables) if tables else None
    if table is None or table.num_rows == 0:
        raise ValueError(
            f"The snapshot has no transactions from {window[0]} to {window[1]}"
        )
    if table["is_fraud"].null_count:
        raise ValueError(
            f"{table['is_fraud'].null_count} transactions from {window[0]} to {window[1]} "
            "have no released label at this snapshot's cutoff"
        )
    if "amount" in table.schema.names:
        table = table.set_column(
            table.schema.get_field_index("amount"),
            "amount",
            pc.cast(table["amount"], pa.float64()),
        )
    frame = table.to_pandas()
    if feature_spec.history_features and history is None:
        history = card_history(reader)
    features = build_transaction_features(frame, feature_spec, history=history)
    return features, frame.is_fraud.astype(np.int64)
