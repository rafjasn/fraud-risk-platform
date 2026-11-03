"""Transaction features shared by snapshot training and prediction."""

import numpy as np
import pandas as pd

from fraud_risk_platform.contracts.features import FeatureSet, basic_feature_set
from fraud_risk_platform.features.history import build_history_features


def build_transaction_features(
    frame: pd.DataFrame, spec: FeatureSet, *, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Stateless features, plus card history when the spec selects it.

    `history` holds observed transactions (features/history.HISTORY_COLUMNS);
    only rows strictly before each target and released labels are used.
    """
    missing = set(spec.inputs) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing required transaction fields: {', '.join(sorted(missing))}"
        )
    for name in spec.inputs:
        if name not in {"merchant_state", "postal_code"} and frame[name].isna().any():
            raise ValueError(f"Transaction field {name} cannot be null")
    values = {}
    if "amount" in spec.inputs:
        amount = frame.amount.astype(float)
        if not np.isfinite(amount).all():
            raise ValueError("Amount must be finite")
        values.update(
            amount=amount,
            log_abs_amount=np.log1p(abs(amount)),
            is_refund=(amount < 0).astype(float),
        )
    if "tx_datetime" in spec.inputs:
        timestamps = pd.to_datetime(frame.tx_datetime)
        if timestamps.dt.tz is not None:
            raise ValueError("Timestamp must be naive UTC, matching the source data")
        hour_angle = 2 * np.pi * timestamps.dt.hour / 24
        weekday = timestamps.dt.dayofweek
        weekday_angle = 2 * np.pi * weekday / 7
        values.update(
            hour_sin=np.sin(hour_angle),
            hour_cos=np.cos(hour_angle),
            weekday_sin=np.sin(weekday_angle),
            weekday_cos=np.cos(weekday_angle),
            is_weekend=(weekday >= 5).astype(float),
            day_of_week=weekday.astype(str),
        )
    if spec.history_features:
        if history is None:
            raise ValueError("Card-history features require transaction history")
        values.update(
            build_history_features(frame, history, spec.history_features).items()
        )
    for name in spec.categorical:
        if name != "day_of_week":
            # Preserve category identity, including signed 64-bit merchant IDs.
            values[name] = frame[name].map(
                lambda value: np.nan if pd.isna(value) else str(value)
            )
    return pd.DataFrame(
        {
            name: pd.Series(
                values[name],
                index=frame.index,
                dtype=object if name in spec.categorical else np.float64,
            )
            for name in spec.columns
        },
        index=frame.index,
    )


def feature_records(frame):
    """JSON-safe feature examples, retaining nulls for nullable categories."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def model_feature_spec(model):
    saved = getattr(model, "fraud_feature_spec_", None)
    return (
        FeatureSet.model_validate(saved) if saved is not None else basic_feature_set()
    )


def artifact_feature_spec(metadata):
    """Read an immutable saved definition, with support for original basic bundles."""
    saved = metadata.get("feature_spec")
    spec = (
        FeatureSet.model_validate(saved) if saved is not None else basic_feature_set()
    )
    if (
        metadata["features"] != spec.columns
        or metadata["feature_version"] != spec.version
    ):
        raise ValueError("Incompatible artifact feature contract")
    return spec
