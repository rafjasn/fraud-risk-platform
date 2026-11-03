"""Supported transaction features, the source fields each needs, and feature sets."""

from typing import Literal

from pydantic import Field, model_validator

from fraud_risk_platform.contracts.base import Contract

FEATURES = ["amount", "hour_sin", "hour_cos"]
FEATURE_VERSION = "basic-v1"
NUMERIC_FEATURE_INPUTS = {
    "amount": "amount",
    "log_abs_amount": "amount",
    "is_refund": "amount",
    "hour_sin": "tx_datetime",
    "hour_cos": "tx_datetime",
    "weekday_sin": "tx_datetime",
    "weekday_cos": "tx_datetime",
    "is_weekend": "tx_datetime",
}
# Numeric features that are 0/1 by construction; spline encoding passes them through.
INDICATOR_FEATURES = {
    "is_refund",
    "is_weekend",
    "new_merchant_for_card_30d",
    "new_mcc_for_card_30d",
}
# Point-in-time card history (numeric); computed in features/history.py from the
# card's transactions in the preceding HISTORY_DAYS and its released labels.
HISTORY_DAYS = 30
HISTORY_VERSION = "history-v1"
CARD_INPUTS = ("user_id", "card_id", "tx_datetime")
HISTORY_FEATURE_INPUTS = {
    "card_count_1h": CARD_INPUTS,
    "card_count_24h": CARD_INPUTS,
    "card_count_7d": CARD_INPUTS,
    "card_amount_sum_24h": CARD_INPUTS,
    "amount_vs_card_mean_30d": (*CARD_INPUTS, "amount"),
    "hours_since_card_last": CARD_INPUTS,
    "new_merchant_for_card_30d": (*CARD_INPUTS, "merchant_id"),
    "new_mcc_for_card_30d": (*CARD_INPUTS, "mcc"),
    "card_confirmed_frauds_30d": CARD_INPUTS,
}
CATEGORICAL_FEATURE_INPUTS = {
    "use_chip": "use_chip",
    "mcc": "mcc",
    "merchant_city": "merchant_city",
    "merchant_state": "merchant_state",
    "postal_code": "postal_code",
    "merchant_id": "merchant_id",
    "day_of_week": "tx_datetime",
}


class FeatureSet(Contract):
    version: Literal["basic-v1", "transaction-v1", "history-v1"] = "transaction-v1"
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)

    @property
    def columns(self):
        return self.numeric + self.categorical

    @property
    def history_features(self):
        return [name for name in self.numeric if name in HISTORY_FEATURE_INPUTS]

    @property
    def inputs(self):
        definitions = NUMERIC_FEATURE_INPUTS | CATEGORICAL_FEATURE_INPUTS
        names = []
        for name in self.columns:
            names.extend(HISTORY_FEATURE_INPUTS.get(name) or [definitions[name]])
        return list(dict.fromkeys(names))

    @model_validator(mode="after")
    def supported_features(self):
        if not self.columns or len(self.columns) != len(set(self.columns)):
            raise ValueError("A feature set needs distinct, nonempty feature names")
        if (
            set(self.numeric)
            - NUMERIC_FEATURE_INPUTS.keys()
            - HISTORY_FEATURE_INPUTS.keys()
            or set(self.categorical) - CATEGORICAL_FEATURE_INPUTS.keys()
        ):
            raise ValueError(
                "Unsupported transaction feature or incorrect feature type"
            )
        if self.version == FEATURE_VERSION and (
            self.numeric != FEATURES or self.categorical
        ):
            raise ValueError(
                "basic-v1 is the immutable original three-feature contract"
            )
        if (self.version == HISTORY_VERSION) != bool(self.history_features):
            raise ValueError(
                "Card-history features require version history-v1, which requires at least one"
            )
        return self


def basic_feature_set():
    return FeatureSet(version=FEATURE_VERSION, numeric=list(FEATURES))
