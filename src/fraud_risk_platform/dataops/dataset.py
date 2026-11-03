"""Versioned IBM observations and dataset selection, independent of model recipes."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import (
    Field,
    NaiveDatetime,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from fraud_risk_platform.contracts.base import Contract, midnight


class DatasetProfile(Contract):
    schema_version: Literal[1] = 1
    dataset: Literal["ibm-tabformer"] = "ibm-tabformer"
    train_start: date
    context_days: int = Field(default=30, ge=0, le=3650)
    validation_days: int = Field(default=30, gt=0)
    evaluation_days: int = Field(default=30, gt=0)
    final_holdout_start: date
    max_train_rows: int | None = Field(default=1_000_000, gt=0)
    selection_seed: int = Field(default=42, ge=0)
    minimum_rows_per_split: int = Field(default=100, gt=0)
    minimum_frauds_per_split: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.train_start >= self.final_holdout_start:
            raise ValueError("Training start must precede the final holdout")
        return self

    @property
    def context_start(self) -> datetime:
        return midnight(self.train_start) - timedelta(days=self.context_days)

    def windows(self, as_of: str, label_delay_days: int) -> dict:
        end = midnight(as_of) - timedelta(days=label_delay_days)
        evaluation = end - timedelta(days=self.evaluation_days)
        validation = evaluation - timedelta(days=self.validation_days)
        start = midnight(self.train_start)
        if start >= validation:
            raise ValueError(
                "Cutoff leaves no training period before validation/evaluation"
            )
        return {
            "train": [start.isoformat(), validation.isoformat()],
            "validation": [validation.isoformat(), evaluation.isoformat()],
            "evaluation": [evaluation.isoformat(), end.isoformat()],
        }


class DataSettings(Contract):
    """One project configuration; normal commands need no selection arguments."""

    initial_snapshot: date
    label_delay_days: int = Field(default=7, ge=0, le=365)
    dataset: DatasetProfile


class Observation(Contract):
    source_row: StrictInt = Field(gt=0, le=2**63 - 1)
    user_id: StrictInt = Field(ge=0, le=2**31 - 1)
    card_id: StrictInt = Field(ge=0, le=32767)
    tx_datetime: NaiveDatetime
    amount: Decimal = Field(max_digits=18, decimal_places=2, allow_inf_nan=False)
    use_chip: Literal["Swipe Transaction", "Chip Transaction", "Online Transaction"]
    merchant_id: StrictInt = Field(ge=-(2**63), le=2**63 - 1)
    merchant_city: str = Field(min_length=1)
    merchant_state: str | None
    postal_code: str | None
    mcc: str = Field(pattern=r"^[0-9]{4}$")
    errors: str | None
    is_fraud: StrictBool | None
    label_available_at: NaiveDatetime

    @field_validator("amount", mode="before")
    @classmethod
    def exact_money(cls, value):
        if not isinstance(value, (str, Decimal)):
            raise ValueError("Amount must be an exact decimal string")
        return value


def arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("source_row", pa.int64(), nullable=False),
            pa.field("user_id", pa.int32(), nullable=False),
            pa.field("card_id", pa.int16(), nullable=False),
            pa.field("tx_datetime", pa.timestamp("us"), nullable=False),
            pa.field("amount", pa.decimal128(18, 2), nullable=False),
            pa.field("use_chip", pa.string(), nullable=False),
            pa.field("merchant_id", pa.int64(), nullable=False),
            pa.field("merchant_city", pa.string(), nullable=False),
            pa.field("merchant_state", pa.string()),
            pa.field("postal_code", pa.string()),
            pa.field("mcc", pa.string(), nullable=False),
            pa.field("errors", pa.string()),
            pa.field("is_fraud", pa.bool_()),
            pa.field("label_available_at", pa.timestamp("us"), nullable=False),
        ]
    )
