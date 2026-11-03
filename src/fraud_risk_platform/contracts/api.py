"""Prediction API request and response schemas."""

from typing import Annotated

from pydantic import Field, NaiveDatetime

from fraud_risk_platform.contracts.base import Contract


class PredictionRequest(Contract):
    transaction_id: Annotated[int, Field(strict=True, gt=0)]
    amount: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    timestamp: NaiveDatetime
    use_chip: Annotated[str, Field(strict=True, min_length=1)] | None = None
    mcc: Annotated[str, Field(strict=True, pattern=r"^[0-9]{4}$")] | None = None
    merchant_city: Annotated[str, Field(strict=True, min_length=1)] | None = None
    merchant_state: Annotated[str, Field(strict=True, min_length=1)] | None = None
    postal_code: Annotated[str, Field(strict=True, min_length=1)] | None = None
    user_id: Annotated[int, Field(strict=True, ge=0, le=2**31 - 1)] | None = None
    card_id: Annotated[int, Field(strict=True, ge=0, le=32767)] | None = None
    merchant_id: (
        Annotated[int, Field(strict=True, ge=-(2**63), le=2**63 - 1)] | None
    ) = None


class PredictionResponse(Contract):
    source_import_id: str
    prediction_id: str
    transaction_id: int
    fraud_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    flagged_for_review: bool
    threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    release_id: str
    model_uri: str
    source_run_id: str
    feature_version: str
