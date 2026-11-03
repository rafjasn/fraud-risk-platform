import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from fraud_risk_platform.contracts.features import HISTORY_DAYS
from fraud_risk_platform.features.history import HISTORY_COLUMNS
from fraud_risk_platform.features.transaction import (
    artifact_feature_spec,
    build_transaction_features,
)
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.models.releases import load_release


class FraudPredictor:
    def __init__(
        self,
        *,
        root: Path,
        release_id: str,
        history: Callable[..., list[dict]] | None = None,
        step: Callable[[str], AbstractContextManager] | None = None,
    ):
        """`history(import_id, user_id, card_id, start, end)` returns that card's
        transactions in [start, end) with labels masked as of end. `step(name)`, if
        given, times the `history` and `scoring` steps of each prediction."""
        self.manifest, self.model = load_release(root, release_id)
        self.release_id = release_id
        self.threshold = self.manifest["threshold"]
        self.adapter = adapter_for(self.manifest["recipe"]["estimator"])
        self.feature_spec = artifact_feature_spec(self.manifest)
        if self.feature_spec.history_features and history is None:
            raise ValueError("This release needs a card-history source")
        self.history = history
        self.step = step or (lambda name: nullcontext())

    def predict(
        self, *, amount: float, timestamp: datetime, **transaction_fields
    ) -> dict:
        if isinstance(amount, bool) or not math.isfinite(amount):
            raise ValueError("Amount must be finite")
        if timestamp.tzinfo is not None:
            raise ValueError("Timestamp must be naive UTC, matching the source data")
        frame = pd.DataFrame(
            [{**transaction_fields, "amount": float(amount), "tx_datetime": timestamp}]
        )
        history = None
        if self.feature_spec.history_features:
            missing = [
                name
                for name in ("user_id", "card_id")
                if transaction_fields.get(name) is None
            ]
            if missing:
                raise ValueError(
                    f"Missing required transaction fields: {', '.join(missing)}"
                )
            with self.step("history"):
                history = pd.DataFrame(
                    self.history(
                        import_id=self.manifest["source_import_id"],
                        user_id=transaction_fields["user_id"],
                        card_id=transaction_fields["card_id"],
                        start=timestamp - timedelta(days=HISTORY_DAYS),
                        end=timestamp,
                    ),
                    columns=HISTORY_COLUMNS,
                )
        with self.step("scoring"):
            features = build_transaction_features(
                frame, self.feature_spec, history=history
            )
            score = float(self.adapter.scores(self.model, features)[0])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Model returned an invalid probability")
        return {
            "source_import_id": self.manifest["source_import_id"],
            "fraud_score": score,
            "flagged_for_review": score >= self.threshold,
            "threshold": self.threshold,
            "release_id": self.release_id,
            "model_uri": self.manifest.get(
                "model_uri", f"releases/{self.release_id}/model.joblib"
            ),
            "source_run_id": self.manifest["source_run_id"],
            "feature_version": self.feature_spec.version,
        }
