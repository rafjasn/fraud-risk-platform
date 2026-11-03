"""Framework boundary shared by fitting, artifact verification, and scoring."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from fraud_risk_platform.contracts.training import Recipe
from fraud_risk_platform.features.transaction import model_feature_spec
from fraud_risk_platform.models.recipes import BUILDERS, build_model


class ModelAdapter(Protocol):
    # Files the adapter writes into a trial or release directory.
    files: tuple[str, ...]

    def fit(self, recipe: Recipe, features, labels, *, validation=None): ...
    def scores(self, model, features) -> np.ndarray: ...
    def save(self, model, directory: Path): ...
    def load(self, directory: Path): ...


class SklearnAdapter:
    """scikit-learn pipelines, including XGBoost's scikit-learn estimator, via joblib."""

    files = ("model.joblib",)

    def fit(self, recipe, features, labels, *, validation=None):
        model = build_model(recipe)
        model.fit(normalize_features(features, recipe.features), labels)
        return model

    def scores(self, model, features):
        spec = model_feature_spec(model)
        if (
            list(model.classes_) != [0, 1]
            or list(model.feature_names_in_) != spec.columns
            or list(features.columns) != spec.columns
        ):
            raise ValueError("Model does not match its declared feature/class contract")
        values = model.predict_proba(normalize_features(features, spec))[:, 1]
        return checked_probabilities(np.asarray(values, dtype=float), len(features))

    def save(self, model, directory):
        joblib.dump(model, directory / "model.joblib")

    def load(self, directory):
        return joblib.load(directory / "model.joblib")


def checked_probabilities(values: np.ndarray, rows: int) -> np.ndarray:
    """One finite probability in [0, 1] per scored row, or a ValueError."""
    if (
        values.shape != (rows,)
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Model returned invalid fraud probabilities")
    return values


def normalize_features(frame, spec):
    if list(frame.columns) != spec.columns:
        raise ValueError("Input columns do not match the model's feature contract")
    if not spec.categorical:
        return frame
    result = frame.copy(deep=False)
    for name in spec.categorical:
        result[name] = frame[name].astype(object).where(pd.notna(frame[name]), np.nan)
    return result


def xgboost_distribution() -> tuple[str, str]:
    """Installed XGBoost build as (name, version): xgboost-cpu in Linux images."""
    for name in ("xgboost-cpu", "xgboost"):
        try:
            return name, version(name)
        except PackageNotFoundError:
            continue
    raise PackageNotFoundError("xgboost")


def adapter_for(estimator: str) -> ModelAdapter:
    # Every estimator recipes.py builds is scikit-learn-compatible.
    if estimator in BUILDERS:
        return SklearnAdapter()
    if estimator == "mlp":
        # Imported on use so scikit-learn-only processes never load ONNX Runtime.
        from fraud_risk_platform.models.neural import NeuralAdapter

        return NeuralAdapter()
    raise ValueError(f"Unsupported estimator: {estimator}")
