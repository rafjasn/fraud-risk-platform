"""Fitted category mappings and feature contracts survive training and serving."""

from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_risk_platform.contracts.features import (
    HISTORY_FEATURE_INPUTS,
    INDICATOR_FEATURES,
    NUMERIC_FEATURE_INPUTS,
    FeatureSet,
)
from fraud_risk_platform.contracts.training import Recipe, TrainingPolicy
from fraud_risk_platform.features.transaction import build_transaction_features
from fraud_risk_platform.models.adapters import (
    SklearnAdapter,
    adapter_for,
    checked_probabilities,
)
from fraud_risk_platform.models.releases import load_release, save_release
from fraud_risk_platform.monitoring.outcomes import query_outcomes
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.persistence.files import identity, write_json
from fraud_risk_platform.serving.app import create_app
from fraud_risk_platform.serving.predictor import FraudPredictor
from fraud_risk_platform.training.cycle import run_cycle
from fraud_risk_platform.training.tracking import configure_tracking, log_trial
from fraud_risk_platform.training.trial import fit_trial, load_trial


@pytest.fixture
def transaction_spec():
    return FeatureSet(
        numeric=["amount", "log_abs_amount", "is_refund", "hour_sin", "hour_cos"],
        categorical=["use_chip", "mcc", "merchant_state", "day_of_week"],
    )


@pytest.fixture
def category_data():
    # Only channel distinguishes the classes; amount/time cannot solve this task.
    return pd.DataFrame(
        {
            "amount": [10.0] * 40,
            "tx_datetime": pd.to_datetime(["2019-01-01T12:00:00"] * 40),
            "use_chip": ["Chip Transaction", "Online Transaction"] * 20,
            "mcc": ["0123"] * 40,
            "merchant_state": [None] * 40,
        }
    ), pd.Series([0, 1] * 20)


ESTIMATOR_PARAMETERS = {
    "logistic": {"max_iter": 1000, "C": 10.0},
    "hist_gradient_boosting": {
        "max_iter": 20,
        "min_samples_leaf": 2,
        "max_bins": 8,
        "early_stopping": False,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 20,
        "max_depth": 3,
        "min_child_weight": 0.01,
        "random_state": 42,
    },
    "mlp": {
        "hidden_units": 8,
        "hidden_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.05,
        "epochs": 60,
        "batch_size": 8,
        "random_state": 0,
    },
}


def configured_recipe(spec, estimator="logistic"):
    return Recipe(
        estimator=estimator,
        feature_set="details",
        feature_spec=spec,
        parameters=ESTIMATOR_PARAMETERS[estimator],
    )


def release_metadata(recipe):
    return {
        "recipe": recipe.model_dump(),
        "source_run_id": "fixture-run",
        "dataset": "ibm-tabformer",
        "source_import_id": "a" * 64,
        "threshold": 0.5,
        "promotion_eligible": False,
    }


@pytest.mark.parametrize(
    "definition",
    [
        {"numeric": ["is_fraud"]},
        {"numeric": ["amount", "amount"]},
        {"numeric": ["mcc"]},
        {"categorical": ["label_available_at"]},
        {},
        {"version": "basic-v1", "numeric": ["amount"]},
    ],
)
def test_rejects_unsupported_or_ambiguous_features(definition):
    with pytest.raises(ValidationError):
        FeatureSet.model_validate(definition)


def test_policy_resolves_and_freezes_feature_sets(policy, transaction_spec):
    data = policy.model_dump()
    data["recipes"]["logistic_basic"]["feature_set"] = "details"
    with pytest.raises(ValidationError, match="Unknown feature set"):
        TrainingPolicy.model_validate(data)
    data["feature_sets"] = {"details": transaction_spec.model_dump()}
    configured = TrainingPolicy.model_validate(data)
    recipe = configured.resolve_recipe("logistic_basic")
    configured.feature_sets["details"].categorical.clear()
    assert recipe.features == transaction_spec


def test_transforms_signed_amounts_time_and_large_category_ids():
    spec = FeatureSet(
        numeric=["log_abs_amount", "is_refund", "is_weekend", "weekday_sin"],
        categorical=["merchant_id", "mcc", "day_of_week"],
    )
    frame = pd.DataFrame(
        {
            "amount": [-10.0, 0.0],
            "tx_datetime": pd.to_datetime(
                ["2019-01-05T12:00:00", "2019-01-07T00:00:00"]
            ),
            "merchant_id": [-(2**63), 2**63 - 1],
            "mcc": ["0123", "5411"],
        }
    )
    values = build_transaction_features(frame, spec)
    assert values.log_abs_amount.tolist() == [np.log1p(10), 0]
    assert values.is_refund.tolist() == [1, 0]
    assert values.is_weekend.tolist() == [1, 0]
    assert values.mcc.tolist() == ["0123", "5411"]
    assert values.day_of_week.tolist() == ["5", "0"]
    assert values.merchant_id.tolist() == [str(-(2**63)), str(2**63 - 1)]


@pytest.mark.parametrize("estimator", list(ESTIMATOR_PARAMETERS))
def test_saved_preprocessing_handles_unseen_and_missing_categories(
    tmp_path, transaction_spec, category_data, estimator
):
    raw, labels = category_data
    recipe = configured_recipe(transaction_spec, estimator)
    X = build_transaction_features(raw, transaction_spec)
    result = fit_trial(
        recipe=recipe,
        train=(X, labels),
        validation=(X, labels),
        review_budget=0.5,
        output=tmp_path / "trial",
    )
    assert result["qualified"]
    model = load_trial(tmp_path / "trial")
    adapter = adapter_for(estimator)
    assert adapter.scores(model, X)[1] > adapter.scores(model, X)[0]
    unseen = raw.iloc[:2].copy()
    unseen["mcc"] = ["9999", "0123"]
    unseen["merchant_state"] = ["NEW", None]
    features = build_transaction_features(unseen, transaction_spec)
    if estimator in {"logistic", "mlp"}:
        # The MLP shares logistic regression's one-hot preprocessing.
        preprocessor = (
            model.preprocessor
            if estimator == "mlp"
            else model.named_steps["preprocessor"]
        )
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    else:
        encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
    assert "9999" not in encoder.categories_[1]
    release = save_release(tmp_path, model, release_metadata(recipe), features)
    manifest, restored = load_release(tmp_path, release["release_id"])
    # Only releases that need a scoring library pin it, so earlier releases keep loading.
    assert ("xgboost" in manifest["runtime_versions"]) == (estimator == "xgboost")
    assert ("onnxruntime" in manifest["runtime_versions"]) == (estimator == "mlp")
    np.testing.assert_allclose(
        adapter.scores(model, features), adapter.scores(restored, features)
    )
    predictor = FraudPredictor(root=tmp_path, release_id=release["release_id"])
    scores = []
    for row in unseen.to_dict(orient="records"):
        row["timestamp"] = row.pop("tx_datetime").to_pydatetime()
        scores.append(predictor.predict(**row)["fraud_score"])
    np.testing.assert_allclose(scores, adapter.scores(model, features))
    with pytest.raises(ValueError, match="Missing required transaction fields"):
        predictor.predict(amount=10.0, timestamp=datetime(2019, 1, 1))
    with pytest.raises(ValueError, match="cannot be null"):
        predictor.predict(
            amount=10.0,
            timestamp=datetime(2019, 1, 1),
            use_chip=None,
            mcc="0123",
            merchant_state=None,
        )


@pytest.mark.parametrize("estimator", ["logistic", "xgboost", "mlp"])
def test_mlflow_reload_preserves_feature_contract(
    tmp_path, transaction_spec, category_data, monkeypatch, estimator
):
    raw, labels = category_data
    X = build_transaction_features(raw, transaction_spec)
    recipe = configured_recipe(transaction_spec, estimator)
    directory = tmp_path / "trial"
    result = fit_trial(
        recipe=recipe,
        train=(X, labels),
        validation=(X, labels),
        review_budget=0.5,
        output=directory,
    )
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", (tmp_path / "mlflow-artifacts").as_uri())
    configure_tracking(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run():
        uri = log_trial(directory, result, X.iloc[:5])
    loaded = mlflow.pyfunc.load_model(uri)
    np.testing.assert_allclose(
        loaded.predict(X).fraud_score,
        adapter_for(estimator).scores(load_trial(directory), X),
    )
    requirements = Path(mlflow.pyfunc.get_model_dependencies(uri)).read_text()
    assert ("xgboost" in requirements) == (estimator == "xgboost")
    # Neural models are served through ONNX Runtime, never PyTorch.
    assert ("onnxruntime" in requirements) == (estimator == "mlp")
    assert "torch" not in requirements


def test_spline_encoding_follows_each_feature_curve_and_survives_release(tmp_path):
    spec = FeatureSet(
        numeric=["amount", "is_refund", "hour_sin"], categorical=["use_chip"]
    )
    amounts = np.linspace(1.0, 100.0, 200)
    raw = pd.DataFrame(
        {
            "amount": amounts,
            "tx_datetime": pd.to_datetime(["2019-01-01T12:00:00"] * 200),
            "use_chip": ["Chip Transaction", "Online Transaction"] * 100,
        }
    )
    # Fraud sits in a middle band of amounts, which no single coefficient can rank.
    labels = pd.Series(((amounts > 40) & (amounts < 60)).astype(int))
    X = build_transaction_features(raw, spec)
    results = {}
    for encoding in ("standard", "quantile_spline"):
        recipe = Recipe(
            estimator="logistic",
            numeric_encoding=encoding,
            feature_set="details",
            feature_spec=spec,
            parameters={"solver": "newton-cholesky", "C": 10.0},
        )
        results[encoding] = fit_trial(
            recipe=recipe,
            train=(X, labels),
            validation=(X, labels),
            review_budget=0.3,
            output=tmp_path / encoding,
        )
    assert results["standard"]["validation_average_precision"] < 0.5
    assert results["quantile_spline"]["validation_average_precision"] > 0.95
    model = load_trial(tmp_path / "quantile_spline")
    numeric = model.named_steps["preprocessor"].named_transformers_["numeric"]
    # Continuous features get spline bases; the 0/1 refund flag passes through.
    assert [(name, columns) for name, _, columns in numeric.transformers_] == [
        ("curves", ["amount", "hour_sin"]),
        ("indicators", ["is_refund"]),
    ]
    release = save_release(tmp_path, model, release_metadata(recipe), X.iloc[:20])
    _, restored = load_release(tmp_path, release["release_id"])
    np.testing.assert_allclose(
        SklearnAdapter().scores(restored, X), SklearnAdapter().scores(model, X)
    )


def test_spline_encoding_needs_logistic_regression_on_configured_features(
    transaction_spec,
):
    with pytest.raises(ValidationError, match="quantile_spline"):
        Recipe(
            estimator="hist_gradient_boosting",
            numeric_encoding="quantile_spline",
            feature_set="details",
            feature_spec=transaction_spec,
            parameters={},
        )
    with pytest.raises(ValidationError, match="quantile_spline"):
        Recipe(estimator="logistic", numeric_encoding="quantile_spline", parameters={})
    assert (
        INDICATOR_FEATURES
        <= NUMERIC_FEATURE_INPUTS.keys() | HISTORY_FEATURE_INPUTS.keys()
    )


@pytest.mark.parametrize(
    "values",
    [[0.5], [0.5, np.nan], [0.5, 1.5], [-0.1, 0.5]],
)
def test_scores_must_be_one_probability_per_row(values):
    with pytest.raises(ValueError, match="invalid fraud probabilities"):
        checked_probabilities(np.asarray(values), rows=2)


def test_original_release_schema_still_loads(tmp_path, release_factory):
    release = release_factory()
    directory = tmp_path / "releases" / release["release_id"]
    legacy = dict(release)
    legacy["schema_version"] = 1
    legacy.pop("feature_spec")
    legacy.pop("feature_spec_sha256")
    legacy["recipe"].pop("feature_set")
    legacy["recipe"].pop("feature_spec")
    legacy["release_id"] = identity(
        {k: v for k, v in legacy.items() if k != "release_id"}
    )
    write_json(directory / "manifest.json", legacy)
    directory.rename(directory.parent / legacy["release_id"])
    predictor = FraudPredictor(root=tmp_path, release_id=legacy["release_id"])
    assert (
        predictor.predict(amount=20.0, timestamp=datetime(2018, 4, 1))[
            "feature_version"
        ]
        == "basic-v1"
    )


@pytest.mark.integration
def test_feature_sets_compete_and_refit_against_basic_incumbent(
    tmp_path, snapshot_job, dataset_profile, policy, transaction_spec, release_factory
):
    sampled = snapshot_job()
    full = snapshot_job(
        profile=dataset_profile.model_copy(update={"max_train_rows": None})
    )
    data = policy.model_dump()
    data["feature_sets"] = {"details": transaction_spec.model_dump()}
    data["recipes"]["logistic_details"] = {
        "estimator": "logistic",
        "parameters": {"max_iter": 1000},
        "feature_set": "details",
    }
    # The first recipe wins the recall/precision tie, exercising a different last-loaded feature set.
    data["initial_recipe"] = "logistic_details"
    data["challengers"] = ["logistic_basic"]
    incumbent = release_factory()
    report = run_cycle(
        root=tmp_path,
        snapshot=sampled["snapshot_id"],
        refit_snapshot=full["snapshot_id"],
        policy=TrainingPolicy.model_validate(data),
        incumbent_release_id=incumbent["release_id"],
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    assert report["validation_winner"] == "logistic_details"
    assert report["refit"]["training_rows"] == full["splits"]["train"]["rows"]
    assert report["evaluation"]["incumbent"] is not None
    assert report["evaluation"]["candidate"] is not None
    manifest, _ = load_release(tmp_path, report["release_id"])
    assert manifest["feature_spec"] == transaction_spec.model_dump()
    assert manifest["recipe"]["feature_set"] == "details"
    assert (
        report["searches"]["logistic_basic"]["trials"][0]["feature_version"]
        == "basic-v1"
    )


@pytest.mark.integration
def test_api_requires_and_records_model_inputs(
    tmp_path, transaction_spec, category_data, database, source_database, ibm_history
):
    raw, labels = category_data
    X = build_transaction_features(raw, transaction_spec)
    recipe = configured_recipe(transaction_spec)
    model = SklearnAdapter().fit(recipe, X, labels)
    release = save_release(tmp_path, model, release_metadata(recipe), X.iloc[:3])
    payload = {
        "transaction_id": 1,
        "amount": -10.50,
        "timestamp": "2019-01-01T12:00:00",
        "use_chip": "Online Transaction",
        "mcc": "5411",
        "merchant_state": None,
    }
    with TestClient(
        create_app(
            root=tmp_path,
            database_url=database,
            source_database_url=source_database,
            release_id=release["release_id"],
        )
    ) as client:
        missing = {k: v for k, v in payload.items() if k != "use_chip"}
        assert client.post("/predict", json=missing).status_code == 422
        assert client.post("/predict", json={**payload, "mcc": 5411}).status_code == 422
        assert (
            client.post("/predict", json={**payload, "mcc": "9999"}).status_code == 409
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["feature_version"] == "transaction-v1"
        # Unknown source rows are allowed, with their actual category inputs retained.
        assert (
            client.post(
                "/predict", json={**payload, "transaction_id": 999999, "mcc": "9999"}
            ).status_code
            == 200
        )
    with connect(database) as conn:
        stored = conn.execute(
            "SELECT feature_inputs FROM predictions WHERE transaction_id=1"
        ).fetchone()
        assert stored["feature_inputs"] == {
            "use_chip": "Online Transaction",
            "mcc": "5411",
            "merchant_state": None,
        }
    outcomes = query_outcomes(
        datetime(2019, 1, 4), database, source_url=source_database
    )
    assert sum(row["fraud"] is not None for row in outcomes) == 1
    with connect(database) as conn:
        conn.execute(
            "UPDATE predictions SET feature_inputs=jsonb_set(feature_inputs, '{mcc}', '\"9999\"') WHERE transaction_id=1"
        )
    assert all(
        row["fraud"] is None
        for row in query_outcomes(
            datetime(2019, 1, 4), database, source_url=source_database
        )
    )
