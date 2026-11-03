"""Real fitting/search/artifact tests, plus snapshot and container integration."""

import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraud_risk_platform.contracts.features import FeatureSet
from fraud_risk_platform.contracts.training import (
    Recipe,
    SearchParameter,
    SearchSettings,
    TrainingPolicy,
)
from fraud_risk_platform.models.adapters import SklearnAdapter
from fraud_risk_platform.models.recipes import check_parameters
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.files import read_json, write_json
from fraud_risk_platform.training.cycle import _add_incumbent_recipe, run_cycle
from fraud_risk_platform.training.inputs import prepare_snapshot
from fraud_risk_platform.training.search import run_search
from fraud_risk_platform.training.tracking import code_version, configure_tracking
from fraud_risk_platform.training.trial import fit_trial, load_trial


@pytest.fixture
def training_frame():
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
            "hour_sin": [0.0] * 6,
            "hour_cos": [1.0] * 6,
        }
    ), pd.Series([0, 0, 0, 1, 1, 1])


@pytest.fixture
def tuned_policy(policy):
    value = policy.model_copy(deep=True)
    value.challengers = []
    value.recipes["logistic_basic"].search_space = {
        "C": SearchParameter(kind="float", low=0.01, high=10, log=True),
        "class_weight": SearchParameter(kind="categorical", choices=[None, "balanced"]),
    }
    value.search = SearchSettings(trials=2, seed=123)
    return value


def test_trial_fits_preprocessing_only_on_training_and_reloads(
    tmp_path, training_frame
):
    X, y = training_frame
    validation = X.copy()
    validation["amount"] += 10
    result = fit_trial(
        recipe=Recipe(estimator="logistic", parameters={"max_iter": 1000}),
        train=(X, y),
        validation=(validation, y),
        review_budget=0.5,
        output=tmp_path / "trial",
    )
    restored = load_trial(tmp_path / "trial")
    assert restored.named_steps["scaler"].mean_[0] == pytest.approx(X.amount.mean())
    assert result["qualified"]
    with (tmp_path / "trial" / "model.joblib").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="Checksum"):
        load_trial(tmp_path / "trial")


def test_training_ap_uses_a_fixed_sample_of_large_partitions(
    tmp_path, training_frame, monkeypatch
):
    from fraud_risk_platform.training import trial

    X, y = training_frame
    monkeypatch.setattr(trial, "TRAIN_AP_ROWS", 4)
    results = [
        fit_trial(
            recipe=Recipe(estimator="logistic", parameters={"max_iter": 1000}),
            train=(X, y),
            validation=(X, y),
            review_budget=0.5,
            output=tmp_path / name,
        )
        for name in ("first", "second")
    ]
    assert [r["train_average_precision_rows"] for r in results] == [4, 4]
    assert (
        results[0]["train_average_precision"] == results[1]["train_average_precision"]
    )


def test_logistic_fits_run_with_capped_threads(tmp_path, training_frame, monkeypatch):
    from threadpoolctl import threadpool_info, threadpool_limits

    seen = []
    original = SklearnAdapter.fit

    def recorded(self, recipe, *args, **kwargs):
        seen.append(max(pool["num_threads"] for pool in threadpool_info()))
        return original(self, recipe, *args, **kwargs)

    monkeypatch.setattr(SklearnAdapter, "fit", recorded)
    with threadpool_limits(limits=4):
        fit_trial(
            recipe=Recipe(estimator="logistic", parameters={"max_iter": 1000}),
            train=training_frame,
            validation=training_frame,
            review_budget=0.5,
            output=tmp_path / "trial",
        )
    assert seen == [2]


def test_unqualified_and_unconverged_trials_are_saved(tmp_path, training_frame):
    X, y = training_frame
    tied = X.assign(amount=1.0)
    unqualified = fit_trial(
        recipe=Recipe(estimator="logistic", parameters={"max_iter": 1000}),
        train=(tied, y),
        validation=(tied, y),
        review_budget=0.5,
        output=tmp_path / "tied",
    )
    # Tied scores exceed the budget, so no threshold fits.
    assert unqualified["threshold"] is None and not unqualified["qualified"]
    assert unqualified["validation"] is None
    # Threshold-free quality is recorded even without a threshold.
    assert unqualified["validation_ranking"]["average_precision"] == pytest.approx(0.5)
    assert unqualified["train_average_precision"] == pytest.approx(0.5)
    assert load_trial(tmp_path / "tied") is not None
    unfinished = fit_trial(
        recipe=Recipe(estimator="logistic", parameters={"max_iter": 1}),
        train=(X, y),
        validation=(X, y),
        review_budget=0.5,
        output=tmp_path / "unfinished",
    )
    assert not unfinished["converged"] and not unfinished["qualified"]


@pytest.mark.parametrize(
    "space",
    [
        {"kind": "float", "low": 0, "high": 1, "log": True},
        {"kind": "int", "low": 1.5, "high": 2},
        {"kind": "categorical", "choices": []},
    ],
)
def test_invalid_search_configuration_rejected(space):
    with pytest.raises(ValidationError):
        SearchParameter.model_validate(space)


def test_search_preserves_every_trial_and_uses_reloadable_score_model(
    tmp_path, training_frame, tuned_policy, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", (tmp_path / "artifacts").as_uri())
    configure_tracking(f"sqlite:///{tmp_path}/mlflow.db")
    with mlflow.start_run() as parent:
        result = run_search(
            name="logistic_basic",
            recipe=tuned_policy.recipes["logistic_basic"],
            settings=tuned_policy.search,
            train=training_frame,
            validation=training_frame,
            review_budget=0.5,
            output=tmp_path / "search",
            snapshot_id="a" * 64,
        )
    assert len(result["trials"]) == 2
    client = mlflow.MlflowClient()
    assert (
        client.get_run(result["run_id"]).data.tags["mlflow.parentRunId"]
        == parent.info.run_id
    )
    for trial in result["trials"]:
        assert (
            client.get_run(trial["run_id"]).data.tags["mlflow.parentRunId"]
            == result["run_id"]
        )
        loaded = mlflow.pyfunc.load_model(trial["model_uri"])
        np.testing.assert_allclose(
            loaded.predict(training_frame[0]).fraud_score,
            load_trial(Path(trial["directory"])).predict_proba(training_frame[0])[:, 1],
        )
    assert (tmp_path / "search" / "study.db").exists()
    trial_metrics = client.get_run(result["trials"][0]["run_id"]).data.metrics
    assert {
        "validation_roc_auc",
        "validation_prevalence",
        "validation_alert_rate",
        "validation_recall_at_top_0.5pct",
        "validation_recall_at_top_50pct",
        "train_average_precision",
    } <= trial_metrics.keys()
    search_run = client.get_run(result["run_id"]).data
    assert search_run.metrics["validation_average_precision"] == max(
        trial["validation_average_precision"] for trial in result["trials"]
    )
    assert search_run.tags["highest_average_precision_trial"].startswith("trial-")


def test_search_starts_from_the_configured_defaults(
    tmp_path, training_frame, tuned_policy, monkeypatch
):
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", (tmp_path / "artifacts").as_uri())
    configure_tracking(f"sqlite:///{tmp_path}/mlflow.db")
    recipe = tuned_policy.recipes["logistic_basic"].model_copy(
        update={"parameters": {"max_iter": 1000, "C": 0.5}}
    )
    with mlflow.start_run():
        result = run_search(
            name="logistic_basic",
            recipe=recipe,
            settings=tuned_policy.search,
            train=training_frame,
            validation=training_frame,
            review_budget=0.5,
            output=tmp_path / "search",
            snapshot_id="a" * 64,
        )
    first, second = (trial["recipe"]["parameters"] for trial in result["trials"])
    # Only configured values are fixed; class_weight has none and is still sampled.
    assert first["C"] == 0.5 and second["C"] != 0.5
    assert "class_weight" in first


@pytest.mark.parametrize(
    "key, default, space",
    [
        ("C", 1000.0, {"kind": "float", "low": 0.01, "high": 10, "log": True}),
        ("max_iter", 2.5, {"kind": "int", "low": 1, "high": 10}),
        (
            "class_weight",
            "auto",
            {"kind": "categorical", "choices": [None, "balanced"]},
        ),
    ],
)
def test_searched_defaults_must_lie_in_their_search_space(key, default, space):
    with pytest.raises(ValidationError, match="outside its search space"):
        Recipe(
            estimator="logistic", parameters={key: default}, search_space={key: space}
        )


@pytest.mark.parametrize(
    "recipe, message",
    [
        (
            {"estimator": "logistic", "parameters": {"max_iterations": 5}},
            r"Unknown logistic parameters: \['max_iterations'\]",
        ),
        (
            {
                "estimator": "logistic",
                "parameters": {},
                "search_space": {"Cee": {"kind": "float", "low": 0.1, "high": 1}},
            },
            r"Unknown logistic parameters: \['Cee'\]",
        ),
        (
            {
                "estimator": "hist_gradient_boosting",
                "parameters": {"categorical_features": []},
            },
            "feature set determines",
        ),
        # Rejected before XGBoost is imported, so this case needs no OpenMP runtime.
        (
            {"estimator": "xgboost", "parameters": {"feature_types": []}},
            "feature set determines",
        ),
    ],
)
def test_settings_an_estimator_lacks_or_the_features_determine_are_rejected(
    recipe, message
):
    with pytest.raises(ValueError, match=message):
        check_parameters(Recipe.model_validate(recipe))


def test_misnamed_settings_fail_before_the_cycle_starts(tmp_path, policy, monkeypatch):
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", (tmp_path / "artifacts").as_uri())
    data = policy.model_dump()
    data["recipes"]["boosting_basic"]["parameters"]["max_leafs"] = 7
    with pytest.raises(
        ValueError, match=r"Unknown hist_gradient_boosting parameters: \['max_leafs'\]"
    ):
        run_cycle(
            root=tmp_path,
            snapshot="0" * 64,
            policy=TrainingPolicy.model_validate(data),
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        )
    # The check precedes the cycle's run, its outputs, and any snapshot access.
    assert not (tmp_path / "cycles").exists()


SHIPPED_POLICY = Path(__file__).parents[1] / "config" / "training.json"


@pytest.mark.parametrize("name", json.loads(SHIPPED_POLICY.read_text())["recipes"])
def test_shipped_recipes_name_only_real_estimator_settings(name):
    # The xgboost_history case imports XGBoost, which needs libomp on macOS.
    policy = TrainingPolicy.model_validate_json(SHIPPED_POLICY.read_text())
    check_parameters(policy.resolve_recipe(name))


def test_incumbent_is_retrained_unless_a_recipe_shares_its_encoding():
    spec = FeatureSet(numeric=["amount", "log_abs_amount"])

    def recipe(encoding):
        return Recipe(
            estimator="logistic",
            numeric_encoding=encoding,
            feature_set="details",
            feature_spec=spec,
            parameters={"C": 1.0},
        )

    incumbent = {"recipe": recipe("standard").model_dump()}
    recipes = {"spline": recipe("quantile_spline")}
    _add_incumbent_recipe(recipes, incumbent)
    assert recipes["incumbent_retrained"].numeric_encoding == "standard"
    recipes = {"logistic": recipe("standard")}
    _add_incumbent_recipe(recipes, incumbent)
    assert list(recipes) == ["logistic"]


def test_code_version_comes_from_git_or_the_image_build(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    # A training image has no Git checkout; its build records the commit instead.
    monkeypatch.setattr(subprocess, "check_output", no_git)
    monkeypatch.setenv("TRAINING_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("TRAINING_GIT_DIRTY", "false")
    assert code_version() == {"git_commit": "a" * 40, "working_tree_dirty": False}
    monkeypatch.setenv("TRAINING_GIT_DIRTY", "true")
    assert code_version()["working_tree_dirty"] is True
    monkeypatch.delenv("TRAINING_GIT_COMMIT")
    monkeypatch.delenv("TRAINING_GIT_DIRTY")
    assert code_version() == {
        "git_commit": "unavailable",
        "working_tree_dirty": "unavailable",
    }


@pytest.mark.integration
def test_full_cycle_refit_s3_artifacts_and_local_channel(
    tmp_path, snapshot_job, dataset_profile, object_store, tuned_policy, monkeypatch
):
    sampled = snapshot_job()
    full = snapshot_job(
        profile=dataset_profile.model_copy(update={"max_train_rows": None}),
        chunk_size=7,
    )
    channel = tmp_path / "channel"
    prepare_snapshot(sampled["snapshot_id"], channel)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", object_store.uri("training-artifacts"))
    monkeypatch.setenv("MLFLOW_S3_ENDPOINT_URL", os.environ["TEST_S3_ENDPOINT_URL"])
    fits = []
    original = SklearnAdapter.fit

    def recorded(self, recipe, features, labels, **kwargs):
        fits.append(len(labels))
        return original(self, recipe, features, labels, **kwargs)

    monkeypatch.setattr(SklearnAdapter, "fit", recorded)
    report = run_cycle(
        root=tmp_path,
        snapshot=channel,
        policy=tuned_policy,
        refit_snapshot=full["snapshot_id"],
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    assert fits == [64, 64, full["splits"]["train"]["rows"]]
    assert report["recommendation"] == "initial_promotion"
    assert not (tmp_path / "deployment.json").exists()
    manifest, model = load_release(tmp_path, report["release_id"])
    assert manifest["snapshot_id"] == full["snapshot_id"]
    assert manifest["search_snapshot_id"] == sampled["snapshot_id"]
    assert manifest["threshold"] == report["refit"]["threshold"]
    assert model is not None
    registered = mlflow.MlflowClient().get_model_version_by_alias(
        "fraud-risk", "candidate"
    )
    assert registered.version == report["registry"]["version"]
    assert registered.tags["qualification"] == "passed"
    downloaded = mlflow.pyfunc.load_model(report["selected_model_uri"])
    assert downloaded is not None
    assert report["plots"] == ["validation.png", "trials.png", "evaluation.png"]
    assert report["validation_summary"]["source"] == "selected_candidate"
    assert report["validation_summary"]["run_id"] == report["refit"]["run_id"]
    assert {"roc_auc", "recall_at_top_0.5pct"} <= report["evaluation"][
        "candidate"
    ].keys()
    assert (
        "data:image/png;base64,"
        in (tmp_path / "cycles" / report["run_id"] / "report.html").read_text()
    )


@pytest.mark.integration
def test_unqualified_cycle_still_records_validation_quality(
    tmp_path, snapshot_job, tuned_policy, monkeypatch
):
    from fraud_risk_platform.training import trial

    monkeypatch.setattr(trial, "select_threshold", lambda *args, **kwargs: None)
    snapshot = snapshot_job()
    report = run_cycle(
        root=tmp_path,
        snapshot=snapshot["snapshot_id"],
        policy=tuned_policy,
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    assert report["recommendation"] == "no_initial_model"
    # The evaluation partition stays unscored when nothing is selected.
    assert report["evaluation"] == {"candidate": None, "incumbent": None}
    summary = report["validation_summary"]
    assert summary["source"] == "highest_average_precision_unqualified"
    assert summary["model"].startswith("logistic_basic trial-")
    # Charts cover validation even without a candidate; evaluation stays unscored.
    assert report["plots"] == ["validation.png", "trials.png"]
    assert summary["average_precision"] == max(
        trial["validation_average_precision"]
        for trial in report["searches"]["logistic_basic"]["trials"]
    )
    logged = mlflow.MlflowClient().get_run(report["run_id"]).data.metrics
    assert logged["validation_average_precision"] == summary["average_precision"]
    assert "validation_recall_at_top_0.5pct" in logged
    assert (
        "Validation summary"
        in (tmp_path / "cycles" / report["run_id"] / "report.html").read_text()
    )


@pytest.mark.integration
def test_incompatible_refit_rejected_before_fitting(
    tmp_path, snapshot_job, dataset_profile, tuned_policy, monkeypatch
):
    sampled = snapshot_job()
    later = snapshot_job(
        as_of="2019-02-20",
        profile=dataset_profile.model_copy(update={"max_train_rows": None}),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("An incompatible refit must fail before fitting")

    monkeypatch.setattr(SklearnAdapter, "fit", forbidden)
    with pytest.raises(ValueError, match="Refit snapshot must preserve"):
        run_cycle(
            root=tmp_path,
            snapshot=sampled["snapshot_id"],
            policy=tuned_policy,
            refit_snapshot=later["snapshot_id"],
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        )


@pytest.mark.integration
def test_unconverged_refit_cannot_fall_back_to_sampled_winner(
    tmp_path, snapshot_job, dataset_profile, tuned_policy, monkeypatch
):
    from fraud_risk_platform.training import cycle

    sampled = snapshot_job()
    full = snapshot_job(
        profile=dataset_profile.model_copy(update={"max_train_rows": None})
    )
    original = cycle.fit_trial

    def unfinished_refit(**kwargs):
        configured = kwargs["recipe"].model_copy(deep=True)
        configured.parameters["max_iter"] = 1
        return original(**{**kwargs, "recipe": configured})

    monkeypatch.setattr(cycle, "fit_trial", unfinished_refit)
    report = run_cycle(
        root=tmp_path,
        snapshot=sampled["snapshot_id"],
        policy=tuned_policy,
        refit_snapshot=full["snapshot_id"],
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    assert report["validation_winner"] == "logistic_basic"
    assert not report["refit"]["converged"]
    assert report["recommendation"] == "no_initial_model"
    assert report["evaluation"]["candidate"] is None
    assert not (tmp_path / "deployment.json").exists()
    registered = mlflow.MlflowClient().get_model_version_by_alias(
        "fraud-risk", "candidate"
    )
    assert registered.tags["qualification"] == "failed"


@pytest.mark.integration
def test_sagemaker_entrypoint_uses_prepared_files(
    tmp_path, snapshot_job, policy, monkeypatch
):
    policy.feature_sets["details"] = FeatureSet(
        numeric=["amount", "log_abs_amount"], categorical=["use_chip", "mcc"]
    )
    policy.recipes["logistic_basic"].feature_set = "details"
    snapshot = snapshot_job()
    channels = tmp_path / "channels"
    channels.mkdir()
    prepare_snapshot(snapshot["snapshot_id"], channels / "snapshot")
    (channels / "config").mkdir()
    write_json(
        channels / "config" / "job.json",
        {"policy": policy.model_dump(), "recipe": "logistic_basic", "trials": 1},
    )
    output = tmp_path / "output"
    output.mkdir()
    environment = {
        **os.environ,
        "TRAINING_CHANNEL_ROOT": str(channels),
        "SM_OUTPUT_DIR": str(output),
        "SM_MODEL_DIR": str(tmp_path / "model"),
        "MLFLOW_TRACKING_URI": f"sqlite:///{output}/mlflow.db",
        "MLFLOW_ARTIFACT_URI": (output / "artifacts").as_uri(),
        "SOURCE_DATABASE_URL": "postgresql://unavailable.invalid/source",
        "AWS_ENDPOINT_URL_S3": "http://unavailable.invalid:9",
    }
    result = subprocess.run(
        [sys.executable, "-m", "fraud_risk_platform.training.entrypoint", "train"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert read_json(tmp_path / "model" / "manifest.json")["promotion_eligible"]
    assert (
        read_json(tmp_path / "model" / "manifest.json")["feature_version"]
        == "transaction-v1"
    )
    image = os.environ.get("TRAINING_CONTRACT_IMAGE")
    if image:
        container_output = tmp_path / "container-output"
        container_model = tmp_path / "container-model"
        container_output.mkdir()
        container_model.mkdir()
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{channels}:/opt/ml/input/data:ro",
            "-v",
            f"{container_output}:/opt/ml/output",
            "-v",
            f"{container_model}:/opt/ml/model",
            "-e",
            "MLFLOW_TRACKING_URI=sqlite:////opt/ml/output/data/mlflow.db",
            "-e",
            "MLFLOW_ARTIFACT_URI=file:///opt/ml/output/data/artifacts",
            image,
            "train",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert read_json(container_model / "manifest.json")["promotion_eligible"]
        assert (
            read_json(container_model / "manifest.json")["feature_version"]
            == "transaction-v1"
        )
        write_json(
            channels / "config" / "job.json",
            {"policy": policy.model_dump(), "recipe": "logistic_basic", "trials": 0},
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert "ValidationError" in (container_output / "failure").read_text()
