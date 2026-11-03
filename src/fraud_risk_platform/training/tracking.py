"""Shared MLflow evidence and a framework-independent fraud-score signature."""

import os
import subprocess
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

from fraud_risk_platform.features.transaction import artifact_feature_spec
from fraud_risk_platform.models.adapters import adapter_for, xgboost_distribution
from fraud_risk_platform.persistence.files import read_json, sha256
from fraud_risk_platform.training.trial import load_trial


class FraudScoreModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        directory = Path(context.artifacts["bundle"])
        self.adapter = adapter_for(
            read_json(directory / "trial.json")["recipe"]["estimator"]
        )
        self.model = load_trial(directory)
        self.features = artifact_feature_spec(read_json(directory / "trial.json"))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fraud_score": self.adapter.scores(
                    self.model, model_input[self.features.columns]
                )
            },
            index=model_input.index,
        )


def experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-risk-training-v2")


def registered_model_name() -> str:
    return os.environ.get("MLFLOW_REGISTERED_MODEL", "fraud-risk")


def configure_tracking(uri):
    mlflow.set_tracking_uri(uri or os.environ["MLFLOW_TRACKING_URI"])
    name = experiment_name()
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    destination = os.environ.get("MLFLOW_ARTIFACT_URI")
    if experiment is None:
        client.create_experiment(name, artifact_location=destination)
    elif destination and experiment.artifact_location.rstrip("/") != destination.rstrip(
        "/"
    ):
        raise ValueError(
            "Experiment uses a different artifact store; select a new MLFLOW_EXPERIMENT_NAME"
        )
    mlflow.set_experiment(name)


def source_checksums() -> dict:
    package = Path(__file__).parents[1]
    return {
        str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*.py"))
    }


def dependency_versions() -> dict:
    dependencies = {
        name: version(name)
        for name in [
            "optuna",
            "mlflow",
            "scikit-learn",
            "numpy",
            "pandas",
            "joblib",
            "pyarrow",
            "torch",
            "onnx",
            "onnxruntime",
        ]
    }
    name, installed = xgboost_distribution()
    dependencies[name] = installed
    return dependencies


def code_version() -> dict:
    """The Git commit of the running code and whether the working tree had changes.

    A checkout answers directly. An image has none, so it reports what its build
    recorded (TRAINING_GIT_COMMIT and TRAINING_GIT_DIRTY), or "unavailable".
    """
    try:
        return {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            ),
        }
    except OSError, subprocess.CalledProcessError:
        dirty = os.environ.get("TRAINING_GIT_DIRTY", "unavailable")
        return {
            "git_commit": os.environ.get("TRAINING_GIT_COMMIT", "unavailable"),
            "working_tree_dirty": {"true": True, "false": False}.get(dirty, dirty),
        }


def log_provenance():
    mlflow.log_dict(source_checksums(), "source/checksums.json")
    provenance = {
        "dependencies": dependency_versions(),
        # Fits can differ in the last bits across thread counts; record them.
        "threads": {
            name: os.environ.get(name, "unset")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "image_identity": os.environ.get("TRAINING_IMAGE_ID", "unrecorded"),
        **code_version(),
    }
    mlflow.log_dict(provenance, "environment/provenance.json")
    for filename in ["pyproject.toml", "uv.lock", "config/ibm-tabformer.lock.json"]:
        if Path(filename).exists():
            mlflow.log_artifact(filename, "environment")
    return provenance


def log_trial(directory, result, example):
    features = artifact_feature_spec(result)
    mlflow.log_dict(features.model_dump(), "features.json")
    mlflow.set_tags(
        {
            "feature_set": result["recipe"].get("feature_set", "basic-v1"),
            "feature_version": features.version,
            "numeric_encoding": result["recipe"].get("numeric_encoding", "standard"),
        }
    )
    mlflow.log_params(result["recipe"]["parameters"])
    mlflow.log_metrics(
        {
            **{
                key: result[key]
                for key in [
                    "training_seconds",
                    "validation_inference_seconds",
                    "training_rows",
                    "validation_rows",
                    "train_average_precision",
                ]
            },
            **ranking_fields(result),
        }
    )
    for name, values in (result.get("training_curve") or {}).items():
        for epoch, value in enumerate(values, start=1):
            mlflow.log_metric(name, value, step=epoch)
    if result["validation"]:
        # Metrics at the selected threshold; rows/average precision are logged above.
        mlflow.log_metrics(
            {
                f"validation_{k}": v
                for k, v in result["validation"].items()
                if k not in {"rows", "average_precision"}
            }
        )
    mlflow.set_tags(
        {
            "qualified": str(result["qualified"]).lower(),
            "converged": str(result["converged"]).lower(),
        }
    )
    # The bundle is the authoritative model artifact; releases retain a local serving cache.
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=FraudScoreModel(),
        artifacts={"bundle": str(directory)},
        input_example=example[features.columns],
        signature=ModelSignature(
            inputs=Schema(
                [ColSpec(DataType.double, name) for name in features.numeric]
                + [ColSpec(DataType.string, name) for name in features.categorical]
            ),
            outputs=Schema([ColSpec(DataType.double, "fraud_score")]),
        ),
        code_paths=[str(Path(__file__).parents[1])],
        pip_requirements=[
            f"{name}=={version(name)}"
            for name in [
                "mlflow",
                "scikit-learn",
                "numpy",
                "pandas",
                "joblib",
                "pydantic",
            ]
        ]
        + {
            "xgboost": ["{}=={}".format(*xgboost_distribution())],
            # Neural models score with ONNX Runtime; PyTorch is training-only.
            "mlp": [f"onnxruntime=={version('onnxruntime')}"],
        }.get(result["recipe"]["estimator"], []),
    )
    return info.model_uri


def ranking_fields(result):
    """Always-defined validation quality of a fitted trial, as MLflow metric names."""
    return {f"validation_{k}": v for k, v in result["validation_ranking"].items()}


def set_champion(registry: dict, tracking_uri: str | None = None) -> dict:
    """Point the `champion` alias at a release confirmed serving.

    The alias records deployment; it does not deploy. Serving reads only local
    releases, so a registry outage never blocks or changes the API.
    """
    from datetime import datetime, timezone

    mlflow.set_tracking_uri(tracking_uri or os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(registry["name"], "champion", registry["version"])
    client.set_model_version_tag(
        registry["name"],
        registry["version"],
        "confirmed_serving_at",
        datetime.now(timezone.utc).isoformat(),
    )
    return {
        "alias": "champion",
        "name": registry["name"],
        "version": registry["version"],
    }


def archive_tracking(suffix: str, tracking_uri: str | None = None) -> dict:
    """Rename the experiment and registered model out of a new simulation's way.

    Their runs and versions stay readable under the new names; the next cycle creates
    a fresh experiment, and the next candidate a fresh model, under the configured ones.
    """
    mlflow.set_tracking_uri(tracking_uri or os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.MlflowClient()
    archived = {}
    experiment = client.get_experiment_by_name(experiment_name())
    if experiment is not None:
        archived["experiment"] = f"{experiment.name}-archived-{suffix}"
        client.rename_experiment(experiment.experiment_id, archived["experiment"])
    model = registered_model_name()
    if client.search_registered_models(filter_string=f"name = '{model}'"):
        archived["registered_model"] = f"{model}-archived-{suffix}"
        client.rename_registered_model(model, archived["registered_model"])
    return archived


def register_candidate(model_uri, *, cycle_run_id, qualified):
    name = registered_model_name()
    registered = mlflow.register_model(model_uri, name)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(name, registered.version, "cycle_run_id", cycle_run_id)
    client.set_model_version_tag(
        name, registered.version, "qualification", "passed" if qualified else "failed"
    )
    client.set_registered_model_alias(name, "candidate", registered.version)
    return {
        "name": name,
        "version": registered.version,
        "uri": f"models:/{name}/{registered.version}",
    }
