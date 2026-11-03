"""Verified release bundles: saving, loading, and compatibility checks."""

import platform
import re
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fraud_risk_platform.contracts.training import Recipe
from fraud_risk_platform.features.transaction import (
    artifact_feature_spec,
    feature_records,
    model_feature_spec,
)
from fraud_risk_platform.models.adapters import adapter_for, xgboost_distribution
from fraud_risk_platform.persistence.files import (
    identity,
    read_json,
    safe_id,
    sha256,
    verify_files,
    write_json,
)


def runtime_versions(estimator: str | None = None) -> dict:
    versions = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        **{
            name: version(name)
            for name in ["scikit-learn", "numpy", "pandas", "joblib"]
        },
    }
    # Only releases that need a scoring library pin it, so earlier releases keep loading.
    if estimator == "xgboost":
        versions["xgboost"] = xgboost_distribution()[1]
    if estimator == "mlp":
        versions["onnxruntime"] = version("onnxruntime")
    return versions


def load_release(root: Path, release_id: str):
    directory = root / "releases" / safe_id(release_id)
    manifest = read_json(directory / "manifest.json")
    if (
        manifest["release_id"] != release_id
        or identity({k: v for k, v in manifest.items() if k != "release_id"})
        != release_id
    ):
        raise ValueError("Release manifest identity mismatch")
    if (
        manifest.get("dataset") != "ibm-tabformer"
        or not re.fullmatch("[0-9a-f]{64}", manifest.get("source_import_id", ""))
        or manifest["schema_version"] not in {1, 2}
        or manifest["comparison"] != ">="
    ):
        raise ValueError("Incompatible release feature contract")
    spec = artifact_feature_spec(manifest)
    if (
        (manifest["schema_version"] == 1 and spec.version != "basic-v1")
        or (
            manifest["schema_version"] == 2
            and manifest.get("feature_spec_sha256") != identity(spec.model_dump())
        )
        or Recipe.model_validate(manifest["recipe"]).features != spec
    ):
        raise ValueError("Incompatible release feature definition")
    if not 0 <= manifest["threshold"] <= 1 or manifest["classes"] != [0, 1]:
        raise ValueError("Invalid threshold or probability classes")
    if manifest["runtime_versions"] != runtime_versions(
        manifest["recipe"]["estimator"]
    ):
        raise ValueError(
            "Release requires different runtime versions; rebuild with the release lockfile"
        )
    adapter = adapter_for(manifest["recipe"]["estimator"])
    if set(manifest["files"]) != {*adapter.files, "verification.json"}:
        raise ValueError("Release file contract mismatch")
    verify_files(directory, manifest["files"])
    # joblib, like pickle, is executable: only load local trusted bundles after verification.
    model = adapter.load(directory)
    if model_feature_spec(model) != spec:
        raise ValueError(
            "Model preprocessing differs from the saved feature definition"
        )
    sample = read_json(directory / "verification.json")
    scores = adapter.scores(model, pd.DataFrame(sample["inputs"])[spec.columns])
    np.testing.assert_allclose(scores, sample["scores"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(scores >= manifest["threshold"], sample["decisions"])
    return manifest, model


def save_release(
    root: Path, model, metadata: dict, verification_features: pd.DataFrame
) -> dict:
    spec = Recipe.model_validate(metadata["recipe"]).features
    if model_feature_spec(model) != spec:
        raise ValueError(
            "Model preprocessing differs from the release feature definition"
        )
    output = root / "releases"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".release-", dir=output) as temporary:
        stage = Path(temporary)
        adapter = adapter_for(metadata["recipe"]["estimator"])
        adapter.save(model, stage)
        scores = adapter.scores(model, verification_features)
        write_json(
            stage / "verification.json",
            {
                "inputs": feature_records(verification_features),
                "scores": scores.tolist(),
                "decisions": (scores >= metadata["threshold"]).tolist(),
            },
        )
        manifest = {
            "schema_version": 2,
            "feature_version": spec.version,
            "features": spec.columns,
            "feature_spec": spec.model_dump(),
            "feature_spec_sha256": identity(spec.model_dump()),
            "comparison": ">=",
            "classes": [0, 1],
            "runtime_versions": runtime_versions(metadata["recipe"]["estimator"]),
            **metadata,
            "files": {p.name: sha256(p) for p in sorted(stage.iterdir())},
        }
        manifest["release_id"] = identity(manifest)
        write_json(stage / "manifest.json", manifest)
        target = output / manifest["release_id"]
        if not target.exists():
            stage.rename(target)
    load_release(root, manifest["release_id"])
    return manifest
