"""Release integrity, predictor round trips, promotion, and rollback."""

from datetime import datetime

import numpy as np
import pytest

from fraud_risk_platform.models.deployment import (
    confirm_rollout,
    deployment,
    promote,
    rollback,
)
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.files import read_json, write_json
from fraud_risk_platform.serving.predictor import FraudPredictor


def test_real_artifact_saved_threshold_and_features(tmp_path, release_factory):
    release = release_factory(threshold=0.0)
    predictor = FraudPredictor(root=tmp_path, release_id=release["release_id"])
    result = predictor.predict(amount=20, timestamp=datetime(2018, 4, 1))
    assert result["flagged_for_review"] is True
    assert result["threshold"] == 0.0
    assert result["release_id"] == release["release_id"]
    features = predictor.model.feature_names_in_
    np.testing.assert_array_equal(features, ["amount", "hour_sin", "hour_cos"])
    first_model = predictor.model
    predictor.predict(amount=500, timestamp=datetime(2018, 4, 1))
    assert predictor.model is first_model


def test_corrupt_model_and_manifest_fail_closed(tmp_path, release_factory):
    release = release_factory()
    directory = tmp_path / "releases" / release["release_id"]
    with (directory / "model.joblib").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        load_release(tmp_path, release["release_id"])
    manifest = read_json(directory / "manifest.json")
    manifest["threshold"] = 0.8
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity"):
        load_release(tmp_path, release["release_id"])


def test_explicit_promotion_stale_recommendation_and_rollback(
    tmp_path, release_factory
):
    first = release_factory()
    with pytest.raises(ValueError, match="initial"):
        promote(tmp_path, first["release_id"], initial=False, reason="test")
    promote(tmp_path, first["release_id"], initial=True, reason="test")
    second = release_factory(baseline=first["release_id"])
    stale = release_factory(baseline=first["release_id"])
    promote(tmp_path, second["release_id"], initial=False, reason="test")
    with pytest.raises(ValueError, match="Stale"):
        promote(tmp_path, stale["release_id"], initial=False, reason="test")
    rollback(tmp_path, reason="test")
    assert deployment(tmp_path)["current"] == first["release_id"]
    assert len(deployment(tmp_path)["history"]) == 3


def test_rollout_is_confirmed_or_restores_the_last_confirmed_release(
    tmp_path, release_factory
):
    first = release_factory()
    promote(tmp_path, first["release_id"], initial=True, reason="test")
    served = {"release": None}
    ticks = iter(range(1000))
    rollout = {
        "probe": lambda: served["release"],
        "timeout": 3,
        "interval": 1,
        "clock": lambda: next(ticks),
        "sleep": lambda seconds: None,
    }
    # The API is still starting, then serves the selected release.
    responses = iter([None, None, first["release_id"]])
    rollout["probe"] = lambda: next(responses)
    assert confirm_rollout(tmp_path, **rollout) == {
        "status": "deployed",
        "release_id": first["release_id"],
    }
    assert deployment(tmp_path)["deployed"] == first["release_id"]

    second = release_factory(baseline=first["release_id"])
    promote(tmp_path, second["release_id"], initial=False, reason="test")
    # The API never serves the new release, e.g. it fails to load it.
    served["release"] = None
    rollout["probe"] = lambda: served["release"]
    failed = confirm_rollout(tmp_path, **rollout)
    assert failed["status"] == "rollout_failed"
    state = deployment(tmp_path)
    assert state["current"] == state["deployed"] == first["release_id"]
    assert state["history"][-1]["action"] == "rollout_failed"
    # A release that failed to roll out is never a rollback target.
    with pytest.raises(ValueError, match="No previous deployment"):
        rollback(tmp_path, reason="test")


def test_failed_rollout_of_an_unconfirmed_history_restores_the_previous_release(
    tmp_path, release_factory
):
    first = release_factory()
    promote(tmp_path, first["release_id"], initial=True, reason="test")
    # A deployment record written before confirmation existed has no "deployed" key.
    state = read_json(tmp_path / "deployment.json")
    state.pop("deployed", None)
    write_json(tmp_path / "deployment.json", state)
    second = release_factory(baseline=first["release_id"])
    promote(tmp_path, second["release_id"], initial=False, reason="test")
    ticks = iter(range(100))
    result = confirm_rollout(
        tmp_path,
        lambda: first["release_id"],
        timeout=2,
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    assert result["restored"] == first["release_id"]
    assert deployment(tmp_path)["current"] == first["release_id"]


def test_failed_initial_rollout_leaves_nothing_selected(tmp_path, release_factory):
    release = release_factory()
    promote(tmp_path, release["release_id"], initial=True, reason="test")
    ticks = iter(range(100))
    result = confirm_rollout(
        tmp_path,
        lambda: None,
        timeout=2,
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    assert result["restored"] is None
    assert deployment(tmp_path)["current"] is None
    with pytest.raises(ValueError, match="No release is selected"):
        confirm_rollout(tmp_path, lambda: None, timeout=0)


def test_served_release_reads_readiness_and_tolerates_unready_api():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from fraud_risk_platform.cli import served_release

    replies = iter([(503, {"detail": "not ready"}), (200, {"release_id": "abc"})])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = next(replies)
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert served_release(url) is None
        assert served_release(url) == "abc"
    finally:
        server.shutdown()
    assert served_release(url) is None


def test_champion_alias_follows_confirmed_release(tmp_path):
    import mlflow

    from fraud_risk_platform.training.tracking import set_champion

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    client = mlflow.MlflowClient(tracking_uri=uri)
    client.create_registered_model("fraud-risk")
    for _ in range(2):
        client.create_model_version("fraud-risk", source=str(tmp_path))
    for version in ("2", "1"):
        set_champion({"name": "fraud-risk", "version": version}, tracking_uri=uri)
        champion = client.get_model_version_by_alias("fraud-risk", "champion")
        assert str(champion.version) == version
        assert "confirmed_serving_at" in champion.tags


def test_rejects_unqualified_or_incomplete_release(tmp_path, release_factory):
    release = release_factory(eligible=False)
    with pytest.raises(ValueError, match="keeping"):
        promote(tmp_path, release["release_id"], initial=True, reason="test")
    release = release_factory()
    (tmp_path / "cycles" / release["source_run_id"] / "complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        promote(tmp_path, release["release_id"], initial=True, reason="test")
    assert deployment(tmp_path)["current"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_version", "future-v2"),
        ("features", ["amount"]),
        ("runtime_versions", {}),
    ],
)
def test_incompatible_release_rejected(tmp_path, release_factory, field, value):
    from fraud_risk_platform.persistence.files import identity

    release = release_factory()
    directory = tmp_path / "releases" / release["release_id"]
    manifest = read_json(directory / "manifest.json")
    manifest[field] = value
    manifest["release_id"] = identity(
        {k: v for k, v in manifest.items() if k != "release_id"}
    )
    write_json(directory / "manifest.json", manifest)
    directory.rename(directory.parent / manifest["release_id"])
    with pytest.raises(ValueError, match="Incompatible|runtime versions"):
        load_release(tmp_path, manifest["release_id"])


def test_decision_includes_exact_saved_threshold(tmp_path, release_factory):
    release = release_factory()
    predictor = FraudPredictor(root=tmp_path, release_id=release["release_id"])
    score = predictor.predict(amount=20, timestamp=datetime(2018, 4, 1))["fraud_score"]
    boundary_release = release_factory(threshold=score)
    boundary = FraudPredictor(root=tmp_path, release_id=boundary_release["release_id"])
    assert (
        boundary.predict(amount=20, timestamp=datetime(2018, 4, 1))[
            "flagged_for_review"
        ]
        is True
    )
