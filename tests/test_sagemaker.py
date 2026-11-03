"""SageMaker inputs staged in S3, the entrypoint reading them, and the release imported."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from fraud_risk_platform.models.deployment import promote
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.training.sagemaker import import_release, stage_inputs


def archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def cycle_files(release_id, recommendation) -> dict[str, bytes]:
    report = json.dumps(
        {"run_id": "r1", "release_id": release_id, "recommendation": recommendation}
    ).encode()
    complete = {"run_id": "r1", "report_sha256": hashlib.sha256(report).hexdigest()}
    return {
        "cycle/report.json": report,
        "cycle/complete.json": json.dumps(complete).encode(),
    }


def test_a_cycle_without_a_release_imports_as_trained(tmp_path):
    marker = {"run_id": "r1", "recommendation": "no_initial_model"}
    files = {
        "no-qualified-release.json": json.dumps(marker).encode(),
        **cycle_files(None, "no_initial_model"),
    }
    result = import_release(tmp_path / "state", str(archive(tmp_path / "a.tgz", files)))
    assert result == {
        "status": "no_release",
        "run_id": "r1",
        "recommendation": "no_initial_model",
    }
    assert (tmp_path / "state" / "cycles" / "r1" / "complete.json").exists()


def test_import_refuses_unsafe_or_unsupported_archives(tmp_path):
    root = tmp_path / "state"
    escaping = archive(tmp_path / "escaping.tgz", {"../outside.json": b"{}"})
    with pytest.raises(tarfile.OutsideDestinationError):
        import_release(root, str(escaping))
    assert not (root / "outside.json").exists()
    marker = json.dumps({"run_id": "r1", "recommendation": "no_initial_model"})
    bare = archive(
        tmp_path / "bare.tgz", {"no-qualified-release.json": marker.encode()}
    )
    with pytest.raises(ValueError, match="lacks the training report"):
        import_release(root, str(bare))
    # A report about another release cannot travel with this one.
    files = {"manifest.json": json.dumps({"release_id": "a" * 64}).encode()}
    mismatched = archive(
        tmp_path / "mismatched.tgz", {**files, **cycle_files("b" * 64, "promote")}
    )
    with pytest.raises(ValueError, match="does not match its release"):
        import_release(root, str(mismatched))
    assert list((root / "releases").iterdir()) == []
    assert not (root / "cycles").exists()


def download_channels(client, channels, base: Path):
    """Lay out channels as SageMaker File mode does: each key at its relative path."""

    def fetch(uri, destination):
        location = urlsplit(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(
            location.netloc, location.path.lstrip("/"), str(destination)
        )

    for channel in channels:
        source = channel["DataSource"]["S3DataSource"]
        target = base / channel["ChannelName"]
        location = urlsplit(source["S3Uri"])
        if source["S3DataType"] == "ManifestFile":
            body = client.get_object(
                Bucket=location.netloc, Key=location.path.lstrip("/")
            )
            prefix, *keys = json.loads(body["Body"].read())
            for key in keys:
                fetch(prefix["prefix"] + key, target / key)
        else:
            prefix = location.path.lstrip("/")
            listing = client.list_objects_v2(Bucket=location.netloc, Prefix=prefix)
            for item in listing["Contents"]:
                fetch(
                    f"s3://{location.netloc}/{item['Key']}",
                    target / item["Key"][len(prefix) :],
                )


@pytest.mark.integration
def test_staged_job_trains_from_s3_channels_and_its_release_imports(
    tmp_path, snapshot_job, object_store, policy
):
    snapshot = snapshot_job()["snapshot_id"]
    root = tmp_path / "state"
    staged = stage_inputs(
        root,
        object_store,
        snapshot_id=snapshot,
        policy=policy,
        recipe_name="logistic_basic",
        trials=1,
    )
    # An identical request stages the same job without rewriting anything.
    assert (
        stage_inputs(
            root,
            object_store,
            snapshot_id=snapshot,
            policy=policy,
            recipe_name="logistic_basic",
            trials=1,
        )
        == staged
    )
    assert len(staged["training_job_name"]) <= 63
    assert [c["ChannelName"] for c in staged["input_data_config"]] == [
        "snapshot",
        "config",
    ]
    channels = tmp_path / "input"
    download_channels(object_store.client, staged["input_data_config"], channels)
    # The channel holds only this snapshot's keys; the job names the snapshot.
    assert not (channels / "snapshot" / "manifest.json").exists()
    assert (channels / "snapshot" / "snapshots" / snapshot / "manifest.json").exists()

    output, model = tmp_path / "output" / "data", tmp_path / "model"
    output.mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "-m", "fraud_risk_platform.training.entrypoint", "train"],
        env={
            **os.environ,
            "TRAINING_CHANNEL_ROOT": str(channels),
            "SM_OUTPUT_DIR": str(output),
            "SM_MODEL_DIR": str(model),
            "MLFLOW_TRACKING_URI": f"sqlite:///{output}/mlflow.db",
            "MLFLOW_ARTIFACT_URI": (output / "artifacts").as_uri(),
            # Training reads only its channels.
            "SOURCE_DATABASE_URL": "postgresql://unavailable.invalid/source",
            "AWS_ENDPOINT_URL_S3": "http://unavailable.invalid:9",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    # SageMaker packs the model directory's contents into model.tar.gz.
    packed = tmp_path / "model.tar.gz"
    with tarfile.open(packed, "w:gz") as tar:
        for path in sorted(model.iterdir()):
            tar.add(path, arcname=path.name)
    imported = import_release(root, str(packed))
    assert imported["status"] == "imported"
    manifest, _ = load_release(root, imported["release_id"])
    assert manifest["snapshot_id"] == snapshot
    assert (root / "cycles" / manifest["source_run_id"] / "complete.json").exists()
    assert not (root / "releases" / imported["release_id"] / "cycle").exists()
    assert import_release(root, str(packed)) == {
        "status": "already_imported",
        "release_id": imported["release_id"],
    }

    # Once deployed, the release becomes the next job's incumbent channel.
    # The packaged cycle report makes the imported release promotable.
    promote(root, imported["release_id"], initial=True, reason="test")
    next_job = stage_inputs(
        root,
        object_store,
        snapshot_id=snapshot,
        policy=policy,
        recipe_name="logistic_basic",
        trials=1,
    )
    assert next_job["job_id"] != staged["job_id"]
    assert next_job["request"]["incumbent_release_id"] == imported["release_id"]
    incumbent = tmp_path / "next-input"
    download_channels(object_store.client, next_job["input_data_config"], incumbent)
    assert (incumbent / "incumbent" / "manifest.json").exists()
