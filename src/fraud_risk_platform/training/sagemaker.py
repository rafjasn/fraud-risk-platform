"""What a SageMaker training job reads from S3, and the release it produces.

`stage_inputs` writes a job's inputs to the data bucket and returns them as
SageMaker `InputDataConfig` channels. The `objects/` prefix holds every snapshot's
content-addressed files, so each snapshot channel is a manifest file that lists
exactly that snapshot's keys; SageMaker File mode downloads them under the channel at
the same relative keys, where the entrypoint reads the snapshot named in `job.json`.
The job ID is the training-job ledger's request identity, so a workflow can use it
as its execution name to start each job once.

`import_release` takes the job's `model.tar.gz`, which holds the cycle report under
`cycle/` and, when the cycle selected one, a release bundle. It adds the release after
the same verification as any release, and the report where `promote` and `make month`
look for completed cycles, so a cycle without a release still counts as trained. Release
bundles contain joblib files, which execute code when loaded, so import archives only
from the project's own training output location.

Both run locally against LocalStack; neither has run against SageMaker itself.
"""

import logging
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from fraud_risk_platform.contracts.training import TrainingPolicy
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.files import identity, read_json, safe_id, sha256
from fraud_risk_platform.persistence.objects import ObjectStore
from fraud_risk_platform.training.inputs import snapshot_files
from fraud_risk_platform.training.jobs import job_request

logger = logging.getLogger(__name__)


def _channel(name: str, data_type: str, uri: str) -> dict:
    return {
        "ChannelName": name,
        "DataSource": {
            "S3DataSource": {
                "S3DataType": data_type,
                "S3Uri": uri,
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "InputMode": "File",
    }


def _bucket_root(store: ObjectStore) -> str:
    return f"s3://{store.bucket}/" + (f"{store.prefix}/" if store.prefix else "")


def _snapshot_channel(store: ObjectStore, name: str, snapshot_id: str, key: str):
    """A manifest-file channel listing one snapshot's manifest and objects."""
    manifest_key = f"snapshots/{safe_id(snapshot_id)}/manifest.json"
    manifest = store.read_json(manifest_key)
    keys = [manifest_key, *(file["key"] for file in snapshot_files(manifest))]
    store.put_json(key, [{"prefix": _bucket_root(store)}, *keys])
    return _channel(name, "ManifestFile", store.uri(key))


def stage_inputs(
    root: Path,
    store: ObjectStore,
    *,
    snapshot_id: str,
    policy: TrainingPolicy,
    refit_snapshot_id: str | None = None,
    recipe_name: str | None = None,
    trials: int | None = None,
) -> dict:
    """Write a job's channels to S3; repeating an identical request rewrites nothing."""
    incumbent = deployment(root)["current"]
    request = job_request(
        snapshot_id,
        policy,
        incumbent_release_id=incumbent,
        recipe_name=recipe_name,
        trials=trials,
        refit_snapshot=refit_snapshot_id,
    )
    job_id = identity(request)
    prefix = f"training-inputs/{job_id}"
    channels = [
        _snapshot_channel(store, "snapshot", snapshot_id, f"{prefix}/snapshot.manifest")
    ]
    if refit_snapshot_id:
        channels.append(
            _snapshot_channel(
                store, "refit", refit_snapshot_id, f"{prefix}/refit.manifest"
            )
        )
    store.put_json(
        f"{prefix}/config/job.json",
        {
            "policy": policy.model_dump(),
            "snapshot_id": snapshot_id,
            "refit_snapshot_id": refit_snapshot_id,
            "recipe": recipe_name,
            "trials": trials,
        },
    )
    channels.append(_channel("config", "S3Prefix", store.uri(f"{prefix}/config/")))
    if incumbent:
        # Verified before upload, so the job compares against the deployed bundle.
        load_release(root, incumbent)
        directory = root / "releases" / incumbent
        for path in sorted(directory.iterdir()):
            store.put_file(f"{prefix}/incumbent/{path.name}", path)
        channels.append(
            _channel("incumbent", "S3Prefix", store.uri(f"{prefix}/incumbent/"))
        )
    return {
        "job_id": job_id,
        # SageMaker job names allow 63 characters.
        "training_job_name": f"fraud-train-{job_id[:40]}",
        "request": request,
        "input_data_config": channels,
    }


def _download(archive: str, directory: Path) -> Path:
    location = urlsplit(archive)
    if location.scheme != "s3":
        return Path(archive)
    path = directory / "model.tar.gz"
    store = ObjectStore(f"s3://{location.netloc}")
    store.client.download_file(location.netloc, location.path.lstrip("/"), str(path))
    return path


def import_release(root: Path, archive: str) -> dict:
    """Verify a training job's model.tar.gz and add its cycle and release, once."""
    releases = root / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".import-", dir=releases) as temporary:
        stage = Path(temporary)
        bundle = stage / "bundle"
        with tarfile.open(_download(archive, stage), "r:gz") as tar:
            # The data filter rejects absolute paths, links outside, and devices.
            tar.extractall(bundle, filter="data")
        released = not (bundle / "no-qualified-release.json").exists()
        release_id = (
            safe_id(read_json(bundle / "manifest.json")["release_id"])
            if released
            else None
        )
        evidence = _cycle_evidence(bundle, release_id)
        # Moved out first, so the release directory holds only the release.
        (bundle / "cycle").rename(stage / "cycle")
        if not released:
            _place_cycle(root, stage / "cycle", evidence)
            report = read_json(root / "cycles" / evidence["run_id"] / "report.json")
            logger.info("Imported cycle %s without a release", evidence["run_id"])
            return {
                "status": "no_release",
                "run_id": evidence["run_id"],
                "recommendation": report["recommendation"],
            }
        destination = releases / release_id
        if destination.exists():
            # The ID hashes a manifest that checksums every file, so this is the same release.
            load_release(root, release_id)
            _place_cycle(root, stage / "cycle", evidence)
            return {"status": "already_imported", "release_id": release_id}
        bundle.rename(destination)
        try:
            manifest, _ = load_release(root, release_id)
        except Exception:
            shutil.rmtree(destination)
            raise
        _place_cycle(root, stage / "cycle", evidence)
    logger.info("Imported release %s from %s", release_id, archive)
    return {
        "status": "imported",
        "release_id": release_id,
        "recommendation": manifest["recommendation"],
        "promotion_eligible": manifest["promotion_eligible"],
    }


def _cycle_evidence(bundle: Path, release_id: str | None) -> dict:
    """The packaged cycle report, checked to be complete and about this release."""
    cycle = bundle / "cycle"
    if not (cycle / "complete.json").exists():
        raise ValueError(
            "The archive lacks the training report that promotion requires"
        )
    completed = read_json(cycle / "complete.json")
    report = read_json(cycle / "report.json")
    if (
        completed["report_sha256"] != sha256(cycle / "report.json")
        or report["release_id"] != release_id
        or completed["run_id"] != report["run_id"]
    ):
        raise ValueError("The archive's training report does not match its release")
    return completed


def _place_cycle(root: Path, staged: Path, evidence: dict):
    destination = root / "cycles" / safe_id(evidence["run_id"])
    if destination.exists():
        if read_json(destination / "complete.json") != evidence:
            raise ValueError(
                f"A different cycle report already exists at {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(destination)
