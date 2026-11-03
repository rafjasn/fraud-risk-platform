"""Training inputs: immutable snapshots from S3 or a prepared local directory.

Every snapshot is checked for training eligibility and integrity before features are
read, and evaluation access stays separated from model search.
"""

import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path
from time import perf_counter

from fraud_risk_platform.dataops.reader import SnapshotReader
from fraud_risk_platform.features.matrices import feature_matrices
from fraud_risk_platform.persistence.files import read_json, sha256, write_json
from fraud_risk_platform.persistence.objects import ObjectStore

logger = logging.getLogger(__name__)


class LocalSnapshotStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def path(self, key):
        path = (self.directory / key).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Unsafe local snapshot path")
        return path

    def read_json(self, key):
        return read_json(self.path(key))

    def verify(self, descriptor, destination=None):
        path = self.path(descriptor["key"])
        if (
            path.stat().st_size != descriptor["bytes"]
            or sha256(path) != descriptor["sha256"]
        ):
            raise ValueError("Local snapshot checksum/size mismatch")
        if destination is not None:
            shutil.copyfile(path, destination)


@dataclass(frozen=True)
class LocalSnapshot:
    """A snapshot's files at their original keys in a local directory, such as a
    SageMaker channel, named explicitly rather than by a root `manifest.json`."""

    directory: Path
    snapshot_id: str

    def __str__(self):
        return self.snapshot_id


def snapshot_files(manifest) -> list[dict]:
    """Descriptors of every object a snapshot's integrity check reads."""
    return [f for files in manifest["files"].values() for f in files] + list(
        manifest["reports"].values()
    )


@contextmanager
def open_snapshot(reference):
    directory = Path(str(reference))
    if isinstance(reference, LocalSnapshot):
        store = LocalSnapshotStore(reference.directory)
        reference = reference.snapshot_id
    elif directory.is_dir():
        manifest = read_json(directory / "manifest.json")
        store = LocalSnapshotStore(directory)
        reference = manifest["snapshot_id"]
    else:
        store = ObjectStore()
    with SnapshotReader(store, reference) as reader:
        yield reader


def prepare_snapshot(reference, output):
    """Export a verified channel without querying upstream or changing its identity."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with open_snapshot(reference) as reader:
        manifest = reader.verify_integrity()
        for descriptor in snapshot_files(manifest):
            path = output / descriptor["key"]
            path.parent.mkdir(parents=True, exist_ok=True)
            reader.store.verify(descriptor, path)
        destination = output / "snapshots" / manifest["snapshot_id"] / "manifest.json"
        destination.parent.mkdir(parents=True)
        write_json(destination, manifest)
        write_json(output / "manifest.json", manifest)
    return {"snapshot_id": manifest["snapshot_id"], "directory": str(output)}


def reaches_holdout(manifest, profile) -> bool:
    """Whether a snapshot's evaluation window runs into the reserved final holdout."""
    return (
        manifest["windows"]["evaluation"][1]
        > profile.final_holdout_start.isoformat() + "T00:00:00"
    )


def check_eligibility(manifest, profile, policy):
    if reaches_holdout(manifest, profile):
        raise ValueError(
            "This snapshot reaches the reserved final holdout; use an earlier snapshot for training"
        )
    for name in ("train", "validation", "evaluation"):
        counts = manifest["splits"][name]
        if (
            counts["rows"] < policy.minimum_rows_per_split
            or counts["frauds"] < policy.minimum_frauds_per_split
            or counts["frauds"] == counts["rows"]
        ):
            raise ValueError(
                f"Snapshot is valid data but has insufficient observations or classes for training in {name}"
            )


@contextmanager
def verified_data(snapshot, policy):
    with open_snapshot(snapshot) as reader:
        check_eligibility(reader.manifest, reader.profile, policy)
        logger.info("Checking snapshot %s integrity", reader.manifest["snapshot_id"])
        started = perf_counter()
        reader.verify_integrity()
        logger.info("Snapshot integrity checked in %.1fs", perf_counter() - started)
        yield reader


def training_data(snapshot, policy):
    with verified_data(snapshot, policy) as reader:
        X, y = feature_matrices(reader)
        return reader.manifest, X, y


def check_refit_snapshot(search_reader, refit_reader):
    left, right = search_reader.manifest, refit_reader.manifest
    for key in (
        "dataset",
        "record_version",
        "source",
        "as_of",
        "windows",
        "label_delay_days",
        "context_start",
    ):
        if left[key] != right[key]:
            raise ValueError(
                f"Refit snapshot must preserve the search snapshot's {key}"
            )
    if right["dataset_profile"]["max_train_rows"] is not None:
        raise ValueError("Refit snapshot must be uncapped")
    if right["splits"]["train"]["rows"] != left["selection"]["eligible_rows"]:
        raise ValueError("Refit snapshot must contain all eligible training rows")
    # Equal context proves sampled-row membership too. Compare records because
    # valid Parquet exports may have different chunk boundaries.
    for split in ("context", "validation", "evaluation"):
        a = chain.from_iterable(search_reader.records(split))
        b = chain.from_iterable(refit_reader.records(split))
        if any(x != y for x, y in zip_longest(a, b)):
            raise ValueError(f"Refit snapshot changes {split} records")
