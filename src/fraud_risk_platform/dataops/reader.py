"""Resolve, verify, and batch-read completed IBM snapshots without source DB access."""

import re
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fraud_risk_platform.contracts.base import midnight
from fraud_risk_platform.dataops.dataset import (
    DatasetProfile,
    Observation,
    arrow_schema,
)
from fraud_risk_platform.dataops.history_store import HistoryStore
from fraud_risk_platform.persistence.files import identity, sha256


def validate_manifest(manifest):
    snapshot_id = manifest.get("snapshot_id", "")
    if (
        not re.fullmatch("[0-9a-f]{64}", snapshot_id)
        or identity({k: v for k, v in manifest.items() if k != "snapshot_id"})
        != snapshot_id
    ):
        raise ValueError("Snapshot manifest identity mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "ibm-tabformer"
        or manifest.get("record_version") != "ibm-observation-v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Unsupported or incomplete snapshot")
    profile = DatasetProfile.model_validate(manifest["dataset_profile"])
    if identity(profile.model_dump(mode="json")) != manifest["profile_sha256"]:
        raise ValueError("Dataset profile identity mismatch")
    if (
        manifest["windows"]
        != profile.windows(manifest["as_of"][:10], manifest["label_delay_days"])
        or manifest["context_start"] != profile.context_start.isoformat()
    ):
        raise ValueError("Snapshot windows do not match its dataset profile")
    if set(manifest["files"]) != {"context", "train", "validation", "evaluation"}:
        raise ValueError("Snapshot partition contract mismatch")
    if manifest["selection"]["method"] != "sha256-source-seed-row-v1":
        raise ValueError("Unsupported training selection method")
    for split, files in manifest["files"].items():
        for file in files:
            if (
                not re.fullmatch("objects/[0-9a-f]{64}\\.parquet", file["key"])
                or file["key"] != f"objects/{file['sha256']}.parquet"
                or file["bytes"] <= 0
                or file["rows"] <= 0
            ):
                raise ValueError("Invalid snapshot object descriptor")
        if sum(f["rows"] for f in files) != manifest["splits"][split]["rows"]:
            raise ValueError("Snapshot file counts disagree with split counts")
    return profile


class SnapshotReader:
    def __init__(self, store, reference):
        reference = str(reference)
        if reference.startswith("s3://"):
            match = re.search(r"/snapshots/([0-9a-f]{64})/manifest.json$", reference)
            if not match or reference != store.uri(
                f"snapshots/{match[1]}/manifest.json"
            ):
                raise ValueError(
                    "Snapshot URI must belong to the configured dataset store"
                )
            reference = match[1]
        if not re.fullmatch("[0-9a-f]{64}", reference):
            raise ValueError("Supply an immutable snapshot ID or its S3 manifest URI")
        self.store = store
        self.manifest = store.read_json(f"snapshots/{reference}/manifest.json")
        self.profile = validate_manifest(self.manifest)
        if reference != self.manifest["snapshot_id"]:
            raise ValueError("Snapshot key and manifest identity disagree")
        self.temporary = TemporaryDirectory(prefix="snapshot-reader-")
        self.directory = Path(self.temporary.name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.temporary.cleanup()

    def local_path(self, descriptor):
        """A local Parquet object after its bytes, schema, and row count are checked."""
        self._parquet(descriptor)
        return self.directory / f"{descriptor['sha256']}.parquet"

    def _parquet(self, descriptor):
        path = self.directory / f"{descriptor['sha256']}.parquet"
        if not path.exists():
            self.store.verify(descriptor, path)
        elif (
            path.stat().st_size != descriptor["bytes"]
            or sha256(path) != descriptor["sha256"]
        ):
            raise ValueError("Snapshot cache checksum mismatch")
        parquet = pq.ParquetFile(path)
        if (
            not parquet.schema_arrow.equals(arrow_schema(), check_metadata=False)
            or parquet.metadata.num_rows != descriptor["rows"]
        ):
            raise ValueError("Parquet schema/count does not match snapshot")
        return parquet

    def iter_batches(self, split, *, batch_size=10_000, columns=None):
        if split not in self.manifest["files"] or batch_size <= 0:
            raise ValueError("Unknown partition or invalid batch size")
        for descriptor in self.manifest["files"][split]:
            yield from self._parquet(descriptor).iter_batches(
                batch_size=batch_size, columns=columns
            )

    def records(self, split):
        for batch in self.iter_batches(split):
            yield [Observation.model_validate(row) for row in batch.to_pylist()]

    def verify_integrity(self):
        """Check bytes, schema, counts, split windows, and label visibility.

        Publication already replayed the context and derived every split; verify()
        repeats that replay. Training needs only proof that it reads those outputs.
        """
        cutoff = midnight(self.manifest["as_of"][:10])
        bounds = {
            "context": [self.manifest["context_start"], self.manifest["as_of"]],
            **self.manifest["windows"],
        }
        for split, (start, end) in bounds.items():
            frauds = pending = 0
            for descriptor in self.manifest["files"][split]:
                table = self._parquet(descriptor).read(
                    columns=["tx_datetime", "is_fraud", "label_available_at"]
                )
                times = pc.min_max(table["tx_datetime"]).as_py()
                if not (
                    datetime.fromisoformat(start)
                    <= times["min"]
                    <= times["max"]
                    < datetime.fromisoformat(end)
                ):
                    raise ValueError(f"{split} has rows outside its declared window")
                visible = pc.less_equal(
                    table["label_available_at"], pa.scalar(cutoff, pa.timestamp("us"))
                )
                if not pc.all(
                    pc.equal(visible, pc.is_valid(table["is_fraud"]))
                ).as_py():
                    raise ValueError(f"{split} labels disagree with the cutoff")
                frauds += pc.sum(table["is_fraud"]).as_py() or 0
                pending += table["is_fraud"].null_count
            stats = self.manifest["splits"][split]
            if (
                frauds != stats["frauds"]
                or pending != stats["unavailable_labels"]
                or (split != "context" and pending)
            ):
                raise ValueError(f"{split} label counts disagree with manifest")
        if (
            self.manifest["selection"]["selected_rows"]
            != self.manifest["splits"]["train"]["rows"]
        ):
            raise ValueError("Selected training row count mismatch")
        for descriptor in self.manifest["reports"].values():
            self.store.verify(descriptor)
        return self.manifest

    def verify(self):
        paths = {
            split: [self.local_path(descriptor) for descriptor in files]
            for split, files in self.manifest["files"].items()
        }
        database = self.directory / "verify.duckdb"
        context = HistoryStore(database)
        try:
            findings = context.apply_parquet(
                paths["context"],
                start=self.profile.context_start,
                cutoff=midnight(self.manifest["as_of"][:10]),
                label_delay_days=self.manifest["label_delay_days"],
            )
            stats = self.manifest["splits"]["context"]
            if (
                findings["invalid"]
                or findings["received"] != stats["rows"]
                or findings["summary"]["frauds"] != stats["frauds"]
                or findings["summary"]["unknown_labels"] != stats["unavailable_labels"]
            ):
                raise ValueError("Snapshot context failed semantic verification")
            context.select(
                self.profile,
                self.manifest["windows"],
                self.manifest["source"]["manifest"]["sha256"],
            )
            eligible = context.count_eligible(self.manifest["windows"]["train"])
            if eligible != self.manifest["selection"]["eligible_rows"]:
                raise ValueError("Eligible training row count mismatch")
            for split in ("train", "validation", "evaluation"):
                stored = context.compare(split, self.manifest["windows"], paths[split])
                if not stored["identical"]:
                    raise ValueError(
                        f"{split} differs from the declared context/window/selection"
                    )
                stats = self.manifest["splits"][split]
                if (
                    stored["rows"] != stats["rows"]
                    or stored["frauds"] != stats["frauds"]
                    or stats["unavailable_labels"] != 0
                ):
                    raise ValueError(f"{split} counts disagree with manifest")
            if (
                self.manifest["selection"]["selected_rows"]
                != self.manifest["splits"]["train"]["rows"]
            ):
                raise ValueError("Selected training row count mismatch")
            for descriptor in self.manifest["reports"].values():
                self.store.verify(descriptor)
        finally:
            context.close()
            for path in (database, database.parent / f"{database.name}.wal"):
                path.unlink(missing_ok=True)
        return self.manifest


def list_snapshots(store, *, as_of=None, profile=None, import_id=None):
    for key in store.list_manifests("snapshots/"):
        manifest = store.read_json(key)
        validate_manifest(manifest)
        if as_of and manifest["as_of"] != midnight(as_of).isoformat():
            continue
        if profile and manifest["profile_sha256"] != identity(
            profile.model_dump(mode="json")
        ):
            continue
        if import_id and manifest["source"]["import_id"] != import_id:
            continue
        yield {
            "snapshot_id": manifest["snapshot_id"],
            "manifest_uri": store.uri(key),
            "as_of": manifest["as_of"],
            "import_id": manifest["source"]["import_id"],
            "profile_sha256": manifest["profile_sha256"],
            "splits": manifest["splits"],
            "max_train_rows": manifest["dataset_profile"]["max_train_rows"],
            "report_uri": store.uri(manifest["reports"]["html"]["key"]),
        }


def resolve_snapshot(store, *, as_of, profile=None, import_id=None):
    matches = list(
        list_snapshots(store, as_of=as_of, profile=profile, import_id=import_id)
    )
    if len(matches) != 1:
        raise ValueError(
            f"Cutoff matched {len(matches)} snapshots; specify a snapshot ID or narrow the profile/import"
        )
    return matches[0]["snapshot_id"]
