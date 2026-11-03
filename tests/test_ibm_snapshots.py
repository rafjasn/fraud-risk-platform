from datetime import datetime
from decimal import Decimal

import pytest
from botocore.exceptions import ClientError

from fraud_risk_platform.cli import parser
from fraud_risk_platform.dataops.reader import (
    SnapshotReader,
    list_snapshots,
    resolve_snapshot,
)
from fraud_risk_platform.features.matrices import feature_matrices
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.persistence.files import identity
from fraud_risk_platform.training.cycle import run_cycle
from fraud_risk_platform.training.inputs import training_data


def test_dataset_windows_and_holdout(dataset_profile):
    first = dataset_profile.windows("2019-02-10", 2)
    second = dataset_profile.windows("2019-02-20", 2)
    assert first["train"][0] == second["train"][0] == "2019-01-02T00:00:00"
    assert first["train"][1] < second["train"][1]
    assert first["evaluation"][1] == "2019-02-08T00:00:00"
    assert (
        dataset_profile.windows("2019-03-04", 2)["evaluation"][1]
        == "2019-03-02T00:00:00"
    )
    with pytest.raises(ValueError, match="no training"):
        dataset_profile.windows("2019-01-03", 2)


def test_training_selector_is_explicit():
    command = parser()
    assert command.parse_args(["train", "--snapshot", "a" * 64]).snapshot == "a" * 64
    assert (
        command.parse_args(["train", "--snapshot-as-of", "2019-07-08"]).snapshot_as_of
        == "2019-07-08"
    )
    with pytest.raises(SystemExit):
        command.parse_args(["train"])
    with pytest.raises(SystemExit):
        command.parse_args(
            ["train", "--snapshot", "a" * 64, "--snapshot-as-of", "2019-07-08"]
        )


@pytest.mark.integration
def test_snapshot_roundtrip_profiles_and_retry(
    snapshot_job, object_store, dataset_profile, database
):
    small = snapshot_job()
    repeated = snapshot_job(chunk_size=7)
    full = snapshot_job(
        profile=dataset_profile.model_copy(update={"max_train_rows": None})
    )
    assert repeated["snapshot_id"] == small["snapshot_id"]
    assert small["snapshot_id"] != full["snapshot_id"]
    assert small["splits"]["train"]["rows"] == 64
    assert full["splits"]["train"]["rows"] > 64
    assert small["files"]["validation"] == full["files"]["validation"]
    assert small["files"]["evaluation"] == full["files"]["evaluation"]
    assert small["files"]["context"] == full["files"]["context"]
    with SnapshotReader(object_store, small["snapshot_id"]) as reader:
        reader.verify()
        context = [row for batch in reader.records("context") for row in batch]
        assert len(context) == 320
        assert sum(row.is_fraud is None for row in context) == 16
        assert any(row.amount < 0 for row in context)
        assert all(row.tx_datetime < datetime(2019, 2, 10) for row in context)
        assert (
            max(len(batch) for batch in reader.iter_batches("train", batch_size=9)) <= 9
        )
        X, y = feature_matrices(reader)
        assert len(X["train"]) == len(y["train"]) == 64
    with connect(database) as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM ingestion_attempts").fetchone()["n"]
            == 1
        )
    with pytest.raises(ValueError, match="matched 2"):
        resolve_snapshot(object_store, as_of="2019-02-10")
    assert (
        resolve_snapshot(object_store, as_of="2019-02-10", profile=dataset_profile)
        == small["snapshot_id"]
    )
    with pytest.raises(ValueError, match="matched 0"):
        resolve_snapshot(object_store, as_of="2018-01-01")


@pytest.mark.integration
def test_monthly_batch_updates_labels_and_accumulates_history(
    snapshot_job, object_store, database
):
    first = snapshot_job()
    second = snapshot_job(as_of="2019-02-20")
    assert second["splits"]["context"]["rows"] == 400
    assert second["batch_ids"][:1] == first["batch_ids"]
    with connect(database) as conn:
        batch = conn.execute(
            "SELECT manifest FROM ingestion_batches ORDER BY as_of DESC LIMIT 1"
        ).fetchone()["manifest"]
        assert (
            batch["received"] == 96
        )  # 80 new transactions and 16 newly available labels
        report = object_store.read_json(batch["reports"]["json"]["key"])
        assert report["new_transactions"] == 80
        assert report["label_updates"] == 16
        assert (
            report["extracted_records"]
            == report["received"]
            == report["accepted_records"]
            == 96
        )
        assert report["new_transaction_summary"]["rows"] == 80
    report = object_store.read_json(second["reports"]["json"]["key"])
    comparison = report["comparison"]
    assert comparison["current"]["rows"] == comparison["reference"]["rows"] == 80
    assert (
        comparison["current"]["known_labels"]
        == comparison["reference"]["known_labels"]
        == 64
    )
    assert comparison["comparable"]
    assert report["delivery"]["label_updates"] == 16
    assert report["sampling"]["selected"]["rows"] == second["splits"]["train"]["rows"]
    assert (
        report["sampling"]["eligible"]["rows"] == second["selection"]["eligible_rows"]
    )
    with SnapshotReader(object_store, first["snapshot_id"]) as reader:
        reader.verify()
        assert (
            sum(
                row.is_fraud is None
                for batch in reader.records("context")
                for row in batch
            )
            == 16
        )
    assert snapshot_job()["snapshot_id"] == first["snapshot_id"]
    with pytest.raises(ValueError, match="new --simulation"):
        snapshot_job(as_of="2019-02-15")
    backfill = snapshot_job(as_of="2019-02-15", simulation="backfill")
    assert backfill["as_of"] == "2019-02-15T00:00:00"


@pytest.mark.integration
@pytest.mark.parametrize(
    "fault", ["missing-amount", "invalid-amount", "duplicate", "conflict"]
)
def test_rejection_preserves_evidence_and_allows_corrected_retry(
    snapshot_job, object_store, database, source_database, fault
):
    with pytest.raises(ValueError, match="Delivery rejected"):
        snapshot_job(fault=fault)
    assert list(list_snapshots(object_store)) == []
    with connect(database) as conn:
        assert (
            conn.execute("SELECT accepted_cutoff FROM ingestion_streams").fetchone()[
                "accepted_cutoff"
            ]
            is None
        )
        attempt = conn.execute(
            "SELECT status,report_key FROM ingestion_attempts"
        ).fetchone()
        assert attempt["status"] == "rejected"
        report = object_store.read_json(attempt["report_key"])
        assert report["invalid"] > 0
        assert report["accepted_records"] == 0
        assert report["valid"] + report["invalid"] == report["received"]
    with connect(source_database) as conn:
        assert conn.execute(
            "SELECT amount FROM ibm_transactions WHERE source_row=1"
        ).fetchone()["amount"] == Decimal("-10.50")
    snapshot_job()
    with connect(database) as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM ingestion_batches").fetchone()["n"]
            == 1
        )


@pytest.mark.integration
def test_failed_snapshot_publication_reuses_accepted_delivery(
    snapshot_job, object_store, monkeypatch, database
):
    original = object_store.put_json

    def fail(key, value):
        if key.startswith("snapshots/"):
            raise RuntimeError("upload interrupted")
        return original(key, value)

    monkeypatch.setattr(object_store, "put_json", fail)
    with pytest.raises(RuntimeError, match="upload interrupted"):
        snapshot_job()
    assert list(list_snapshots(object_store)) == []
    monkeypatch.setattr(object_store, "put_json", original)
    completed = snapshot_job()
    with SnapshotReader(object_store, completed["snapshot_id"]) as reader:
        reader.verify()
    with connect(database) as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM ingestion_attempts").fetchone()["n"]
            == 1
        )


@pytest.mark.integration
def test_failed_raw_upload_does_not_advance_cursor(
    snapshot_job, object_store, monkeypatch, database
):
    original = object_store.put_file

    def fail(key, path, content_type="application/octet-stream"):
        if key.endswith("part-000001.jsonl"):
            raise RuntimeError("raw upload interrupted")
        return original(key, path, content_type)

    monkeypatch.setattr(object_store, "put_file", fail)
    with pytest.raises(RuntimeError, match="raw upload interrupted"):
        snapshot_job()
    with connect(database) as conn:
        assert (
            conn.execute("SELECT accepted_cutoff FROM ingestion_streams").fetchone()[
                "accepted_cutoff"
            ]
            is None
        )
        assert (
            conn.execute("SELECT status FROM ingestion_attempts").fetchone()["status"]
            == "failed"
        )
    monkeypatch.setattr(object_store, "put_file", original)
    snapshot_job()


@pytest.mark.integration
def test_rejection_report_failure_records_failed_attempt(
    snapshot_job, object_store, monkeypatch, database
):
    original = object_store.put_file

    def fail(key, path, content_type="application/octet-stream"):
        if key.startswith("reports/ingestion/") and key.endswith("report.html"):
            raise RuntimeError("report upload interrupted")
        return original(key, path, content_type)

    monkeypatch.setattr(object_store, "put_file", fail)
    with pytest.raises(RuntimeError, match="report upload interrupted"):
        snapshot_job(fault="missing-amount")
    with connect(database) as conn:
        attempt = conn.execute("SELECT status,error FROM ingestion_attempts").fetchone()
        assert attempt["status"] == "failed"
        assert "report upload interrupted" in attempt["error"]
        assert (
            conn.execute("SELECT accepted_cutoff FROM ingestion_streams").fetchone()[
                "accepted_cutoff"
            ]
            is None
        )
    assert list(list_snapshots(object_store)) == []
    monkeypatch.setattr(object_store, "put_file", original)
    snapshot_job()


@pytest.mark.integration
def test_missing_accepted_ledger_entry_cannot_skip_history(snapshot_job, database):
    snapshot_job()
    with connect(database) as conn:
        conn.execute("DELETE FROM ingestion_batches")
    with pytest.raises(ValueError, match="ledger and cursor do not agree"):
        snapshot_job(as_of="2019-02-20")


@pytest.mark.integration
def test_reader_rejects_missing_and_changed_data(snapshot_job, object_store):
    manifest = snapshot_job()
    file = manifest["files"]["train"][0]
    object_store.client.put_object(
        Bucket=object_store.bucket, Key=object_store.key(file["key"]), Body=b"changed"
    )
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        with pytest.raises(ValueError, match="checksum"):
            reader.verify()
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        with pytest.raises(ValueError, match="checksum"):
            reader.verify_integrity()
    object_store.client.delete_object(
        Bucket=object_store.bucket, Key=object_store.key(file["key"])
    )
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        with pytest.raises(ClientError):
            reader.verify()


@pytest.mark.integration
def test_reader_checks_actual_split_membership(snapshot_job, object_store):
    manifest = snapshot_job()
    manifest["files"]["train"] = manifest["files"]["validation"]
    manifest["splits"]["train"] = manifest["splits"]["validation"]
    manifest["selection"]["selected_rows"] = manifest["splits"]["train"]["rows"]
    manifest.pop("snapshot_id")
    manifest["snapshot_id"] = identity(manifest)
    object_store.put_json(
        f"snapshots/{manifest['snapshot_id']}/manifest.json", manifest
    )
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        with pytest.raises(ValueError, match="declared context/window/selection"):
            reader.verify()
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        with pytest.raises(ValueError, match="train has rows outside its declared"):
            reader.verify_integrity()


@pytest.mark.integration
def test_integrity_check_compares_label_counts(snapshot_job, object_store):
    manifest = snapshot_job()
    manifest["splits"]["validation"]["frauds"] += 1
    manifest.pop("snapshot_id")
    manifest["snapshot_id"] = identity(manifest)
    object_store.put_json(
        f"snapshots/{manifest['snapshot_id']}/manifest.json", manifest
    )
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        with pytest.raises(ValueError, match="validation label counts"):
            reader.verify_integrity()


@pytest.mark.integration
def test_training_skips_full_snapshot_replay(
    snapshot_job, object_store, policy, monkeypatch
):
    manifest = snapshot_job()

    def replay(self):
        raise AssertionError("training must not replay the published snapshot")

    monkeypatch.setattr(SnapshotReader, "verify", replay)
    _, X, y = training_data(manifest["snapshot_id"], policy)
    assert len(y["train"]) == manifest["splits"]["train"]["rows"]


@pytest.mark.integration
def test_s3_conditional_publication(object_store):
    object_store.put_json("test.json", {"value": 1})
    object_store.put_json("test.json", {"value": 1})
    with pytest.raises(ValueError, match="checksum"):
        object_store.put_json("test.json", {"value": 2})
    assert object_store.read_json("test.json") == {"value": 1}


@pytest.mark.integration
def test_snapshot_trains_without_source_database(
    snapshot_job, object_store, policy, tmp_path, monkeypatch
):
    manifest = snapshot_job()
    monkeypatch.setenv(
        "SOURCE_DATABASE_URL", "postgresql://invalid:invalid@127.0.0.1:1/invalid"
    )
    report = run_cycle(
        root=tmp_path / "model-state",
        snapshot=manifest["snapshot_id"],
        policy=policy,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    assert report["snapshot_id"] == manifest["snapshot_id"]
    assert report["dataset"] == "ibm-tabformer"
    assert report["deployment_supported"] is True
    assert report["candidates"]["logistic_basic"]["validation"] is not None
    assert report["recommendation"] == "initial_promotion"
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        html = tmp_path / "data-report.html"
        object_store.verify(reader.manifest["reports"]["html"], html)
        assert "data:image/png;base64" in html.read_text()
    assert not (tmp_path / "model-state" / "deployment.json").exists()


@pytest.mark.integration
def test_refresh_old_report_preserves_snapshot_artifacts(
    snapshot_job, object_store, monkeypatch
):
    from fraud_risk_platform.dataops.reports.data import (
        REPORT_VERSION,
        refresh_snapshot_report,
    )

    manifest = snapshot_job()
    original_id = manifest["snapshot_id"]
    rich_report = object_store.read_json(manifest["reports"]["json"]["key"])
    legacy_report = {
        k: rich_report[k]
        for k in (
            "kind",
            "status",
            "as_of",
            "population",
            "dataset_profile",
            "windows",
            "splits",
            "eligible_training_rows",
            "training_ready",
            "summary",
        )
    }
    # Model an older immutable snapshot, without replacing the original object.
    legacy_descriptor = object_store.put_json("legacy/report.json", legacy_report)
    manifest["reports"]["json"] = legacy_descriptor
    manifest.pop("snapshot_id")
    manifest["snapshot_id"] = identity(manifest)
    key = f"snapshots/{manifest['snapshot_id']}/manifest.json"
    object_store.put_json(key, manifest)
    monkeypatch.setenv(
        "SOURCE_DATABASE_URL", "postgresql://invalid:invalid@127.0.0.1:1/invalid"
    )
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        refreshed = refresh_snapshot_report(reader)
    assert refreshed["report_version"] == REPORT_VERSION == 5
    classes = refreshed["fraud_exploration"]["classes"]
    assert (
        sum(c["rows"] for c in classes.values()) == manifest["splits"]["train"]["rows"]
    )
    assert classes["fraud"]["rows"] == manifest["splits"]["train"]["frauds"]
    assert refreshed["fraud_exploration"] == rich_report["fraud_exploration"]
    assert (
        refreshed["sampling"]["eligible"]["rows"]
        == rich_report["eligible_training_rows"]
    )
    assert (
        refreshed["sampling"]["selected"]["rows"] == manifest["splits"]["train"]["rows"]
    )
    assert refreshed["delivery"] is None
    assert object_store.read_json(key) == manifest
    assert object_store.read_json("legacy/report.json") == legacy_report
    assert object_store.exists(f"snapshots/{original_id}/manifest.json")
