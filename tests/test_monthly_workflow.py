from datetime import datetime

import pytest

from fraud_risk_platform.cli import parser
from fraud_risk_platform.dataops.dataset import DataSettings
from fraud_risk_platform.dataops.ingestion import source_import
from fraud_risk_platform.dataops.reader import SnapshotReader, list_snapshots
from fraud_risk_platform.dataops.workflow import monthly_snapshot, next_month
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.training.inputs import training_data


@pytest.fixture
def monthly_job(object_store, ibm_history, dataset_profile, databases, tmp_path):
    settings = DataSettings(
        initial_snapshot="2019-02-10", label_delay_days=2, dataset=dataset_profile
    )

    def run(initial=False, **overrides):
        return monthly_snapshot(
            initial=initial,
            settings=settings,
            store=object_store,
            source_url=databases["source"],
            platform_url=databases["platform"],
            work_dir=tmp_path,
            **overrides,
        )

    return run


def test_commands_need_no_dataset_selection():
    for name in ("seed-source", "initial-snapshot", "next-snapshot"):
        args = parser().parse_args([name])
        assert args.command == name
        assert not hasattr(args, "import_id")
        assert not hasattr(args, "profile")
        assert not hasattr(args, "as_of")


@pytest.mark.parametrize(
    "start,end",
    [
        ("2019-01-31", "2019-02-28"),
        ("2020-01-31", "2020-02-29"),
        ("2019-12-08", "2020-01-08"),
    ],
)
def test_calendar_months(start, end):
    assert next_month(datetime.fromisoformat(start)) == datetime.fromisoformat(end)


@pytest.mark.integration
def test_initial_next_and_exhaustion(monthly_job, object_store, database, policy):
    with pytest.raises(ValueError, match="initial snapshot first"):
        monthly_job()
    first = monthly_job(initial=True)
    assert first["status"] == "created"
    assert monthly_job(initial=True)["snapshot_id"] == first["snapshot_id"]
    second = monthly_job()
    assert second["as_of"] == "2019-03-10T00:00:00"
    assert second["splits"]["context"]["rows"] > first["splits"]["context"]["rows"]
    # A valid data snapshot is allowed even when it cannot be used for fitting.
    with SnapshotReader(object_store, second["snapshot_id"]) as reader:
        reader.verify()
    with pytest.raises(ValueError, match="reserved final holdout"):
        training_data(second["snapshot_id"], policy)
    final = monthly_job()
    assert final["as_of"] == "2019-03-24T00:00:00"
    assert final["splits"]["context"] == {
        "rows": 640,
        "frauds": 160,
        "unavailable_labels": 0,
    }
    assert monthly_job()["status"] == "no_new_data"
    assert monthly_job()["status"] == "no_new_data"
    assert len(list(list_snapshots(object_store))) == 3
    assert monthly_job(initial=True)["snapshot_id"] == first["snapshot_id"]
    with connect(database) as conn:
        assert (
            conn.execute(
                "SELECT last_snapshot_id FROM monthly_snapshot_progress"
            ).fetchone()["last_snapshot_id"]
            == final["snapshot_id"]
        )


@pytest.mark.integration
def test_failed_publication_retries_the_same_month(
    monthly_job, object_store, database, monkeypatch
):
    first = monthly_job(initial=True)
    original = object_store.put_json

    def fail(key, value):
        if key.startswith("snapshots/"):
            raise RuntimeError("publication interrupted")
        return original(key, value)

    monkeypatch.setattr(object_store, "put_json", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        monthly_job()
    with connect(database) as conn:
        assert (
            conn.execute(
                "SELECT last_snapshot_id FROM monthly_snapshot_progress"
            ).fetchone()["last_snapshot_id"]
            == first["snapshot_id"]
        )
    monkeypatch.setattr(object_store, "put_json", original)
    assert monthly_job()["as_of"] == "2019-03-10T00:00:00"
    with connect(database) as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM ingestion_batches").fetchone()["n"]
            == 2
        )


@pytest.mark.integration
def test_rejected_month_does_not_advance(monthly_job, database):
    first = monthly_job(initial=True)
    with pytest.raises(ValueError, match="Delivery rejected"):
        monthly_job(fault="duplicate")
    with connect(database) as conn:
        assert (
            conn.execute(
                "SELECT last_snapshot_id FROM monthly_snapshot_progress"
            ).fetchone()["last_snapshot_id"]
            == first["snapshot_id"]
        )
    assert monthly_job()["as_of"] == "2019-03-10T00:00:00"


@pytest.mark.integration
def test_initial_rebuilds_after_ephemeral_storage_loss(monthly_job, object_store):
    first = monthly_job(initial=True)
    for page in object_store.client.get_paginator("list_objects_v2").paginate(
        Bucket=object_store.bucket
    ):
        keys = [{"Key": row["Key"]} for row in page.get("Contents", [])]
        if keys:
            object_store.client.delete_objects(
                Bucket=object_store.bucket, Delete={"Objects": keys}
            )
    with pytest.raises(ValueError, match="storage was reset"):
        monthly_job()
    rebuilt = monthly_job(initial=True)
    assert rebuilt["snapshot_id"] != first["snapshot_id"]
    assert rebuilt["splits"] == first["splits"]
    assert monthly_job()["status"] == "created"


@pytest.mark.integration
def test_full_import_selected_automatically(source_database, ibm_history):
    with connect(source_database) as conn:
        conn.execute(
            """INSERT INTO ibm_source_imports
            (import_id,source_manifest,adapter_version,customer_limit,label_delay_days,status,completed_at)
            SELECT %s,source_manifest,adapter_version,20,label_delay_days,status,now()
            FROM ibm_source_imports WHERE import_id=%s""",
            ("d" * 64, ibm_history),
        )
    assert source_import(source_database)["import_id"] == ibm_history


@pytest.mark.integration
def test_no_frauds_is_valid_snapshot_but_not_training(
    snapshot_job, source_database, object_store, policy
):
    with connect(source_database) as conn:
        conn.execute("UPDATE ibm_transactions SET is_fraud=false")
    manifest = snapshot_job()
    with SnapshotReader(object_store, manifest["snapshot_id"]) as reader:
        reader.verify()
    with pytest.raises(ValueError, match="insufficient observations or classes"):
        training_data(manifest["snapshot_id"], policy)
