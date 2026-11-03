"""Initial/next monthly snapshots with internal progress and source selection."""

from calendar import monthrange
from datetime import datetime
from uuid import uuid4

from fraud_risk_platform.contracts.base import midnight
from fraud_risk_platform.dataops.ingestion import create_snapshot, source_import
from fraud_risk_platform.dataops.reader import SnapshotReader
from fraud_risk_platform.persistence.database import connect, connect_source
from fraud_risk_platform.persistence.files import identity


def next_month(value):
    year, month = (
        (value.year + 1, 1) if value.month == 12 else (value.year, value.month + 1)
    )
    return value.replace(
        year=year, month=month, day=min(value.day, monthrange(year, month)[1])
    )


def workflow_identity(source, settings) -> str:
    return identity({"source": source, "settings": settings.model_dump(mode="json")})


def simulation_clock(settings, *, source_url=None, platform_url=None) -> dict:
    """The monthly workflow's simulated time and the next cutoff it would publish.

    Months without any transactions or labels would be skipped by the workflow; the
    IBM timeline has none before its horizon, so the next cutoff is the next month.
    """
    source = source_import(source_url)
    with connect(platform_url) as conn:
        state = conn.execute(
            "SELECT * FROM monthly_snapshot_progress WHERE workflow_id=%s",
            (workflow_identity(source, settings),),
        ).fetchone()
    if not state or not state["last_cutoff"]:
        raise ValueError("Create the initial snapshot first: make initial-snapshot")
    horizon = datetime.fromisoformat(source["end_exclusive"])
    return {
        "simulation_id": state["simulation_id"],
        "import_id": source["import_id"],
        "clock": state["last_cutoff"],
        "next_cutoff": min(next_month(state["last_cutoff"]), horizon),
        "snapshot_id": state["last_snapshot_id"],
    }


def snapshot_result(store, manifest, *, reused=False):
    return {
        "status": "already_exists" if reused else "created",
        "snapshot_id": manifest["snapshot_id"],
        "as_of": manifest["as_of"],
        "manifest_uri": store.uri(f"snapshots/{manifest['snapshot_id']}/manifest.json"),
        "splits": manifest["splits"],
        "report_uri": store.uri(manifest["reports"]["html"]["key"]),
    }


def monthly_snapshot(
    *,
    initial,
    settings,
    store,
    source_url=None,
    platform_url=None,
    work_dir=None,
    fault=None,
):
    source = source_import(source_url)
    if source["label_delay_days"] != settings.label_delay_days:
        raise ValueError(
            "Source label delay differs from config/data.json; seed the database with the current configuration"
        )
    workflow_id = workflow_identity(source, settings)
    with connect(platform_url) as conn:
        conn.autocommit = True
        lock_id = int(identity({"monthly_workflow": workflow_id})[:15], 16)
        if not conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS locked", (lock_id,)
        ).fetchone()["locked"]:
            raise ValueError(
                "Another snapshot command is running; retry after it finishes"
            )
        try:
            state = conn.execute(
                "SELECT * FROM monthly_snapshot_progress WHERE workflow_id=%s",
                (workflow_id,),
            ).fetchone()
            if not initial and (not state or not state["last_snapshot_id"]):
                raise ValueError(
                    "Create the initial snapshot first: make initial-snapshot"
                )
            if initial and state and state["initial_snapshot_id"]:
                key = f"snapshots/{state['initial_snapshot_id']}/manifest.json"
                if store.exists(key):
                    with SnapshotReader(store, state["initial_snapshot_id"]) as reader:
                        return snapshot_result(store, reader.verify(), reused=True)
                # A fresh stream rebuilds after ephemeral S3 state is lost.
                conn.execute(
                    "DELETE FROM monthly_snapshot_progress WHERE workflow_id=%s",
                    (workflow_id,),
                )
                state = None
            if not state:
                conn.execute(
                    "INSERT INTO monthly_snapshot_progress(workflow_id,simulation_id) VALUES (%s,%s)",
                    (workflow_id, uuid4().hex),
                )
                state = conn.execute(
                    "SELECT * FROM monthly_snapshot_progress WHERE workflow_id=%s",
                    (workflow_id,),
                ).fetchone()
            horizon = datetime.fromisoformat(source["end_exclusive"])
            if initial:
                cutoff = min(midnight(settings.initial_snapshot), horizon)
            else:
                if not store.exists(
                    f"snapshots/{state['last_snapshot_id']}/manifest.json"
                ):
                    raise ValueError(
                        "Snapshot storage was reset; run make initial-snapshot to rebuild"
                    )
                previous = state["last_cutoff"]
                with connect_source(source_url) as upstream:
                    available = upstream.execute(
                        """SELECT min(CASE WHEN tx_datetime >= %(previous)s
                        THEN tx_datetime ELSE label_available_at END) AS next_event
                        FROM ibm_complete_transactions WHERE import_id=%(import_id)s
                        AND tx_datetime >= %(start)s
                        AND (tx_datetime >= %(previous)s OR label_available_at > %(previous)s)""",
                        {
                            "previous": previous,
                            "import_id": source["import_id"],
                            "start": settings.dataset.context_start,
                        },
                    ).fetchone()["next_event"]
                if available is None or previous >= horizon:
                    return {
                        "status": "no_new_data",
                        "message": "No new transactions or labels are available. No snapshot was created.",
                        "last_snapshot_id": state["last_snapshot_id"],
                        "as_of": previous.isoformat(),
                    }
                cutoff = min(next_month(previous), horizon)
                while available >= cutoff and cutoff < horizon:
                    cutoff = min(next_month(cutoff), horizon)
            manifest = create_snapshot(
                store=store,
                profile=settings.dataset,
                as_of=cutoff.date().isoformat(),
                source_url=source_url,
                platform_url=platform_url,
                import_id=source["import_id"],
                simulation=state["simulation_id"],
                work_dir=work_dir,
                fault=fault,
            )
            conn.execute(
                """UPDATE monthly_snapshot_progress SET
                initial_snapshot_id=COALESCE(initial_snapshot_id,%s),last_snapshot_id=%s,
                last_cutoff=%s,updated_at=now() WHERE workflow_id=%s""",
                (manifest["snapshot_id"], manifest["snapshot_id"], cutoff, workflow_id),
            )
            return snapshot_result(store, manifest)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
