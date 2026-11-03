"""One simulated month as a single command: replay, advance, monitor, train.

The month starts at the simulation's current cutoff. Traffic up to the next cutoff
is replayed through the deployed release, the next snapshot delivers that month and
its released labels, monitoring reports the replayed predictions, and a training
cycle compares challengers with the deployed release. Promotion stays an explicit
operator step, so the command stops at the cycle's recommendation.

With `train=False` the command stops before training and returns the snapshot to
train, so a workflow can train it elsewhere, such as in a SageMaker job whose imported
release brings its cycle report back.

Reruns are safe: a month already replayed is not replayed again, and when the latest
snapshot has no completed training cycle the command resumes at monitoring and
training instead of advancing past it. Training runs as a ledgered job, so a failed
cycle is retried as the same job's next attempt.

A snapshot whose evaluation window reaches the reserved final holdout is never
trained. The month stops after monitoring with status `holdout_reached`, and later
runs stop there again, because that snapshot never gets a completed cycle. The holdout
is scored once, by `make final-evaluation`.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from psycopg import sql

from fraud_risk_platform.dataops.reader import SnapshotReader
from fraud_risk_platform.dataops.workflow import monthly_snapshot, simulation_clock
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.monitoring.replay import any_replays, replay, replayed
from fraud_risk_platform.monitoring.report import monitor
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.persistence.files import read_json, work_root
from fraud_risk_platform.training.inputs import reaches_holdout
from fraud_risk_platform.training.jobs import run_training_job
from fraud_risk_platform.training.tracking import archive_tracking

logger = logging.getLogger(__name__)
PROMOTIONS = {"promote", "initial_promotion"}
# Platform tables a simulation writes. The source import is in another database.
SIMULATION_TABLES = (
    "predictions",
    "replays",
    "training_attempts",
    "training_jobs",
    "ingestion_batches",
    "ingestion_attempts",
    "ingestion_streams",
    "monthly_snapshot_progress",
)


def reset_simulation(
    root: Path, *, platform_url=None, tracking_uri: str | None = None
) -> dict:
    """Clear one simulation's records, so the next initial-snapshot starts afresh.

    MLflow's experiment and registered model are renamed with an archive suffix, the
    platform database's simulation tables are emptied, and everything in the state
    directory moves to archive/<UTC time>/. The source import, S3 objects, and earlier
    archives stay. MLflow goes first, so an unreachable server changes nothing.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tracking = archive_tracking(stamp, tracking_uri)
    with connect(platform_url) as conn:
        cleared = {
            table: conn.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
            ).fetchone()["count"]
            for table in SIMULATION_TABLES
        }
        conn.execute(
            sql.SQL("TRUNCATE {}").format(
                sql.SQL(", ").join(map(sql.Identifier, SIMULATION_TABLES))
            )
        )
    archive = root / "archive" / stamp
    moved = [
        path
        for path in sorted(root.iterdir() if root.exists() else [])
        if path.name != "archive"
    ]
    if moved:
        archive.mkdir(parents=True)
    for path in moved:
        path.rename(archive / path.name)
    logger.info(
        "Simulation reset: cleared %s; state archived in %s",
        ", ".join(f"{count} {table}" for table, count in cleared.items()),
        archive if moved else "nothing to archive",
    )
    return {
        "status": "reset",
        "cleared_rows": cleared,
        "mlflow_archived": tracking,
        "state_archive": str(archive) if moved else None,
        "next": "make initial-snapshot",
    }


def latest_cycle(root: Path, snapshot_id: str) -> dict | None:
    """The most recently completed training report for a snapshot, if any."""
    completed = [
        path
        for path in (root / "cycles").glob("*/complete.json")
        if read_json(path.parent / "report.json").get("snapshot_id") == snapshot_id
    ]
    if not completed:
        return None
    latest = max(completed, key=lambda path: path.stat().st_mtime)
    return read_json(latest.parent / "report.json")


def run_month(
    root: Path,
    settings,
    policy,
    *,
    store,
    trials: int | None = None,
    skip_promotion: bool = False,
    train: bool = True,
    source_url=None,
    platform_url=None,
    tracking_uri: str | None = None,
) -> dict:
    urls = {"source_url": source_url, "platform_url": platform_url}
    clock = simulation_clock(settings, **urls)
    snapshot_id = clock["snapshot_id"]
    steps = {}
    previous = latest_cycle(root, snapshot_id)
    if previous is None:
        logger.info(
            "Snapshot %s has no completed training cycle; resuming at monitoring and training",
            snapshot_id[:12],
        )
        steps["resumed"] = snapshot_id
    else:
        if clock["clock"] >= clock["next_cutoff"]:
            return {
                "status": "complete",
                "message": "The simulation has reached the end of the source data",
                "simulation_clock": clock["clock"].isoformat(),
            }
        recommended = (
            previous["release_id"] if previous["recommendation"] in PROMOTIONS else None
        )
        if (
            recommended
            and recommended != deployment(root)["current"]
            and not skip_promotion
        ):
            raise ValueError(
                f"The last cycle recommends promoting {recommended[:12]}. Run "
                f"make promote RELEASE={recommended}{' INITIAL=--initial' if previous['recommendation'] == 'initial_promotion' else ''}, "
                "or continue with the deployed release: make month SKIP_PROMOTION=1"
            )
        if replayed(clock["simulation_id"], clock["clock"], platform_url):
            steps["replay"] = "already replayed"
        else:
            steps["replay"] = replay(root, settings, **urls)
        advanced = monthly_snapshot(
            initial=False,
            settings=settings,
            store=store,
            work_dir=work_root(root),
            **urls,
        )
        steps["snapshot"] = {
            key: advanced.get(key)
            for key in ("status", "snapshot_id", "as_of", "message")
        }
        if advanced["status"] == "no_new_data":
            return {"status": "complete", "steps": steps}
        snapshot_id = advanced["snapshot_id"]
    if any_replays(clock["simulation_id"], platform_url):
        monitoring = monitor(root, settings, review_budget=policy.review_budget, **urls)
        steps["monitoring"] = {
            "report": monitoring["report"],
            "latest_window": monitoring["windows"][-1],
        }
    else:
        steps["monitoring"] = "no replayed traffic yet"
    with SnapshotReader(store, snapshot_id) as reader:
        holdout = reaches_holdout(reader.manifest, reader.profile)
    if holdout:
        logger.info(
            "Snapshot %s reaches the final holdout; it is scored by final-evaluation, not trained",
            snapshot_id[:12],
        )
        return {
            "status": "holdout_reached",
            "snapshot_id": snapshot_id,
            "steps": steps,
            "next": "make final-evaluation",
        }
    if not train:
        return {"status": "ready_to_train", "snapshot_id": snapshot_id, "steps": steps}
    report = run_training_job(
        root=root,
        snapshot=snapshot_id,
        policy=policy,
        trials=trials,
        platform_url=platform_url,
        tracking_uri=tracking_uri,
    )
    steps["training"] = {
        key: report.get(key)
        for key in (
            "run_id",
            "validation_winner",
            "recommendation",
            "reason",
            "release_id",
        )
    }
    promotable = report["recommendation"] in PROMOTIONS
    return {
        "status": "trained",
        "snapshot_id": snapshot_id,
        "steps": steps,
        "next": f"make promote RELEASE={report['release_id']}"
        + (
            " INITIAL=--initial"
            if report["recommendation"] == "initial_promotion"
            else ""
        )
        if promotable
        else "No promotion recommended; the deployed release keeps serving",
    }
