"""Training jobs: one ledger entry per distinct request, so repeats never retrain.

A job's ID is the hash of its request: the snapshot, refit snapshot, incumbent,
training policy, recipe and trial overrides, package sources, dependency versions,
image identity, and an optional experiment name. Repeating a request that succeeded
reuses its cycle report, a request already running is refused, and a failed request
runs again as its next attempt. Every attempt stays in the ledger with its MLflow run
and error.

A PostgreSQL advisory lock, held for the whole attempt, marks it as running. Closing
the connection releases the lock, so an attempt still marked running without its lock
belongs to a process that died, and is recorded as abandoned.

The SageMaker entrypoint runs cycles directly; on AWS, the workflow execution name
identifies a job instead.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
from psycopg.types.json import Jsonb

from fraud_risk_platform.contracts.training import TrainingPolicy
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.persistence.files import identity, read_json
from fraud_risk_platform.training.cycle import run_cycle
from fraud_risk_platform.training.tracking import dependency_versions, source_checksums

logger = logging.getLogger(__name__)


def snapshot_id(reference) -> str:
    """The ID of a snapshot reference or of a prepared local snapshot directory."""
    directory = Path(str(reference))
    if directory.is_dir():
        return read_json(directory / "manifest.json")["snapshot_id"]
    return str(reference)


def job_request(
    snapshot,
    policy: TrainingPolicy,
    *,
    incumbent_release_id: str | None,
    bootstrap: bool = False,
    recipe_name: str | None = None,
    trials: int | None = None,
    refit_snapshot=None,
    experiment: str | None = None,
) -> dict:
    """Everything that makes two training requests the same job."""
    return {
        "schema_version": 1,
        "purpose": "bootstrap" if bootstrap else "train",
        "snapshot_id": snapshot_id(snapshot),
        "refit_snapshot_id": snapshot_id(refit_snapshot) if refit_snapshot else None,
        "incumbent_release_id": incumbent_release_id,
        "policy_sha256": identity(policy.model_dump()),
        "recipe": recipe_name,
        "trials": trials if trials is not None else policy.search.trials,
        "code_sha256": identity(source_checksums()),
        "dependencies_sha256": identity(dependency_versions()),
        "image_identity": os.environ.get("TRAINING_IMAGE_ID", "unrecorded"),
        "experiment": experiment,
    }


def run_training_job(
    *,
    root: Path,
    snapshot,
    policy: TrainingPolicy,
    platform_url: str | None = None,
    tracking_uri: str | None = None,
    bootstrap: bool = False,
    recipe_name: str | None = None,
    trials: int | None = None,
    refit_snapshot=None,
    experiment: str | None = None,
) -> dict:
    """Run, retry, or reuse the training cycle for this request; returns its report."""
    incumbent = deployment(root)["current"]
    request = job_request(
        snapshot,
        policy,
        incumbent_release_id=incumbent,
        bootstrap=bootstrap,
        recipe_name=recipe_name,
        trials=trials,
        refit_snapshot=refit_snapshot,
        experiment=experiment,
    )
    job_id = identity(request)
    with connect(platform_url) as conn:
        conn.autocommit = True
        lock_id = int(identity({"training_job": job_id})[:15], 16)
        if not conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS locked", (lock_id,)
        ).fetchone()["locked"]:
            raise ValueError(
                f"Training job {job_id[:12]} is already running; retry after it finishes"
            )
        try:
            conn.execute(
                "INSERT INTO training_jobs(job_id,request) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (job_id, Jsonb(request)),
            )
            if conn.execute(
                """UPDATE training_attempts SET status='abandoned',finished_at=now()
                WHERE job_id=%s AND status='running'""",
                (job_id,),
            ).rowcount:
                logger.warning(
                    "Training job %s: an earlier attempt stopped without finishing",
                    job_id[:12],
                )
            succeeded = conn.execute(
                "SELECT attempt,run_id FROM training_attempts WHERE job_id=%s AND status='succeeded'",
                (job_id,),
            ).fetchone()
            if succeeded:
                return _reuse(root, job_id, succeeded)
            attempt = conn.execute(
                """INSERT INTO training_attempts(job_id,attempt,status)
                SELECT %s,COALESCE(max(attempt),0)+1,'running'
                FROM training_attempts WHERE job_id=%s RETURNING attempt""",
                (job_id, job_id),
            ).fetchone()["attempt"]
            where = (job_id, attempt)

            def started(run_id):
                conn.execute(
                    "UPDATE training_attempts SET run_id=%s WHERE job_id=%s AND attempt=%s",
                    (run_id, *where),
                )
                mlflow.set_tags(
                    {"training_job_id": job_id, "training_attempt": str(attempt)}
                )

            logger.info("Training job %s attempt %d started", job_id[:12], attempt)
            try:
                report = run_cycle(
                    root=root,
                    snapshot=snapshot,
                    policy=policy,
                    tracking_uri=tracking_uri,
                    bootstrap=bootstrap,
                    recipe_name=recipe_name,
                    trials=trials,
                    refit_snapshot=refit_snapshot,
                    incumbent_release_id=incumbent,
                    on_start=started,
                )
            except Exception as error:
                conn.execute(
                    """UPDATE training_attempts SET status='failed',error=%s,finished_at=now()
                    WHERE job_id=%s AND attempt=%s""",
                    (f"{type(error).__name__}: {error}", *where),
                )
                raise
            conn.execute(
                """UPDATE training_attempts SET status='succeeded',run_id=%s,
                recommendation=%s,release_id=%s,finished_at=now()
                WHERE job_id=%s AND attempt=%s""",
                (
                    report["run_id"],
                    report["recommendation"],
                    report["release_id"],
                    *where,
                ),
            )
            return {
                **report,
                "training_job": {"job_id": job_id, "attempt": attempt, "reused": False},
            }
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))


def _reuse(root: Path, job_id: str, succeeded: dict) -> dict:
    output = root / "cycles" / succeeded["run_id"]
    if not (output / "complete.json").exists():
        raise ValueError(
            f"Training job {job_id[:12]} succeeded as run {succeeded['run_id']}, but "
            f"its report is missing from {output}; restore the state directory or "
            "train under a new experiment name"
        )
    logger.info(
        "Training job %s already succeeded as run %s; reusing its report",
        job_id[:12],
        succeeded["run_id"],
    )
    return {
        **read_json(output / "report.json"),
        "training_job": {
            "job_id": job_id,
            "attempt": succeeded["attempt"],
            "reused": True,
        },
    }


def list_jobs(platform_url: str | None = None, limit: int = 20) -> list[dict]:
    """The most recent training jobs, newest first, each with every attempt."""
    with connect(platform_url) as conn:
        jobs = conn.execute(
            "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT %s", (limit,)
        ).fetchall()
        attempts = conn.execute(
            "SELECT * FROM training_attempts WHERE job_id = ANY(%s) ORDER BY attempt",
            ([job["job_id"] for job in jobs],),
        ).fetchall()

    def plain(row):
        return {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in row.items()
            if key != "job_id"
        }

    result = []
    for job in jobs:
        history = [plain(row) for row in attempts if row["job_id"] == job["job_id"]]
        result.append(
            {
                "job_id": job["job_id"],
                "status": history[-1]["status"] if history else "not started",
                **plain(job),
                "attempts": history,
            }
        )
    return result
