"""Training jobs: identical requests share one job, failures retry, successes are reused."""

import shutil

import mlflow
import psycopg
import pytest

from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.persistence.files import identity, read_json
from fraud_risk_platform.training import cycle, jobs
from fraud_risk_platform.training.jobs import job_request, list_jobs, run_training_job


class Killed(BaseException):
    """A process stopping mid-attempt, which no `except Exception` handler sees."""


def test_identical_requests_share_a_job_and_any_difference_separates_them(policy):
    base = {"snapshot": "s" * 64, "policy": policy, "incumbent_release_id": None}
    job = identity(job_request(**base))
    assert identity(job_request(**base)) == job
    # The default trial count, stated explicitly, is the same request.
    assert identity(job_request(**base, trials=policy.search.trials)) == job
    for change in (
        {"snapshot": "t" * 64},
        {"incumbent_release_id": "r" * 64},
        {"policy": policy.model_copy(update={"review_budget": 0.2})},
        {"recipe_name": "logistic_basic"},
        {"trials": policy.search.trials + 1},
        {"refit_snapshot": "u" * 64},
        {"bootstrap": True},
        {"experiment": "repeat"},
    ):
        assert identity(job_request(**{**base, **change})) != job, change


@pytest.fixture
def train_job(tmp_path, snapshot_job, database, policy):
    snapshot = snapshot_job()["snapshot_id"]

    def run(**options):
        return run_training_job(
            root=tmp_path,
            snapshot=snapshot,
            policy=policy,
            platform_url=database,
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            recipe_name="logistic_basic",
            trials=1,
            **options,
        )

    return run


@pytest.mark.integration
def test_failed_attempt_is_retried_and_success_is_reused(
    train_job, tmp_path, database, monkeypatch
):
    def failed_search(**kwargs):
        raise RuntimeError("fit failed")

    with monkeypatch.context() as patch:
        patch.setattr(cycle, "run_search", failed_search)
        with pytest.raises(RuntimeError, match="fit failed"):
            train_job()
    report = train_job()
    job = report["training_job"]
    assert (job["attempt"], job["reused"]) == (2, False)

    again = train_job()
    assert again["training_job"] == {**job, "reused": True}
    assert again["run_id"] == report["run_id"]
    assert len(list((tmp_path / "cycles").glob("*/complete.json"))) == 1

    [listed] = list_jobs(database)
    assert (listed["job_id"], listed["status"]) == (job["job_id"], "succeeded")
    failed, succeeded = listed["attempts"]
    assert (failed["status"], failed["error"]) == ("failed", "RuntimeError: fit failed")
    # The failed attempt keeps its MLflow run and failure report.
    failure = read_json(tmp_path / "cycles" / failed["run_id"] / "report.json")
    assert failure["recommendation"] == "failed"
    tags = mlflow.MlflowClient().get_run(failed["run_id"]).data.tags
    assert (tags["training_job_id"], tags["training_attempt"]) == (job["job_id"], "1")
    assert (
        succeeded["run_id"],
        succeeded["recommendation"],
        succeeded["release_id"],
    ) == (report["run_id"], report["recommendation"], report["release_id"])

    # A deliberate repeat is a separate job with its own cycle.
    repeat = train_job(experiment="repeat")
    assert repeat["training_job"]["job_id"] != job["job_id"]
    assert repeat["run_id"] != report["run_id"]


@pytest.mark.integration
def test_running_job_is_refused_and_a_stopped_attempt_is_abandoned(
    train_job, tmp_path, database, monkeypatch
):
    def stopped(**kwargs):
        with pytest.raises(ValueError, match="already running"):
            train_job()
        raise Killed

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "run_cycle", stopped)
        with pytest.raises(Killed):
            train_job()
    assert train_job()["training_job"]["attempt"] == 2
    [listed] = list_jobs(database)
    assert [a["status"] for a in listed["attempts"]] == ["abandoned", "succeeded"]

    # The ledger never holds two successes for one job.
    with connect(database) as conn, pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO training_attempts(job_id,attempt,status) VALUES (%s,3,'succeeded')",
            (listed["job_id"],),
        )
    # A recorded success whose report is gone fails closed instead of retraining.
    shutil.rmtree(tmp_path / "cycles")
    with pytest.raises(ValueError, match="report is missing"):
        train_job()
