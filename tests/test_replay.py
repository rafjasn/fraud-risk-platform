"""Replayed traffic is scored like the API, recorded once, and monitored as outcomes mature."""

from datetime import date, datetime

import pytest

from fraud_risk_platform import simulation as month_module
from fraud_risk_platform.models.deployment import promote
from fraud_risk_platform.monitoring.replay import replay
from fraud_risk_platform.monitoring.report import monitor
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.serving.predictor import FraudPredictor
from fraud_risk_platform.serving.storage import card_history
from fraud_risk_platform.training import jobs
from fraud_risk_platform.training.cycle import run_cycle
from fraud_risk_platform.training.jobs import list_jobs


@pytest.mark.integration
def test_replay_scores_like_the_api_and_is_monitored_as_outcomes_mature(
    tmp_path, simulate, card_policy, databases
):
    settings, urls, advance = simulate(date(2019, 3, 20))
    snapshot = advance(initial=True)
    with pytest.raises(ValueError, match="Promote a release"):
        replay(tmp_path, settings, **urls)
    report = run_cycle(
        root=tmp_path,
        snapshot=snapshot["snapshot_id"],
        policy=card_policy,
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    promote(tmp_path, report["release_id"], initial=True, reason="test")

    result = replay(tmp_path, settings, **urls)
    # The month after the 2019-03-10 cutoff, clamped to the source horizon.
    assert result["window"] == ["2019-03-10T00:00:00", "2019-03-24T00:00:00"]
    assert result["transactions"] == 12 * 8 and result["rejected"] == {}
    with connect(databases["platform"]) as conn:
        stored = conn.execute(
            """SELECT fraud_score, feature_inputs FROM predictions
            WHERE replay_id=%s AND transaction_id=567""",
            (result["replay_id"],),
        ).fetchone()
    # Source row 567: 2019-03-12 12:06, user 3 card 0, a fraud. The API path agrees.
    served = FraudPredictor(
        root=tmp_path,
        release_id=report["release_id"],
        history=lambda **query: card_history(
            lambda: connect(databases["source"]), **query
        ),
    ).predict(
        amount=1000.0,
        timestamp=datetime(2019, 3, 12, 12, 6),
        use_chip="Online Transaction",
        user_id=3,
        card_id=0,
    )
    assert stored["fraud_score"] == pytest.approx(served["fraud_score"], rel=1e-12)
    assert stored["feature_inputs"] == {
        "user_id": 3,
        "card_id": 0,
        "use_chip": "Online Transaction",
    }
    with pytest.raises(ValueError, match="already replayed"):
        replay(tmp_path, settings, **urls)

    # At the window's start no outcome in it has been released.
    early = monitor(tmp_path, settings, review_budget=0.3, **urls)["windows"][0]
    assert (early["mature"], early["pending"], early["recall"]) == (0, 96, None)
    advance()
    later = monitor(tmp_path, settings, review_budget=0.3, **urls)
    window = later["windows"][0]
    assert (window["mature"], window["pending"], window["frauds"]) == (96, 0, 24)
    assert 0 <= window["recall"] <= 1 and window["expected"]["recall"] is not None
    assert (tmp_path / "monitoring" / "2019-03-24" / "report.html").exists()
    with pytest.raises(ValueError, match="end of the source data"):
        replay(tmp_path, settings, **urls)


@pytest.mark.integration
def test_month_runs_the_loop_waits_for_promotion_and_resumes_after_failure(
    tmp_path, simulate, card_policy, object_store, monkeypatch
):
    settings, urls, advance = simulate(date(2019, 4, 1))
    first_snapshot = advance(initial=True)

    def month(**options):
        return month_module.run_month(
            tmp_path,
            settings,
            card_policy,
            store=object_store,
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            **urls,
            **options,
        )

    # The initial snapshot has no cycle yet, so the loop starts with its training.
    initial = month()
    assert initial["steps"]["resumed"] == first_snapshot["snapshot_id"]
    assert initial["steps"]["monitoring"] == "no replayed traffic yet"
    assert initial["steps"]["training"]["recommendation"] == "initial_promotion"
    with pytest.raises(ValueError, match="recommends promoting"):
        month()
    promote(
        tmp_path,
        initial["steps"]["training"]["release_id"],
        initial=True,
        reason="test",
    )

    def failed_fit(**kwargs):
        raise RuntimeError("fit failed")

    # A scoped patch: undoing the whole fixture would also drop its S3 settings.
    with monkeypatch.context() as patch:
        patch.setattr(jobs, "run_cycle", failed_fit)
        with pytest.raises(RuntimeError, match="fit failed"):
            month()
    # The rerun trains the snapshot it already published rather than skipping a month,
    # as the next attempt of the same training job.
    resumed = month()
    assert "replay" not in resumed["steps"] and "snapshot" not in resumed["steps"]
    latest_job = list_jobs(urls["platform_url"])[0]
    assert [a["status"] for a in latest_job["attempts"]] == ["failed", "succeeded"]
    latest = resumed["steps"]["monitoring"]["latest_window"]
    assert latest["window"] == ["2019-03-10T00:00:00", "2019-03-24T00:00:00"]
    assert (latest["transactions"], latest["pending"]) == (96, 0)
    assert resumed["steps"]["training"]["recommendation"] in {"promote", "keep_current"}
    assert month(skip_promotion=True)["status"] == "complete"


@pytest.mark.integration
def test_month_can_stop_before_training_and_resumes_at_the_same_snapshot(
    tmp_path, simulate, card_policy, object_store
):
    settings, urls, advance = simulate(date(2019, 4, 1))
    snapshot = advance(initial=True)

    def month():
        return month_module.run_month(
            tmp_path,
            settings,
            card_policy,
            store=object_store,
            train=False,
            **urls,
        )

    prepared = month()
    assert prepared["status"] == "ready_to_train"
    assert prepared["snapshot_id"] == snapshot["snapshot_id"]
    assert not (tmp_path / "cycles").exists()
    # Until a cycle for that snapshot completes elsewhere, the month does not advance.
    assert month()["snapshot_id"] == snapshot["snapshot_id"]
