"""The final holdout is never trained and is scored once; a reset starts a new simulation."""

from datetime import date
from pathlib import Path

import mlflow
import pytest

from fraud_risk_platform import simulation as month_module
from fraud_risk_platform.models.deployment import confirm_rollout, promote
from fraud_risk_platform.monitoring.holdout import _releases, final_evaluation
from fraud_risk_platform.persistence.database import connect


def test_holdout_scores_every_confirmed_release_ending_with_the_champion():
    state = {
        "current": "a",
        "deployed": "a",
        "history": [
            {"action": "initial", "to": "a"},
            {"action": "confirmed", "to": "a"},
            {"action": "promote", "to": "b"},
            {"action": "confirmed", "to": "b"},
            {"action": "rollback", "to": "a"},
            {"action": "confirmed", "to": "a"},
        ],
    }
    assert _releases(state) == ["b", "a"]
    # A selected release that never served is no champion.
    with pytest.raises(ValueError, match="No confirmed champion"):
        _releases({**state, "current": "b"})


@pytest.mark.integration
def test_month_stops_at_the_holdout_which_is_scored_once(
    tmp_path, simulate, card_policy, object_store
):
    # The first update's evaluation window (to 2019-03-15) reaches the holdout.
    settings, urls, advance = simulate(
        date(2019, 2, 20), initial_snapshot=date(2019, 2, 17)
    )
    advance(initial=True)

    def month():
        return month_module.run_month(
            tmp_path,
            settings,
            card_policy,
            store=object_store,
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            **urls,
        )

    initial = month()["steps"]["training"]
    assert initial["recommendation"] == "initial_promotion"
    champion = initial["release_id"]
    promote(tmp_path, champion, initial=True, reason="test")
    confirm_rollout(tmp_path, lambda: champion)

    stopped = month()
    assert stopped["status"] == "holdout_reached"
    assert stopped["steps"]["snapshot"]["as_of"] == "2019-03-17T00:00:00"
    assert "training" not in stopped["steps"]
    # That snapshot never gets a cycle, so later months stop at it too.
    again = month()
    assert again["status"] == "holdout_reached"
    assert again["steps"]["resumed"] == stopped["snapshot_id"]
    assert len(list((tmp_path / "cycles").glob("*/complete.json"))) == 1

    # Its labels end before the holdout month does.
    with pytest.raises(ValueError, match="Advance the simulation"):
        final_evaluation(tmp_path, settings, review_budget=0.3, **urls)
    advance()
    result = final_evaluation(tmp_path, settings, review_budget=0.3, **urls)
    assert result["window"] == ["2019-02-20T00:00:00", "2019-03-20T00:00:00"]
    # 28 days of eight transactions, two of them fraud.
    assert (result["rows"], result["frauds"]) == (224, 56)
    assert result["champion"] == champion
    assert [entry["role"] for entry in result["releases"]] == ["champion"]
    assert 0 <= result["releases"][0]["holdout"]["recall_at_top_30pct"] <= 1
    assert (tmp_path / "final-evaluation" / "report.html").exists()
    with pytest.raises(ValueError, match="already scored"):
        final_evaluation(tmp_path, settings, review_budget=0.3, **urls)


@pytest.mark.integration
def test_reset_starts_a_new_simulation_and_keeps_the_old_one_readable(
    tmp_path, simulate, card_policy, object_store
):
    root = tmp_path / "state"
    tracking = f"sqlite:///{tmp_path}/mlflow.db"
    settings, urls, advance = simulate(date(2019, 4, 1))

    def month():
        return month_module.run_month(
            root,
            settings,
            card_policy,
            store=object_store,
            tracking_uri=tracking,
            **urls,
        )

    advance(initial=True)
    assert month()["status"] == "trained"

    result = month_module.reset_simulation(
        root, platform_url=urls["platform_url"], tracking_uri=tracking
    )
    assert result["cleared_rows"]["training_jobs"] == 1
    assert result["cleared_rows"]["monthly_snapshot_progress"] == 1
    with connect(urls["platform_url"]) as conn:
        for table in month_module.SIMULATION_TABLES:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()["count"]
            assert count == 0, table
    archive = Path(result["state_archive"])
    assert [path.name for path in root.iterdir()] == ["archive"]
    assert (archive / "cycles").is_dir() and (archive / "releases").is_dir()
    # The old runs and versions stay readable under archived names.
    client = mlflow.MlflowClient(tracking)
    assert client.get_experiment_by_name("fraud-risk-training-v2") is None
    assert client.get_experiment_by_name(result["mlflow_archived"]["experiment"])
    assert not client.search_registered_models(filter_string="name = 'fraud-risk'")

    # The next initial snapshot starts a new simulation, trained from scratch.
    assert advance(initial=True)["status"] == "created"
    assert month()["steps"]["training"]["recommendation"] == "initial_promotion"
