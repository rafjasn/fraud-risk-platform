import pytest

from fraud_risk_platform.evaluation.metrics import recommendation
from fraud_risk_platform.models.deployment import deployment, promote
from fraud_risk_platform.persistence.files import read_json
from fraud_risk_platform.training.cycle import run_cycle


@pytest.mark.integration
def test_training_failure_preserves_release_and_rejects_overlap(
    tmp_path, snapshot_job, policy
):
    first = snapshot_job()["snapshot_id"]
    tracking = f"sqlite:///{tmp_path}/mlflow.db"
    report = run_cycle(
        root=tmp_path,
        snapshot=first,
        policy=policy,
        tracking_uri=tracking,
        bootstrap=True,
    )
    assert report["recommendation"] == "initial_promotion"
    assert list(report["candidates"]) == ["logistic_basic"]
    assert deployment(tmp_path)["current"] is None
    promote(tmp_path, report["release_id"], initial=True, reason="test")
    pointer = (tmp_path / "deployment.json").read_bytes()
    with pytest.raises(ValueError, match="Bootstrap"):
        run_cycle(
            root=tmp_path,
            snapshot=first,
            policy=policy,
            tracking_uri=tracking,
            bootstrap=True,
        )
    with pytest.raises(ValueError, match="overlaps"):
        run_cycle(root=tmp_path, snapshot=first, policy=policy, tracking_uri=tracking)
    second = snapshot_job(as_of="2019-02-20")["snapshot_id"]
    second_report = run_cycle(
        root=tmp_path, snapshot=second, policy=policy, tracking_uri=tracking
    )
    assert second_report["recommendation"] == "keep_current"
    # Both configured recipes are searched and compared on validation.
    assert list(second_report["comparison"]) == ["logistic_basic", "boosting_basic"]
    winner = second_report["validation_winner"]
    for name, entry in second_report["comparison"].items():
        low, high = entry["recall_interval"]
        assert 0 <= low <= high <= 1
        # The winner's paired lead is reported against every other recipe.
        assert ("winner_lead" in entry) == (name != winner)
    assert second_report["plots"][:2] == ["validation.png", "comparison.png"]
    assert second_report["evaluation"]["incumbent"]["recall"] == 1.0
    invalid_policy = policy.model_copy(deep=True)
    invalid_policy.recipes["boosting_basic"].parameters["unknown_parameter"] = 1
    # Unknown settings fail before any search starts, leaving deployment unchanged.
    with pytest.raises(ValueError, match="Unknown hist_gradient_boosting parameters"):
        run_cycle(
            root=tmp_path, snapshot=second, policy=invalid_policy, tracking_uri=tracking
        )
    assert (tmp_path / "deployment.json").read_bytes() == pointer
    # Only the overlapping cycle started, so only it recorded a failure; like the
    # bootstrap misuse, the unknown setting was rejected before a cycle began.
    failures = [
        read_json(p)
        for p in (tmp_path / "cycles").glob("*/report.json")
        if read_json(p)["recommendation"] == "failed"
    ]
    assert len(failures) == 1


def test_keep_current_is_a_first_class_outcome():
    def allowed(candidate, incumbent=None):
        key = "recall_at_top_0.5pct"
        return recommendation(
            {key: candidate},
            incumbent and {key: incumbent},
            review_budget=0.005,
            minimum_recall_gain=0,
        )[0]

    # Frozen models are compared by evaluation recall at the same review volume.
    assert not allowed(0.5, incumbent=0.5)
    assert allowed(0.6, incumbent=0.5)
    # An initial model must beat reviewing a random 0.5% of transactions.
    assert not allowed(0.005)
    assert allowed(0.2)
