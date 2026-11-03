"""Report charts render for qualified, unqualified, and partially scored cycles."""

import numpy as np
import pytest

from fraud_risk_platform.evaluation.charts import (
    population,
    write_comparison,
    write_evaluation,
    write_learning_curves,
    write_trials,
    write_validation,
)


@pytest.fixture
def scored():
    generator = np.random.default_rng(7)
    labels = np.r_[np.zeros(990, dtype=int), np.ones(10, dtype=int)]
    scores = np.r_[generator.uniform(0, 0.6, 990), generator.uniform(0.4, 1, 10)]
    return labels, scores


@pytest.mark.parametrize("threshold", [None, 0.5])
def test_validation_chart_with_and_without_threshold(tmp_path, scored, threshold):
    labels, scores = scored
    filename = write_validation(
        tmp_path, labels, scores, threshold=threshold, review_budget=0.005, title="t"
    )
    assert filename == "validation.png"
    assert (tmp_path / filename).stat().st_size > 0


def test_tied_scores_still_render(tmp_path, scored):
    labels, _ = scored
    write_validation(
        tmp_path,
        labels,
        np.full(len(labels), 0.3),
        threshold=None,
        review_budget=0.005,
        title="t",
    )
    assert (tmp_path / "validation.png").exists()


def test_trials_and_single_model_evaluation(tmp_path, scored):
    labels, scores = scored
    trial = {
        "number": 0,
        "validation_average_precision": 0.2,
        "validation_ranking": {"recall_at_top_0.5pct": 0.3},
        "train_average_precision": 0.3,
        "qualified": False,
    }
    selected = {**trial, "number": 1, "qualified": True}
    write_trials(
        tmp_path,
        {
            "a": {"trials": [trial], "best": None},
            "b": {"trials": [trial, selected], "best": selected},
        },
        review_budget=0.005,
        title="t",
    )
    write_evaluation(
        tmp_path,
        labels,
        incumbent=("Incumbent", scores, 0.5),
        review_budget=0.005,
        title="t",
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "trials.png",
        "evaluation.png",
    }


def test_recipe_comparison_with_and_without_thresholds(tmp_path, scored):
    labels, scores = scored
    models = [
        ("logistic", scores, 0.5),
        ("boosting", np.sqrt(scores), None),
        ("mlp", scores**2, 0.3),
    ]
    filename = write_comparison(
        tmp_path, labels, models, review_budget=0.005, title="t"
    )
    assert (tmp_path / filename).name == "comparison.png"
    with pytest.raises(ValueError, match="Too many recipes"):
        write_comparison(tmp_path, labels, models * 3, review_budget=0.005, title="t")


@pytest.mark.parametrize("count", [1, 2])
def test_learning_curves_render_one_panel_per_model(tmp_path, count):
    curve = {"train_loss": [0.06, 0.03, 0.025], "validation_loss": [0.04, 0.032, 0.034]}
    filename = write_learning_curves(
        tmp_path, [(f"mlp {i}", curve) for i in range(count)], title="t"
    )
    assert (tmp_path / filename).name == "learning_curves.png"


def test_population_states_inclusive_period():
    window = ["2019-05-02T00:00:00", "2019-06-01T00:00:00"]
    assert (
        population("Validation", window, [0, 1, 0])
        == "Validation, 2019-05-02 to 2019-05-31: 3 rows, 1 frauds"
    )
