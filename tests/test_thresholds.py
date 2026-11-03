"""Validation-only decision threshold selection for a review budget."""

import numpy as np
import pytest

from fraud_risk_platform.evaluation.metrics import (
    bootstrap_recall,
    interval,
    metrics,
    ranking_metrics,
    select_threshold,
)


def test_selects_lowest_threshold_within_review_budget():
    scores = [0.9, 0.8, 0.7, 0.1]
    assert select_threshold(scores, review_budget=0.5) == pytest.approx(0.8)
    assert select_threshold(scores, review_budget=0.25) == pytest.approx(0.9)


def test_tie_straddling_the_budget_is_left_unflagged():
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1])
    threshold = select_threshold(scores, review_budget=0.4)
    assert threshold == pytest.approx(0.9)
    assert (scores >= threshold).sum() <= 2


def test_no_threshold_when_budget_cannot_flag_anything():
    # Less than one reviewable transaction, or a top tie group above capacity.
    assert select_threshold([0.9, 0.1], review_budget=0.4) is None
    assert select_threshold([0.5] * 4, review_budget=0.5) is None


def test_budget_capacity_absorbs_float_error():
    # 0.005 * 1000 must allow five reviews, not four.
    scores = np.arange(1000, 0, -1) / 1000
    threshold = select_threshold(scores, review_budget=0.005)
    assert (scores >= threshold).sum() == 5


def test_alert_rate_metrics_split_tied_boundary_scores():
    labels = np.zeros(1000, dtype=int)
    scores = np.full(1000, 0.1)
    scores[:4] = 0.99
    labels[:3] = 1
    scores[4:12] = 0.5
    labels[4:6] = 1
    labels[[100, 200, 300, 400, 500]] = 1
    result = ranking_metrics(labels, scores, review_budget=0.01)
    assert result["prevalence"] == pytest.approx(0.01)
    # Top 1 row falls inside a 4-way tie with 3 frauds: 0.75 expected hits.
    assert result["precision_at_top_0.1pct"] == pytest.approx(0.75)
    assert result["recall_at_top_0.1pct"] == pytest.approx(0.075)
    # Top 5 rows: 4 above the tie (3 frauds) plus 1 of 8 tied rows (2 frauds).
    assert result["precision_at_top_0.5pct"] == pytest.approx(0.65)
    assert result["recall_at_top_0.5pct"] == pytest.approx(0.325)
    assert result["precision_at_top_1pct"] == pytest.approx(0.45)
    assert result["recall_at_top_1pct"] == pytest.approx(0.45)


def test_review_budget_is_added_to_reported_volumes():
    labels = np.r_[np.ones(10, dtype=int), np.zeros(990, dtype=int)]
    scores = np.linspace(1, 0, 1000)
    result = ranking_metrics(labels, scores, review_budget=0.02)
    assert result["recall_at_top_2pct"] == 1.0
    assert "recall_at_top_0.5pct" in result


def test_budget_threshold_recall_matches_ranked_recall_without_ties():
    generator = np.random.default_rng(3)
    labels = np.r_[np.ones(20, dtype=int), np.zeros(980, dtype=int)]
    scores = np.r_[generator.uniform(0.3, 1, 20), generator.uniform(0, 0.7, 980)]
    threshold = select_threshold(scores, review_budget=0.01)
    at_threshold = metrics(labels, scores, threshold)
    assert at_threshold["alert_rate"] == pytest.approx(0.01)
    assert at_threshold["recall"] == pytest.approx(
        ranking_metrics(labels, scores, review_budget=0.01)["recall_at_top_1pct"]
    )


def test_bootstrap_pairs_resamples_across_models():
    generator = np.random.default_rng(5)
    labels = np.r_[np.ones(50, dtype=int), np.zeros(9950, dtype=int)]
    noisy = np.r_[generator.uniform(0.2, 1, 50), generator.uniform(0, 0.8, 9950)]
    perfect = labels + generator.uniform(0, 0.5, 10_000)
    draws = bootstrap_recall(
        labels,
        {"noisy": noisy, "copy": noisy.copy(), "perfect": perfect},
        review_budget=0.01,
        resamples=200,
    )
    # Identical scores give identical recall on every shared resample.
    np.testing.assert_array_equal(draws["noisy"], draws["copy"])
    # Capacity (100 rows) exceeds the resampled frauds, so a perfect ranking catches all.
    assert (draws["perfect"] == 1.0).all()
    low, high = interval(draws["noisy"])
    point = ranking_metrics(labels, noisy, review_budget=0.01)["recall_at_top_1pct"]
    assert low < point < high
    again = bootstrap_recall(
        labels, {"noisy": noisy}, review_budget=0.01, resamples=200
    )
    np.testing.assert_array_equal(again["noisy"], draws["noisy"])
