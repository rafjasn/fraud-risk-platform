from datetime import datetime, timedelta
from decimal import Decimal

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_data_reports import diagnostics_report, load, observation

from fraud_risk_platform.dataops.reports.data import snapshot_statistics
from fraud_risk_platform.dataops.reports.fraud import category_groups, fraud_exploration
from fraud_risk_platform.dataops.reports.fraud_charts import (
    amount_distribution,
    amount_hour_heatmap,
)
from fraud_risk_platform.dataops.reports.html import fraud_section


def explore(rows, cutoff=datetime(2019, 2, 1)):
    db = load(duckdb.connect(), "training", rows)
    return fraud_exploration(db, "training", cutoff=cutoff)


def test_pending_labels_are_excluded_and_signed_boundaries_are_preserved():
    cutoff = datetime(2019, 1, 4)
    rows = [
        observation(i, amount=amount, fraud=fraud)
        for i, (amount, fraud) in enumerate(
            [("-10", False), ("0", True), ("10", False), ("10000", True)], 1
        )
    ]
    # Both an absent outcome and a supplied outcome unavailable at cutoff are unknown.
    rows += [
        observation(5, fraud=None),
        observation(6, fraud=True).model_copy(
            update={"label_available_at": cutoff + timedelta(seconds=1)}
        ),
        observation(7, fraud=False).model_copy(update={"label_available_at": cutoff}),
    ]
    result = explore(rows, cutoff)
    assert result["excluded_pending_labels"] == 2
    assert result["classes"]["fraud"]["amount_bins"] == [0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert result["classes"]["legitimate"]["amount_bins"] == [0, 0, 0, 1, 0, 2, 0, 0, 0]
    assert result["payment_types"] == {"Online Transaction": {"rows": 5, "frauds": 2}}
    assert result["mcc"] == {"5411": {"rows": 5, "frauds": 2}}
    for counts in result["classes"].values():
        assert sum(map(sum, counts["amount_hour"])) == counts["rows"]


def test_scatter_is_bounded_reproducible_and_does_not_change_rate_counts(monkeypatch):
    monkeypatch.setattr(
        "fraud_risk_platform.dataops.reports.fraud.SCATTER_LIMITS",
        {"legitimate": 7, "fraud": 3},
    )
    rows = [observation(i, amount=str(i), fraud=i % 3 == 0) for i in range(1, 61)]
    result = explore(rows)
    assert result == explore(list(reversed(rows)))
    assert len(result["scatter"]["points"]["legitimate"]) == 7
    assert len(result["scatter"]["points"]["fraud"]) == 3
    assert result["classes"]["legitimate"]["rows"] == 40
    assert result["classes"]["fraud"]["rows"] == 20
    assert result["mcc"]["5411"] == {"rows": 60, "frauds": 20}
    shifted = explore(
        [row.model_copy(update={"amount": row.amount + Decimal(1000)}) for row in rows]
    )
    for name, points in result["scatter"]["points"].items():
        assert [p[1] for p in points] == [
            p[1] - 1000 for p in shifted["scatter"]["points"][name]
        ]


def test_only_selected_training_rows_enter_fraud_exploration(dataset_profile):
    _, report = diagnostics_report(dataset_profile)
    stats = report["fraud_exploration"]
    assert sum(c["rows"] for c in stats["classes"].values()) == 10
    assert stats["classes"]["fraud"]["rows"] == report["splits"]["train"]["frauds"]
    assert report["summary"]["rows"] == 40
    assert report["sampling"]["eligible"]["rows"] == 23
    windows = report["windows"]

    def statistics(training):
        db = load(
            load(duckdb.connect(), "observations", [observation()]),
            "training",
            training,
        )
        return snapshot_statistics(
            db, profile=dataset_profile, cutoff=datetime(2019, 2, 10), windows=windows
        )

    assert all(c["rows"] == 0 for c in statistics([])["fraud"]["classes"].values())
    train_start, train_end = map(datetime.fromisoformat, windows["train"])
    for timestamp in (
        train_start - timedelta(seconds=1),
        train_end,
        datetime(2019, 2, 9),
    ):
        with pytest.raises(ValueError, match="training window"):
            statistics([observation(timestamp=timestamp)])


def test_category_selection_uses_volume_and_preserves_remainder():
    groups = {
        "large": {"rows": 1000, "frauds": 1},
        "tiny_high_rate": {"rows": 2, "frauds": 2},
        "medium": {"rows": 300, "frauds": 0},
        "small": {"rows": 50, "frauds": 1},
    }
    assert category_groups(groups, 2) == [
        ("large", {"rows": 1000, "frauds": 1}),
        ("medium", {"rows": 300, "frauds": 0}),
        ("Other", {"rows": 52, "frauds": 3}),
    ]


def test_amount_plot_normalizes_each_class_and_heatmap_masks_low_support():
    # 100 legitimate in first band; 99 legitimate + one fraud in second; one fraud in third.
    result = explore(
        [
            observation(
                i,
                amount="5" if i <= 100 else "50" if i <= 200 else "500",
                fraud=i >= 200,
            )
            for i in range(1, 202)
        ]
    )
    fig = amount_distribution(result)
    try:
        assert [
            sum(bar.get_height() for bar in group) for group in fig.axes[0].containers
        ] == [100, 100]
    finally:
        plt.close(fig)
    fig = amount_hour_heatmap(result)
    try:
        rates = fig.axes[0].images[0].get_array()
        assert rates[4, 12] == 0
        assert rates[5, 12] == 1
        assert not np.ma.getmaskarray(rates)[4, 12]  # Supported zero-fraud cell.
        assert np.ma.getmaskarray(rates)[6, 12]  # One observation, 100% rate hidden.
        assert np.ma.getmaskarray(rates)[4, 0]  # No observations.
        assert "1/1*" in [text.get_text() for text in fig.axes[0].texts]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fraud", [None, False, True])
def test_empty_and_single_class_reports_do_not_invent_comparisons(fraud):
    stats = explore([observation(fraud=fraud)])
    html = fraud_section({"kind": "Snapshot", "fraud_exploration": stats})
    if fraud is None:
        assert "No labeled training transactions" in html
        assert "data:image/png" not in html
    else:
        assert "Only one class is present" in html
        assert html.count("data:image/png") == 4
    assert not plt.get_fignums()


def test_legacy_reports_explain_how_to_get_missing_plots():
    assert "Refresh the snapshot report" in fraud_section({"kind": "Snapshot"})
    assert fraud_section({"kind": "Ingestion"}) == ""
