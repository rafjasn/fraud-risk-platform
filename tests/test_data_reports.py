from datetime import datetime, timedelta
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

from fraud_risk_platform.dataops.dataset import Observation, arrow_schema
from fraud_risk_platform.dataops.history_store import HistoryStore
from fraud_risk_platform.dataops.reports.data import (
    render_report,
    snapshot_report,
    snapshot_statistics,
)
from fraud_risk_platform.dataops.reports.summary import (
    calendar_series,
    distribution_distance,
    summarize,
)
from fraud_risk_platform.persistence.files import canonical


def observation(index=1, *, timestamp=None, amount="10.50", fraud=False):
    timestamp = timestamp or datetime(2019, 1, 1, 12)
    return Observation(
        source_row=index,
        user_id=index % 4,
        card_id=0,
        tx_datetime=timestamp,
        amount=Decimal(amount),
        use_chip="Online Transaction",
        merchant_id=7,
        merchant_city="ONLINE",
        merchant_state=None,
        postal_code=None,
        mcc="5411",
        errors=None,
        is_fraud=fraud,
        label_available_at=timestamp + timedelta(days=2),
    )


def delivery(path, rows):
    """A raw JSONL delivery part, written as ingestion writes it."""
    path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    return [path]


def load(db, name, rows):
    """Observations as a DuckDB table, typed as snapshot Parquet is."""
    db.register(
        "loaded_rows",
        pa.Table.from_pylist([row.model_dump() for row in rows], schema=arrow_schema()),
    )
    db.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM loaded_rows")
    db.unregister("loaded_rows")
    return db


def summary_of(rows, **options):
    return summarize(load(duckdb.connect(), "rows", rows), "rows", **options)


def diagnostics_report(profile, *, previous=None, start_day=0):
    cutoff = datetime(2019, 2, 10)
    windows = profile.windows("2019-02-10", 2)
    partition_counts = {
        key: {"rows": 0, "frauds": 0, "unavailable_labels": 0}
        for key in ("context", "train", "validation", "evaluation")
    }
    context, training = [], []
    for day in range(start_day, 40):
        row = observation(
            day + 1,
            timestamp=datetime(2019, 1, 1, 12) + timedelta(days=day),
            amount="-10.50" if day % 5 == 0 else "1000.00",
            fraud=None if day >= 38 else day % 3 == 0,
        )
        context.append(row)
        for name, count in partition_counts.items():
            if name != "context":
                begin, end = map(datetime.fromisoformat, windows[name])
                if not begin <= row.tx_datetime < end:
                    continue
                if name == "train":
                    if len(training) >= 10:
                        continue
                    training.append(row)
            count["rows"] += 1
            count["frauds"] += row.is_fraud is True
            count["unavailable_labels"] += row.is_fraud is None
    db = load(load(duckdb.connect(), "observations", context), "training", training)
    statistics = snapshot_statistics(
        db, profile=profile, cutoff=cutoff, windows=windows, previous=previous
    )
    report = snapshot_report(
        statistics,
        profile=profile,
        cutoff=cutoff,
        windows=windows,
        splits=partition_counts,
        source={"dataset": "fixture"},
        batch_ids=[],
    )
    return statistics, report


def test_summary_exact_signs_quantiles_and_missingness():
    result = summary_of(
        [
            observation(index, amount=amount, fraud=index == 4)
            for index, amount in enumerate(["-100.00", "0.00", "10.00", "100.00"], 1)
        ]
    )
    assert result["amount"]["quantiles"]["p50"] == 5
    assert result["amount"]["negative_rows"] == 1
    assert result["amount"]["zero_rows"] == 1
    assert result["amount"]["min"] == "-100.00"
    assert result["no_error_recorded"] == 4
    assert "errors" not in result["missing"]
    assert result["missing"]["postal_code"] == 4
    assert result["payment_types"] == {"Online Transaction": 4}
    assert sum(result["amount_bins"]) == 4


def test_quantiles_are_exact_and_independent_of_order_and_labels():
    row = observation()
    rows = [
        row.model_copy(update={"source_row": index + 1, "amount": Decimal(index)})
        for index in range(5000)
    ]
    labeled = [r.model_copy(update={"is_fraud": True}) for r in reversed(rows)]
    forward, reverse = summary_of(rows)["amount"], summary_of(labeled)["amount"]
    assert forward == reverse
    assert forward["quantiles"] == pytest.approx(
        {"p01": 49.99, "p50": 2499.5, "p95": 4749.05, "p99": 4949.01}
    )
    assert forward["quantile_method"] == "Exact on all rows"
    assert forward["quantile_sample_rows"] == 5000
    assert (forward["min"], forward["max"]) == ("0.00", "4999.00")
    assert Decimal(forward["mean"]) == Decimal("2499.5")


def test_cohort_labels_count_only_once_released():
    rows = [observation(1, fraud=True), observation(2, fraud=False)]
    released = rows[0].label_available_at
    early = summary_of(rows, as_of=released - timedelta(seconds=1))
    assert (early["known_labels"], early["frauds"], early["unknown_labels"]) == (
        0,
        0,
        2,
    )
    on_time = summary_of(rows, as_of=released)
    assert (on_time["known_labels"], on_time["frauds"]) == (2, 1)
    assert on_time["by_day"] == {"2019-01-01": [2, 2, 1]}


def test_calendar_axis_includes_internal_and_trailing_zero_days():
    summary = summary_of(
        [observation(), observation(2, timestamp=datetime(2019, 1, 3, 12))]
    )
    dates, counts = calendar_series(summary, "2019-01-01", "2019-01-05")
    assert [day.isoformat() for day in dates] == [
        "2019-01-01",
        "2019-01-02",
        "2019-01-03",
        "2019-01-04",
    ]
    assert [row[0] for row in counts] == [1, 0, 1, 0]


def test_distance_is_normalized_and_empty_is_not_zero():
    assert distribution_distance({"a": 10, "b": 10}, {"a": 1, "b": 1}) == 0
    assert distribution_distance({"a": 10}, {"b": 20}) == 1
    assert distribution_distance({}, {"a": 1}) is None


def test_reference_labels_are_masked_at_reference_cutoff(dataset_profile):
    statistics, report = diagnostics_report(
        dataset_profile, previous=datetime(2019, 1, 31)
    )
    c = report["comparison"]
    assert c["current_window"] == ["2019-01-31T00:00:00", "2019-02-10T00:00:00"]
    assert c["reference_window"] == ["2019-01-21T00:00:00", "2019-01-31T00:00:00"]
    assert c["comparable"]
    assert c["current"]["rows"] == c["reference"]["rows"] == 10
    assert c["current"]["unknown_labels"] == c["reference"]["unknown_labels"] == 2
    assert (
        c["reference"]["known_labels"] == 8
    )  # All ten labels are known at snapshot cutoff.
    assert "by_month" not in c["distribution_changes"]
    assert report["sampling"]["eligible"]["rows"] == 23
    assert report["sampling"]["selected"]["rows"] == 10
    assert report["eligible_training_rows"] == statistics["eligible"]["rows"]


def test_incomplete_or_empty_reference_is_explicit(dataset_profile):
    _, report = diagnostics_report(dataset_profile)
    assert not report["comparison"]["reference_complete"]
    assert not report["comparison"]["comparable"]
    assert report["comparison"]["distribution_changes"] == {}
    _, empty = diagnostics_report(
        dataset_profile, previous=datetime(2019, 1, 31), start_day=30
    )
    assert empty["comparison"]["reference_complete"]
    assert not empty["comparison"]["comparable"]
    assert any("no transactions" in text for text in empty["comparison"]["reasons"])


def test_valid_rejected_rows_are_not_accepted_and_html_is_escaped(tmp_path):
    context = HistoryStore(tmp_path / "history.duckdb")
    row = observation().model_dump(mode="json")
    try:
        result = context.apply(
            delivery(
                tmp_path / "raw.jsonl",
                [row, row, {**row, "source_row": 2, "amount": "bad"}],
            ),
            start=datetime(2019, 1, 1),
            cutoff=datetime(2019, 1, 10),
            label_delay_days=2,
        )
    finally:
        context.close()
    report = {
        **result,
        "kind": "Ingestion",
        "status": "rejected",
        "as_of": "2019-01-10T00:00:00",
        "context_start": "2019-01-01T00:00:00",
        "extracted_records": 2,
        "accepted_records": 0,
        "fault": '<script>alert("bad")</script>',
    }
    assert report["received"] == 3
    assert report["valid"] == report["new_transactions"] == 1
    assert report["invalid"] == 2
    html = render_report(report)
    assert "duplicate_identity" in html and "schema.amount" in html
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "valid records in a rejected delivery are not accepted" in html
    assert "No error recorded" in html
    assert 'src="http' not in html
    assert "data:image/png;base64" in html


def test_snapshot_report_has_explanations_and_embedded_charts(dataset_profile):
    _, report = diagnostics_report(dataset_profile, previous=datetime(2019, 1, 31))
    html = render_report(report)
    for text in (
        "Time coverage and training partitions",
        "do not sum all four row counts",
        "Training data checks need attention",
        "Selected <strong>10",
        "23</strong> eligible",
        "Change in the latest transaction cohort",
        "95% Wilson",
        "Recorded findings and provenance",
        "No error recorded",
        "Fraud versus legitimate transactions",
        "Validation, evaluation, and extra context rows are excluded",
        "Class-specific display sampling changes the visible class ratio",
    ):
        assert text in html
    assert html.count("data:image/png;base64") == 8
    assert "<svg" in html
    assert '<details id="provenance">' in html
    # Snapshot identities hash the published report, so rendering is repeatable.
    assert render_report(report) == html


def test_empty_and_legacy_reports_remain_readable():
    html = render_report(
        {"kind": "Ingestion", "status": "failed", "error": "Source unavailable"}
    )
    assert "Source unavailable" in html
    assert "No valid transactions" in html
    assert "Unavailable" in html


def test_training_reasons_explain_holdout_and_class_failures(dataset_profile):
    from fraud_risk_platform.dataops.reports.data import training_findings

    counts = {
        name: {"rows": 10, "frauds": 0}
        for name in ("train", "validation", "evaluation")
    }
    reasons = training_findings(
        dataset_profile, {"evaluation": ["2019-03-01", "2019-04-01"]}, counts
    )
    assert len(reasons) == 4
    assert any("reserved holdout" in reason for reason in reasons)
    assert any("train: 0 frauds" in reason for reason in reasons)


def test_context_lightweight_verification_keeps_label_counts(tmp_path):
    context = HistoryStore(tmp_path / "history.duckdb")
    try:
        result = context.apply(
            delivery(tmp_path / "raw.jsonl", [observation().model_dump(mode="json")]),
            start=datetime(2019, 1, 1),
            cutoff=datetime(2019, 1, 2),
            label_delay_days=2,
            reporting=False,
        )
        assert result["invalid"] == 1  # A future label was exposed.
        assert result["summary"]["rows"] == 0
    finally:
        context.close()
