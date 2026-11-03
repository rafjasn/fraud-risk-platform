"""Snapshot statistics and immutable, self-contained data reports."""

import json
import logging
from datetime import datetime, timedelta

from fraud_risk_platform.dataops.history_store import HistoryStore
from fraud_risk_platform.dataops.reports.fraud import fraud_exploration
from fraud_risk_platform.dataops.reports.summary import distribution_changes, summarize

# Bumped whenever report content or rendering changes: reports are hashed into
# snapshot identities and stored under a versioned key.
REPORT_VERSION = 5
logger = logging.getLogger(__name__)


def training_findings(profile, windows, splits):
    reasons = []
    for name in ("train", "validation", "evaluation"):
        counts = splits[name]
        if counts["rows"] < profile.minimum_rows_per_split:
            reasons.append(
                f"{name}: {counts['rows']:,} rows; requires at least {profile.minimum_rows_per_split:,}."
            )
        if counts["frauds"] < profile.minimum_frauds_per_split:
            reasons.append(
                f"{name}: {counts['frauds']:,} frauds; requires at least {profile.minimum_frauds_per_split:,}."
            )
        if counts["frauds"] == counts["rows"]:
            reasons.append(
                f"{name}: no legitimate transactions; both classes are required."
            )
    if (
        datetime.fromisoformat(windows["evaluation"][1]).date()
        > profile.final_holdout_start
    ):
        reasons.append(
            f"Evaluation extends into the reserved holdout beginning {profile.final_holdout_start}."
        )
    return reasons


def snapshot_statistics(db, *, profile, cutoff, windows, previous=None):
    """Statistics of db's `observations` (accepted context) and `training` relations.

    Each comparison cohort counts only labels released by its own period end.
    """
    current_start = previous or cutoff - timedelta(days=30)
    reference_start = current_start - (cutoff - current_start)
    train = dict(zip(("start", "end"), map(datetime.fromisoformat, windows["train"])))
    outside = db.execute(
        """SELECT count(*) FROM training
        WHERE NOT (tx_datetime >= $start AND tx_datetime < $end)""",
        train,
    ).fetchone()[0]
    if outside:
        raise ValueError("Fraud exploration requires rows inside the training window")
    window = "observations WHERE tx_datetime >= $start AND tx_datetime < $end"
    current = {"start": current_start, "end": cutoff}
    return {
        "current_start": current_start,
        "reference_start": reference_start,
        "summary": summarize(db, "observations"),
        "eligible": summarize(db, window + " AND is_fraud IS NOT NULL", train),
        "selected": summarize(db, "training"),
        "fraud": fraud_exploration(db, "training", cutoff=cutoff),
        "current": summarize(db, window, current, as_of=cutoff),
        "reference": summarize(
            db,
            window,
            {"start": reference_start, "end": current_start},
            as_of=current_start,
        ),
        "new_customers": db.execute(
            """SELECT count(DISTINCT user_id) FROM observations
            WHERE tx_datetime >= $start AND tx_datetime < $end AND user_id NOT IN
                (SELECT user_id FROM observations WHERE tx_datetime < $start)""",
            current,
        ).fetchone()[0],
    }


def snapshot_report(
    statistics, *, profile, cutoff, windows, splits, source, batch_ids, delivery=None
):
    current, reference = statistics["current"], statistics["reference"]
    eligible, selected = statistics["eligible"], statistics["selected"]
    current_start, reference_start = (
        statistics["current_start"],
        statistics["reference_start"],
    )
    reasons = training_findings(profile, windows, splits)
    complete = profile.context_start <= reference_start
    comparable = complete and bool(current["rows"] and reference["rows"])
    comparison_reasons = []
    if not complete:
        comparison_reasons.append(
            "The snapshot does not cover the full reference period."
        )
    if not current["rows"] or not reference["rows"]:
        comparison_reasons.append(
            "At least one period has no transactions; distribution changes are unavailable."
        )
    changes = distribution_changes(current, reference) if comparable else {}
    changes.pop(
        "by_month", None
    )  # Different months are expected, not evidence of drift.
    summary = statistics["summary"]
    for name, stats in (("context", summary), ("train", selected)):
        expected = splits[name]
        if (
            stats["rows"] != expected["rows"]
            or stats["frauds"] != expected["frauds"]
            or stats["unknown_labels"] != expected["unavailable_labels"]
        ):
            raise ValueError("Report populations disagree with snapshot counts")
    return {
        "report_version": REPORT_VERSION,
        "kind": "Snapshot",
        "status": "accepted",
        "as_of": cutoff.isoformat(),
        "context_start": profile.context_start.isoformat(),
        "population": "all accepted context transactions; supervised partitions overlap context",
        "dataset_profile": profile.model_dump(mode="json"),
        "source": source,
        "batch_ids": batch_ids,
        "windows": windows,
        "splits": splits,
        "eligible_training_rows": eligible["rows"],
        "training_ready": not reasons,
        "training_reasons": reasons,
        "training_criteria": {
            "minimum_rows_per_split": profile.minimum_rows_per_split,
            "minimum_frauds_per_split": profile.minimum_frauds_per_split,
            "final_holdout_start": profile.final_holdout_start.isoformat(),
        },
        "summary": summary,
        "fraud_exploration": statistics["fraud"],
        "sampling": {
            "method": "Deterministic hash of source identity, seed, and row ID; independent of label",
            "seed": profile.selection_seed,
            "fraction": selected["rows"] / eligible["rows"]
            if eligible["rows"]
            else None,
            "eligible": eligible,
            "selected": selected,
            "distribution_changes": distribution_changes(selected, eligible),
        },
        "delivery": delivery,
        "comparison": {
            "current_window": [current_start.isoformat(), cutoff.isoformat()],
            "reference_window": [
                reference_start.isoformat(),
                current_start.isoformat(),
            ],
            "population": "Unique transactions by event time in consecutive equal-duration periods; label updates do not add transactions",
            "label_policy": "Each cohort uses only labels available at its own period end",
            "reference_complete": complete,
            "comparable": comparable,
            "reasons": comparison_reasons,
            "current": current,
            "reference": reference,
            "new_customers": statistics["new_customers"],
            "new_customer_definition": "Not seen earlier in this snapshot's context; not necessarily new to the entire source",
            "distribution_changes": changes,
        },
    }


def read_report(store, descriptor, directory):
    path = directory / "verified-report.json"
    store.verify(descriptor, path)
    return json.loads(path.read_text())


def refresh_snapshot_report(reader):
    """Recompute local statistics from immutable Parquet; never rewrite S3 artifacts."""
    manifest = reader.manifest
    original = read_report(reader.store, manifest["reports"]["json"], reader.directory)
    delivery = original.get("delivery")
    previous = delivery.get("previous_cutoff") if delivery else None
    logger.info(
        "Refreshing snapshot report %s from immutable Parquet", manifest["snapshot_id"]
    )
    context = HistoryStore(reader.directory / "refresh.duckdb")
    try:
        context.load(
            [reader.local_path(d) for d in manifest["files"]["context"]],
            training=[reader.local_path(d) for d in manifest["files"]["train"]],
        )
        statistics = snapshot_statistics(
            context.db,
            profile=reader.profile,
            cutoff=datetime.fromisoformat(manifest["as_of"]),
            windows=manifest["windows"],
            previous=datetime.fromisoformat(previous) if previous else None,
        )
    finally:
        context.close()
    report = snapshot_report(
        statistics,
        profile=reader.profile,
        cutoff=datetime.fromisoformat(manifest["as_of"]),
        windows=manifest["windows"],
        splits=manifest["splits"],
        source=manifest["source"],
        batch_ids=manifest["batch_ids"],
        delivery=delivery,
    )
    if report["eligible_training_rows"] != manifest["selection"]["eligible_rows"]:
        raise ValueError("Report training eligibility disagrees with snapshot manifest")
    report["snapshot_id"] = manifest["snapshot_id"]
    report["report_origin"] = (
        "Regenerated locally from snapshot context and training Parquet; original artifacts unchanged"
    )
    return report


def render_report(report):
    from fraud_risk_platform.dataops.reports.html import render_report as render

    return render(report)


def publish_report(store, prefix, report, directory):
    path = directory / "report.html"
    path.write_text(render_report(report))
    return {
        "json": store.put_json(f"{prefix}/report.json", report),
        "html": store.put_file(
            f"{prefix}/report.html", path, "text/html; charset=utf-8"
        ),
    }
