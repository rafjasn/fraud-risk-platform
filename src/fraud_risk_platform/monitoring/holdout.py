"""The final holdout: the deployed champion scored once on the reserved month.

The holdout is the calendar month from `final_holdout_start` (October 2019 by
default). No training cycle fits, selects, or evaluates on it, and `make month` stops
before the first snapshot whose evaluation window reaches it. After the last monthly
update, `final_evaluation` scores the confirmed champion on that month, beside every
release confirmed serving before it, so the result shows both the final model and
whether the monthly updates improved on the first one. No decision follows from it.

The month comes from a verified snapshot whose released labels cover it, with the
training feature code and card history from that snapshot's context. The report is
written once: a second run refuses, so the holdout cannot be rescored after a change.
"""

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from fraud_risk_platform.contracts.base import midnight
from fraud_risk_platform.dataops.workflow import next_month, simulation_clock
from fraud_risk_platform.evaluation.charts import population, write_comparison
from fraud_risk_platform.evaluation.metrics import (
    alert_name,
    bootstrap_recall,
    interval,
    metrics,
    ranking_metrics,
)
from fraud_risk_platform.features.matrices import card_history, window_features
from fraud_risk_platform.features.transaction import artifact_feature_spec
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.files import write_json
from fraud_risk_platform.reporting.page import (
    badge,
    evidence,
    figure,
    page,
    percent,
    section,
    table,
    tiles,
)
from fraud_risk_platform.training.inputs import open_snapshot
from fraud_risk_platform.training.tracking import code_version

logger = logging.getLogger(__name__)


def holdout_window(profile) -> list[str]:
    """The reserved month, as [start, end) timestamps."""
    start = midnight(profile.final_holdout_start)
    return [start.isoformat(), next_month(start).isoformat()]


def _releases(state) -> list[str]:
    """Every release confirmed serving, in order, ending with the champion."""
    champion = state["deployed"]
    if champion is None or champion != state["current"]:
        raise ValueError(
            "No confirmed champion: the selected release must be confirmed serving "
            "(make rollout) before the holdout is scored"
        )
    confirmed = [e["to"] for e in state["history"] if e["action"] == "confirmed"]
    return [r for r in dict.fromkeys(confirmed) if r != champion] + [champion]


def final_evaluation(
    root: Path,
    settings,
    *,
    review_budget: float,
    snapshot=None,
    source_url=None,
    platform_url=None,
) -> dict:
    output = root / "final-evaluation"
    if output.exists():
        raise ValueError(
            f"The final holdout was already scored; see {output / 'report.html'}"
        )
    state = deployment(root)
    releases = _releases(state)
    if snapshot is None:
        snapshot = simulation_clock(
            settings, source_url=source_url, platform_url=platform_url
        )["snapshot_id"]
    with open_snapshot(snapshot) as reader:
        manifest = reader.verify_integrity()
        window = holdout_window(reader.profile)
        # Every transaction before the evaluation window's end has a released label.
        released_until = manifest["windows"]["evaluation"][1]
        if released_until < window[1]:
            raise ValueError(
                f"Snapshot {manifest['snapshot_id'][:12]} releases labels only for "
                f"transactions before {released_until[:10]}; the holdout needs them "
                f"before {window[1][:10]}. Advance the simulation first."
            )
        history = labels = None
        entries, scores = [], {}
        for release_id in releases:
            release, model = load_release(root, release_id)
            if release["source_import_id"] != manifest["source"]["import_id"]:
                raise ValueError(
                    f"Release {release_id[:12]} uses a different source import"
                )
            spec = artifact_feature_spec(release)
            if spec.history_features and history is None:
                history = card_history(reader)
            X, y = window_features(reader, window, feature_spec=spec, history=history)
            if labels is None:
                labels = y.to_numpy()
            elif not np.array_equal(labels, y.to_numpy()):
                raise ValueError("Holdout rows differ between feature sets")
            values = adapter_for(release["recipe"]["estimator"]).scores(model, X)
            scores[release_id] = values
            entries.append(
                {
                    "release_id": release_id,
                    "role": "champion" if release_id == releases[-1] else "earlier",
                    "recipe_name": release.get("recipe_name"),
                    "as_of": release["as_of"],
                    "threshold": release["threshold"],
                    "evaluation": (release.get("evaluation") or {}).get("candidate"),
                    # Every release at the policy's budget, so their results compare.
                    "holdout": {
                        **metrics(labels, values, release["threshold"]),
                        **ranking_metrics(labels, values, review_budget=review_budget),
                    },
                }
            )
        snapshot_id = manifest["snapshot_id"]
    champion = entries[-1]
    budget = review_budget
    # Paired: each release's recall at the budget, and the champion's lead over it.
    draws = bootstrap_recall(labels, scores, review_budget=budget, seed=42)
    for entry in entries:
        entry["recall_interval"] = interval(draws[entry["release_id"]])
        if entry is not champion:
            entry["champion_lead_interval"] = interval(
                draws[champion["release_id"]] - draws[entry["release_id"]]
            )
    report = {
        "schema_version": 1,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot_id,
        "window": window,
        "rows": len(labels),
        "frauds": int(labels.sum()),
        "review_budget": budget,
        "champion": champion["release_id"],
        "releases": entries,
        "provenance": code_version(),
    }
    # Written aside and renamed into place, so a failed run leaves no partial report.
    stage = root / f".final-evaluation-{uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        report["plots"] = [
            write_comparison(
                stage,
                labels,
                [
                    (_label(entry), scores[entry["release_id"]], entry["threshold"])
                    for entry in entries
                ],
                review_budget=budget,
                title=population("Final holdout", window, labels),
            )
        ]
        write_json(stage / "report.json", report)
        (stage / "report.html").write_text(_html(stage, report))
        if output.exists():
            raise ValueError("The final holdout was scored by another run meanwhile")
        os.rename(stage, output)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    logger.info(
        "Final holdout %s to %s: champion %s recall at the budget %.4f, %d frauds",
        window[0][:10],
        window[1][:10],
        champion["release_id"][:12],
        champion["holdout"][f"recall_at_{alert_name(budget)}"],
        report["frauds"],
    )
    return {"report": str(output / "report.html"), **report}


def _label(entry) -> str:
    name = f"{entry['recipe_name'] or 'release'} {entry['release_id'][:8]}"
    return f"{name} (champion)" if entry["role"] == "champion" else name


def _interval(values) -> str:
    return f"{values[0]:.1%} to {values[1]:.1%}"


def _html(directory: Path, report: dict) -> str:
    budget = report["review_budget"]
    key = f"recall_at_{alert_name(budget)}"
    champion = report["releases"][-1]
    rows = []
    for entry in report["releases"]:
        holdout = entry["holdout"]
        expected = (entry["evaluation"] or {}).get(key)
        rows.append(
            [
                _label(entry),
                entry["as_of"],
                percent(holdout[key], 1),
                _interval(entry["recall_interval"]),
                percent(expected, 1, missing="–"),
                "–"
                if entry is champion
                else _interval(entry["champion_lead_interval"]),
                f"{holdout['average_precision']:.3f}",
                f"{percent(holdout['precision'], 1)} / {percent(holdout['recall'], 1)}",
                percent(holdout["alert_rate"], 2),
            ]
        )
    releases = table(
        [
            "Release",
            "Trained as of",
            f"Recall, top {budget:.1%}",
            "95% interval",
            "On its evaluation month",
            "Champion's lead",
            "AP",
            "Precision / recall at threshold",
            "Alert rate",
        ],
        rows,
        align="lnrrrrrrr",
    )
    figures = "".join(
        figure(
            (directory / name).read_bytes(),
            alt="Precision-recall and frauds caught by review volume for each release on the final holdout",
        )
        for name in report["plots"]
    )
    window = report["window"]
    body = section(
        "releases",
        "Releases on the holdout",
        releases
        + "<p>Recall at the review budget is the share of the month's frauds among its highest-scored transactions at the budget's volume, the quantity every promotion gate used. Intervals come from 1,000 paired row resamples; frauds cluster on cards, so they are too narrow, and overlapping intervals mean a tie. Precision, recall, and alert rate use each release's saved threshold. Earlier releases are shown for comparison only; nothing is chosen on this month.</p>",
    ) + section("charts", "Precision-recall and review volume", figures)
    return page(
        area="Evaluation",
        title=f"Final holdout — {window[0][:7]}",
        badge=badge("Scored once", "neutral"),
        intro=f"<p>Transactions from {window[0][:10]} up to {window[1][:10]}, read from snapshot {report['snapshot_id'][:12]} after every label in them was released. No training cycle fitted, selected, or evaluated on this month. The champion is the release confirmed serving after the last monthly update.</p>",
        tiles=tiles(
            [
                (
                    f"Champion recall, top {budget:.1%}",
                    percent(champion["holdout"][key], 1),
                    _interval(champion["recall_interval"]),
                ),
                (
                    "Champion AP",
                    f"{champion['holdout']['average_precision']:.3f}",
                    f"fraud rate {report['frauds'] / report['rows']:.2%}",
                ),
                ("Transactions", f"{report['rows']:,}", f"{report['frauds']:,} frauds"),
                (
                    "Alert rate at the saved threshold",
                    percent(champion["holdout"]["alert_rate"], 2),
                    f"budget {budget:.1%}",
                ),
            ]
        ),
        body=body,
        evidence=evidence(report),
    )
