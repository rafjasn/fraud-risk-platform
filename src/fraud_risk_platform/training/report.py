"""A training cycle's report: report.json, and the report.html page rendered from it."""

from pathlib import Path

from fraud_risk_platform.evaluation.metrics import ALERT_RATES, alert_name, review_rates
from fraud_risk_platform.persistence.files import write_json
from fraud_risk_platform.reporting.page import (
    badge,
    e,
    evidence,
    figure,
    page,
    percent,
    section,
    table,
    tiles,
)

# A recommendation's badge; keeping the incumbent is a normal outcome, not a warning.
RECOMMENDATIONS = {
    "promote": ("Promote", "good"),
    "initial_promotion": ("Initial promotion", "good"),
    "keep_current": ("Keep the current release", "neutral"),
    "no_initial_model": ("No initial model", "warning"),
    "failed": ("Failed", "critical"),
}
FIGURE_DESCRIPTIONS = {
    "validation.png": "Validation charts: precision-recall, frauds caught by review volume with the review budget, score distribution by class, precision and recall by threshold, confusion matrix",
    "comparison.png": "Recipe comparison charts: precision-recall and frauds caught by review volume for each recipe's representative model on validation",
    "learning_curves.png": "Learning curves: training and validation loss per epoch for each neural model, with the epoch of lowest validation loss marked",
    "trials.png": "Search trial charts: validation recall at the review budget by recipe with the selected trial marked, train versus validation average precision",
    "evaluation.png": "Evaluation charts: precision-recall and frauds caught by review volume for frozen models",
}
DEFINITIONS = (
    "<p>Average precision (AP) is sklearn.metrics.average_precision_score (a step-weighted PR summary, not trapezoidal area); random scores achieve the fraud rate. Recall at top x% is the share of frauds among the x% highest-scored transactions, a fixed review volume; tied boundary scores are split evenly. The review budget is the share of transactions analysts can review: each saved threshold is the lowest score that flags no more than that share of validation transactions, and a model qualifies when it then catches more fraud than reviewing a random sample of the same size. The later evaluation compares frozen models by recall at the budget volume; precision, recall, and alert rate at the saved threshold show how the threshold itself carries over to a new month. ROC-AUC is secondary: with rare fraud it stays high for models of little practical use. Train AP is computed on a fixed sample of up to 200,000 training rows; far above validation AP, it indicates overfitting. Inference timing covers batch scoring on the recorded validation rows, excluding data loading. Validation charts describe the selected model, or the highest-average-precision trial when nothing qualifies; evaluation charts describe frozen models and are not used to retune them.</p>"
    "<p>Limitations: synthetic source data and a synthetic fixed label delay. No statistical significance claim, confidence interval gate, economic cost model, or production readiness claim.</p>"
)


def write_report(directory: Path, report: dict):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "report.json", report)
    (directory / "report.html").write_text(render(directory, report))


def render(directory: Path, report: dict) -> str:
    # Failure reports can predate the resolved configuration.
    budget = report.get("resolved_config", {}).get("review_budget")
    rates = review_rates(budget) if budget else ALERT_RATES
    figures = {
        name: figure(
            (directory / name).read_bytes(), alt=FIGURE_DESCRIPTIONS.get(name, name)
        )
        for name in report.get("plots", [])
    }
    label, tone = RECOMMENDATIONS.get(
        report["recommendation"], (report["recommendation"], "warning")
    )
    release = report.get("release_id")
    body = (
        _validation_section(report, rates, budget, figures)
        + _selection_section(report, rates, budget, figures)
        + _evaluation_section(report, rates, budget, figures)
        + _trials_section(report, rates, budget, figures)
        + _features_section(report)
        + section("definitions", "Definitions and limitations", DEFINITIONS)
    )
    return page(
        area="Training",
        title=f"Training cycle — {report['as_of']}",
        badge=badge(label, tone),
        intro=(
            f"<p>{e(report['reason'])}</p>"
            "<p>Deployment is unchanged until an operator explicitly promotes a release. Thresholds and winner are selected only on validation. The later evaluation is a one-shot acceptance gate; there is no fallback to the runner-up.</p>"
            f'<p class="muted">Snapshot {e(str(report["snapshot_id"])[:12])} · '
            f"MLflow run {e(report['run_id'])} · "
            f"release {e(release[:12] if release else 'none')}</p>"
        ),
        tiles=_tiles(report, budget),
        body=body,
        evidence=evidence(report),
    )


def _recall_headers(rates, budget):
    return [
        f"Recall, top {rate * 100:g}%" + (" (budget)" if rate == budget else "")
        for rate in rates
    ]


def _recalls(values, rates):
    return [f"{values[f'recall_at_{alert_name(rate)}']:.1%}" for rate in rates]


def _tiles(report, budget):
    summary = report.get("validation_summary")
    if not summary or not budget:
        return ""
    key = f"recall_at_{alert_name(budget)}"
    review = f"top {budget * 100:g}% reviewed"
    winner = report.get("validation_winner")
    entry = report.get("comparison", {}).get(winner) if winner else None
    bounds = entry.get("recall_interval") if entry else None
    items = [
        (
            "Validation winner",
            winner or "None qualified",
            f"trial {entry['trial']}" if entry else "highest-AP trial summarized",
        ),
        (
            f"Validation recall, {review}",
            percent(summary[key], 1),
            f"95% interval {bounds[0]:.1%} to {bounds[1]:.1%}" if bounds else "",
        ),
    ]
    evaluation = report.get("evaluation") or {}
    candidate, incumbent = evaluation.get("candidate"), evaluation.get("incumbent")
    if candidate or incumbent:
        items.append(
            (
                f"Evaluation recall, {review}",
                percent(candidate[key], 1) if candidate else "Not scored",
                f"incumbent {percent(incumbent[key], 1)}"
                if incumbent
                else "no incumbent",
            )
        )
    items.append(
        (
            "Validation average precision",
            f"{summary['average_precision']:.3f}",
            f"train {summary['train_average_precision']:.3f}",
        )
    )
    return tiles(items)


def _validation_section(report, rates, budget, figures):
    summary = report.get("validation_summary")
    if not summary:
        return ""
    body = (
        f"<p>Source: {e(summary['source'].replace('_', ' '))}. Validation fraud rate: {summary['prevalence']:.3%}, which is the average precision of random scores.</p>"
        + table(
            ["Validation AP", "Train AP", "ROC-AUC", *_recall_headers(rates, budget)],
            [
                [
                    f"{summary['average_precision']:.4f}",
                    f"{summary['train_average_precision']:.4f}",
                    f"{summary['roc_auc']:.3f}",
                    *_recalls(summary, rates),
                ]
            ],
            align="r" * (3 + len(rates)),
        )
        + figures.get("validation.png", "")
    )
    return section("validation", "Validation summary", body)


def _selection_section(report, rates, budget, figures):
    comparison = report.get("comparison")
    if not comparison or not budget:
        return ""
    winner = report.get("validation_winner")
    key = f"recall_at_{alert_name(budget)}"
    others = [rate for rate in rates if rate != budget]
    rows = []
    for name, entry in comparison.items():
        metric = entry["validation"] or {}
        ranking = entry["validation_ranking"]
        bounds = entry.get("recall_interval")
        lead = entry.get("winner_lead")
        rows.append(
            [
                name
                + (
                    " (selected)"
                    if name == winner
                    else ""
                    if entry["qualified"]
                    else " (not qualified)"
                ),
                entry["trial"],
                percent(metric.get("recall"), 1, missing="no threshold"),
                f"{ranking[key]:.1%}"
                + (f" ({bounds[0]:.1%}–{bounds[1]:.1%})" if bounds else ""),
                f"{lead['estimate'] * 100:+.1f} ({lead['interval'][0] * 100:+.1f} to {lead['interval'][1] * 100:+.1f})"
                if lead
                else "",
                percent(metric.get("precision"), 1, missing="no threshold"),
                percent(metric.get("alert_rate"), missing="no threshold"),
                f"{ranking['average_precision']:.4f}",
                f"{entry['train_average_precision']:.4f}",
                *_recalls(ranking, others),
            ]
        )
    body = (
        "<p>Each recipe's selected trial, or its highest-recall trial when none qualified. The winner has the highest recall at its saved threshold among qualified recipes. Intervals come from 1,000 resamples of validation transactions shared by all recipes, so the winner's lead over each recipe is a paired comparison. Frauds cluster on cards, so these row-level intervals are, if anything, too narrow: a lead whose interval includes zero is not a measured difference.</p>"
        + table(
            [
                "Recipe",
                "Trial",
                "Recall at threshold",
                f"Recall, top {budget * 100:g}% (95% interval)",
                "Winner's lead, points (95% interval)",
                "Precision at threshold",
                "Alert rate",
                "Validation AP",
                "Train AP",
                *_recall_headers(others, budget),
            ],
            rows,
            align="l" + "r" * (8 + len(others)),
        )
        + figures.get("comparison.png", "")
        + figures.get("learning_curves.png", "")
    )
    return section("selection", "Validation selection", body)


def _evaluation_section(report, rates, budget, figures):
    if "evaluation" not in report:
        return ""
    rows = [
        [
            name,
            percent(metric["precision"]),
            percent(metric["recall"]),
            percent(metric["alert_rate"]),
            f"{metric['average_precision']:.4f}",
            *_recalls(metric, rates),
            f"{metric['true_positives']:,}",
            f"{metric['false_positives']:,}",
            f"{metric['false_negatives']:,}",
        ]
        for name, metric in report["evaluation"].items()
        if metric is not None
    ]
    if rows:
        body = table(
            [
                "Frozen model",
                "Precision at threshold",
                "Recall at threshold",
                "Alert rate at threshold",
                "Average precision",
                *_recall_headers(rates, budget),
                "True positives",
                "False positives",
                "Missed frauds",
            ],
            rows,
            align="l" + "r" * (7 + len(rates)),
        )
    else:
        body = "<p>The evaluation month is scored only for a frozen candidate or the incumbent; nothing was scored in this cycle.</p>"
    body += figures.get("evaluation.png", "")
    refit = report.get("refit")
    if refit:
        body += f"<p>Full-history refit: {refit['training_rows']:,} training rows; qualification on validation: {e(refit['qualified'])}; saved threshold: {e(refit['threshold'])}.</p>"
    return section("evaluation", "Later evaluation", body)


def _parameters(values):
    return ", ".join(
        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in values.items()
    )


def _trials_section(report, rates, budget, figures):
    searches = report.get("searches")
    if not searches:
        return ""
    executed = report.get("resolved_config", {}).get("executed_recipes", {})
    rows = []
    for name, search in searches.items():
        searched = executed.get(name, {}).get("search_space") or {}
        for trial in search["trials"]:
            ranking = trial["validation_ranking"]
            parameters = trial["recipe"]["parameters"]
            shown = {k: parameters[k] for k in searched if k in parameters}
            rows.append(
                [
                    name,
                    trial["number"],
                    _parameters(shown or parameters),
                    "yes" if trial["qualified"] else "no",
                    f"{ranking['average_precision']:.4f}",
                    f"{trial['train_average_precision']:.4f}",
                    *_recalls(ranking, rates),
                    f"{trial['training_seconds']:.1f}",
                ]
            )
    # The chart summarizes every trial; the full log stays one click away.
    body = (
        figures.get("trials.png", "")
        + f"<details><summary>All {len(rows)} trials, with their parameters</summary>"
        + table(
            [
                "Recipe",
                "Trial",
                "Parameters",
                "Qualified",
                "Validation AP",
                "Train AP",
                *_recall_headers(rates, budget),
                "Fit seconds",
            ],
            rows,
            align="nrll" + "r" * (3 + len(rates)),
        )
        + '<p class="muted">Parameters lists the searched settings; a recipe without a search shows its fixed settings.</p></details>'
    )
    return section("trials", "Search trials", body)


def _features_section(report):
    executed = report.get("resolved_config", {}).get("executed_recipes", {})
    if not executed:
        return ""
    rows = []
    for name, recipe in executed.items():
        spec = recipe.get("feature_spec") or {}
        rows.append(
            [
                name,
                recipe.get("feature_set", "basic-v1"),
                ", ".join(spec.get("numeric", [])),
                ", ".join(spec.get("categorical", [])),
            ]
        )
    return section(
        "features",
        "Input features",
        table(["Recipe", "Feature set", "Numeric", "Categorical"], rows),
    )
