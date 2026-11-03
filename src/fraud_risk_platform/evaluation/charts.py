"""Report charts; every title names its partition, period, row/fraud counts, and model."""

import math
from datetime import datetime, timedelta

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    LogFormatter,
    LogitLocator,
    MaxNLocator,
    NullFormatter,
)
from sklearn.metrics import confusion_matrix, precision_recall_curve

from fraud_risk_platform.evaluation.metrics import (
    alert_name,
    ranking_metrics,
    review_rates,
)
from fraud_risk_platform.reporting.palette import (
    CATEGORICAL,
    CRITICAL,
    FRAUD,
    GRID,
    INK,
    LEGITIMATE,
    MUTED,
    SECONDARY,
    SERIOUS,
    SURFACE,
)
from fraud_risk_platform.reporting.style import (
    HAIRLINE,
    LINE,
    MARKER,
    add_legend,
    date_axis,
    new_figure,
    png,
    point,
    reference,
    style_axis,
)

# Two-series charts take the first two categorical slots.
SERIES = CATEGORICAL[:2]
MAX_REVIEW = 0.02


def population(name, window, labels):
    end = (datetime.fromisoformat(window[1]) - timedelta(days=1)).date()
    return (
        f"{name}, {window[0][:10]} to {end}: "
        f"{len(labels):,} rows, {int(np.sum(labels)):,} frauds"
    )


def _logit(values):
    return np.clip(values, 1e-9, 1 - 1e-9)


def _logit_axis(axis):
    axis.set_xscale("logit")
    # Wide "1 - 10^-k" labels collide at the default density.
    axis.xaxis.set_major_locator(LogitLocator(nbins=6))
    axis.xaxis.set_minor_formatter(NullFormatter())


def _precision_recall(axis, labels, models, *, legend=True):
    prevalence = float(np.mean(labels))
    for label, scores, threshold, color in models:
        precision, recall, _ = precision_recall_curve(labels, scores)
        # Drop sklearn's appended (recall 0, precision 1) end point.
        precision, recall = precision[:-1], recall[:-1]
        keep = np.unique(np.linspace(0, len(precision) - 1, 1000, dtype=int))
        axis.plot(
            recall[keep],
            precision[keep],
            color=color,
            linewidth=LINE,
            solid_capstyle="round",
            label=label,
        )
        if threshold is not None:
            flagged = scores >= threshold
            caught = int(labels[flagged].sum())
            if flagged.any():
                point(axis, caught / labels.sum(), caught / flagged.sum(), color)
    # Left end: precision-recall curves start near the top, so the baseline is clear there.
    reference(
        axis,
        ([0, 1], [prevalence] * 2),
        f"Random scores ({prevalence:.2%})",
        at="start",
    )
    marked = any(threshold is not None for _, _, threshold, _ in models)
    style_axis(
        axis,
        "Precision–recall",
        "Recall (share of frauds flagged)"
        + (" · dot: saved threshold" if marked else ""),
        "Precision",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1.02))
    if legend and len(models) > 1:
        add_legend(axis)


def _review(axis, labels, models, review_budget, *, legend=True):
    prevalence = float(np.mean(labels))
    limit = max(MAX_REVIEW, 2 * review_budget)
    count = max(1, math.ceil(limit * len(labels)))
    share = np.arange(1, count + 1) / len(labels) * 100
    for label, scores, _, color in models:
        order = np.argsort(-scores, kind="stable")
        caught = np.cumsum(labels[order][:count]) / labels.sum() * 100
        axis.plot(share, caught, color=color, linewidth=LINE, label=label)
        # Markers match the tie-aware values in the report tables.
        values = ranking_metrics(labels, scores, review_budget=review_budget)
        for rate in review_rates(review_budget):
            point(
                axis, rate * 100, values[f"recall_at_{alert_name(rate)}"] * 100, color
            )
    reviewed = np.linspace(0, limit * 100, 200)
    reference(
        axis,
        (reviewed, np.minimum(100, reviewed / (prevalence * 100) * 100)),
        "Perfect ranking",
    )
    reference(axis, (reviewed, reviewed), "Random review")
    budget = review_budget * 100
    # Bottom of the line: only a near-random model passes that low.
    reference(
        axis, ([budget, budget], [0, 100]), f"Review budget {budget:g}%", at="start"
    )
    dots = ", ".join(f"{rate * 100:g}%" for rate in review_rates(review_budget))
    style_axis(
        axis,
        "Frauds caught by review volume",
        f"Share of transactions reviewed, highest scores first (%) · dots at {dots}",
        "Share of frauds caught (%)",
    )
    axis.set(xlim=(0, limit * 100), ylim=(0, 102))
    if legend and len(models) > 1:
        add_legend(axis)


def _score_distribution(axis, labels, scores, threshold):
    # Logit bins resolve both near-0 (unweighted) and near-1 (balanced) scores.
    clipped = _logit(scores)
    logits = np.log(clipped / (1 - clipped))
    low, high = logits.min(), max(logits.max(), logits.min() + 1)
    bins = 1 / (1 + np.exp(-np.linspace(low, high, 41)))
    for value, name, color in ((0, "Legitimate", LEGITIMATE), (1, "Fraud", FRAUD)):
        members = clipped[labels == value]
        counts, _ = np.histogram(members, bins=bins)
        axis.stairs(
            counts / len(members),
            bins,
            color=color,
            linewidth=LINE,
            label=f"{name} ({len(members):,})",
        )
    if threshold is not None:
        axis.axvline(
            _logit(threshold), color=MUTED, linewidth=HAIRLINE * 1.5, linestyle="--"
        )
    _logit_axis(axis)
    style_axis(
        axis,
        "Score distribution by class",
        "Fraud score (logit scale)"
        + (" · dashed: saved threshold" if threshold is not None else ""),
        "Share of class",
    )
    add_legend(axis)


def _threshold_curves(axis, labels, scores, threshold):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    keep = np.unique(np.linspace(0, len(thresholds) - 1, 1000, dtype=int))
    cuts = _logit(thresholds[keep])
    for values, name, color in (
        (precision, "Precision", SERIES[0]),
        (recall, "Recall", SERIES[1]),
    ):
        axis.plot(cuts, values[keep], color=color, linewidth=LINE, label=name)
    if threshold is not None:
        axis.axvline(
            _logit(threshold), color=MUTED, linewidth=HAIRLINE * 1.5, linestyle="--"
        )
    _logit_axis(axis)
    style_axis(
        axis,
        "Precision and recall by threshold",
        "Threshold (logit scale)"
        + (" · dashed: saved threshold" if threshold is not None else ""),
        "Value",
    )
    axis.set_ylim(0, 1.02)
    add_legend(axis)


def _confusion(axis, labels, scores, threshold, review_budget):
    axis.set_axis_off()
    if threshold is None:
        axis.set_title("Confusion matrix", loc="left", color=INK, fontsize=11, pad=22)
        axis.text(
            0,
            0.5,
            f"Tied top scores exceed the {review_budget * 100:g}% review budget,\n"
            "so no threshold fits and nothing is flagged.",
            color=SECONDARY,
            fontsize=10,
            va="center",
        )
        return
    (tn, fp), (fn, tp) = confusion_matrix(labels, scores >= threshold, labels=[0, 1])
    table = axis.table(
        cellText=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
        rowLabels=["Legitimate", "Fraud"],
        colLabels=["Not flagged", "Flagged"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2.2)
    for cell in table.get_celld().values():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(HAIRLINE)
        cell.get_text().set_color(INK)
    axis.set_title(
        f"Confusion matrix at threshold {threshold:.4g}",
        loc="left",
        color=INK,
        fontsize=11,
        pad=22,
    )


def _save(figure, directory, filename, title):
    figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    (directory / filename).write_bytes(png(figure))
    return filename


def write_validation(directory, labels, scores, *, threshold, review_budget, title):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    figure = new_figure(14, 8)
    axes = figure.subplot_mosaic("ABC;DDE")
    model = [("Model", scores, threshold, SERIES[0])]
    _precision_recall(axes["A"], labels, model)
    _review(axes["B"], labels, model, review_budget)
    _score_distribution(axes["C"], labels, scores, threshold)
    _threshold_curves(axes["D"], labels, scores, threshold)
    _confusion(axes["E"], labels, scores, threshold, review_budget)
    return _save(figure, directory, "validation.png", title)


def _trial_point(axis, x, y, face):
    axis.plot(
        [x],
        [y],
        marker="o",
        markersize=MARKER,
        markerfacecolor=face,
        markeredgecolor=SERIES[0],
        markeredgewidth=LINE,
        linestyle="none",
    )


def _selected_ring(axis, x, y):
    # A ring, not a label, marks the winner: neighboring trials often score alike.
    axis.plot(
        [x],
        [y],
        marker="o",
        markersize=MARKER * 2,
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=HAIRLINE * 2,
        linestyle="none",
    )


def write_trials(directory, searches, *, review_budget, title):
    """The selection quantity per trial, and each trial's overfitting check."""
    figure = new_figure(14, 4.5)
    left, right = figure.subplots(1, 2)
    key = f"recall_at_{alert_name(review_budget)}"
    names = list(searches)
    highest = 0.0
    for position, name in enumerate(names):
        trials = searches[name]["trials"]
        best = searches[name].get("best")
        offsets = np.linspace(-0.2, 0.2, len(trials)) if len(trials) > 1 else [0.0]
        for offset, trial in zip(offsets, trials):
            validation = trial["validation_average_precision"]
            highest = max(highest, validation, trial["train_average_precision"])
            # Shape fill, not hue, distinguishes qualified trials.
            face = SERIES[0] if trial["qualified"] else SURFACE
            points = (
                (left, position + offset, trial["validation_ranking"][key] * 100),
                (right, trial["train_average_precision"], validation),
            )
            for axis, x, y in points:
                _trial_point(axis, x, y, face)
                if best is not None and trial["number"] == best["number"]:
                    _selected_ring(axis, x, y)
    budget = review_budget * 100
    reference(left, ([-0.5, len(names) - 0.5], [budget] * 2), "Random review")
    style_axis(
        left,
        f"Validation recall at the {budget:g}% review budget by trial",
        "Recipe",
        "Share of frauds caught (%)",
    )
    left.set_xticks(range(len(names)), names)
    left.set(xlim=(-0.5, len(names) - 0.5), ylim=(0, 102))
    limit = highest * 1.1 or 1
    reference(
        right,
        (np.linspace(0, limit, 6), np.linspace(0, limit, 6)),
        "Train AP = validation AP",
        at="above",
    )
    style_axis(
        right,
        "Train vs validation average precision",
        "Train AP · far below the dashed line: fits training data much better (overfitting)",
        "Validation AP",
    )
    right.set(xlim=(0, limit), ylim=(0, limit))
    marker = {"marker": "o", "linestyle": "none", "markersize": MARKER}
    handles = [
        Line2D([], [], color=SERIES[0], **marker),
        Line2D(
            [],
            [],
            markerfacecolor=SURFACE,
            markeredgecolor=SERIES[0],
            markeredgewidth=LINE,
            **marker,
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=MARKER * 2,
            markerfacecolor="none",
            markeredgecolor=INK,
            markeredgewidth=HAIRLINE * 2,
        ),
    ]
    add_legend(left, handles=handles, labels=["Qualified", "Not qualified", "Selected"])
    return _save(figure, directory, "trials.png", title)


def write_comparison(directory, labels, models, *, review_budget, title):
    """Each recipe's representative model on validation, colored by recipe order.

    models: (label, scores, threshold) tuples in the cycle's configured recipe order,
    so a recipe keeps its color across cycles.
    """
    if len(models) > len(CATEGORICAL):
        raise ValueError("Too many recipes to compare in one chart")
    labels = np.asarray(labels, dtype=np.int64)
    models = [
        (label, np.asarray(scores, dtype=float), threshold, color)
        for (label, scores, threshold), color in zip(models, CATEGORICAL)
    ]
    figure = new_figure(14, 4.5)
    left, right = figure.subplots(1, 2)
    _precision_recall(left, labels, models, legend=False)
    _review(right, labels, models, review_budget, legend=False)
    # One shared legend below both panels; long recipe labels squeeze per-axis legends.
    figure.legend(
        *left.get_legend_handles_labels(),
        loc="outside lower center",
        ncols=min(len(models), 3),
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY,
        handlelength=1.5,
    )
    return _save(figure, directory, "comparison.png", title)


def write_learning_curves(directory, curves, *, title):
    """Per-epoch loss for each neural representative, one panel per model.

    curves: (label, {"train_loss": [...], "validation_loss": [...]}) pairs.
    """
    # No narrower than the other charts, so its text matches theirs and its title fits.
    figure = new_figure(max(14, 7 * len(curves)), 4.5)
    axes = figure.subplots(1, len(curves), squeeze=False)[0]
    for axis, (label, curve) in zip(axes, curves):
        for key, name, color in (
            ("train_loss", "Training", SERIES[0]),
            ("validation_loss", "Validation", SERIES[1]),
        ):
            values = curve.get(key) or []
            epochs = np.arange(1, len(values) + 1)
            axis.plot(
                epochs,
                values,
                color=color,
                linewidth=LINE,
                marker="o",
                markersize=MARKER * 0.6,
                label=name,
            )
        validation = curve.get("validation_loss") or []
        if validation:
            best = int(np.argmin(validation)) + 1
            axis.axvline(best, color=MUTED, linewidth=HAIRLINE * 1.5, linestyle="--")
        style_axis(
            axis,
            f"Learning curve · {label}",
            "Epoch"
            + (
                f" · dashed: lowest validation loss (epoch {best})"
                if validation
                else ""
            ),
            "Fraud-weighted cross-entropy",
        )
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        add_legend(axis)
    return _save(figure, directory, "learning_curves.png", title)


def write_monitoring(directory, daily, weekly, *, review_budget, title):
    """Daily review volume of replayed traffic and weekly recall on mature outcomes.

    daily: day, alert_rate. weekly: week, recall, expected (evaluation-month recall).
    """
    figure = new_figure(14, 4.5)
    left, right = figure.subplots(1, 2)
    left.plot(
        daily.day,
        daily.alert_rate * 100,
        color=SERIES[0],
        linewidth=LINE,
        label="Flagged per day",
    )
    budget = review_budget * 100
    # A legend, not a direct label: daily rates hover around the budget line.
    left.axhline(
        budget,
        color=MUTED,
        linewidth=HAIRLINE * 1.5,
        linestyle="--",
        label=f"Review budget ({budget:g}%)",
    )
    add_legend(left)
    style_axis(
        left,
        "Share of transactions flagged per day",
        "Transaction date",
        "Flagged (%)",
    )
    left.set_ylim(0, max(budget, float(daily.alert_rate.max()) * 100) * 1.3)
    date_axis(left)
    if len(weekly):
        right.plot(
            weekly.week,
            weekly.recall * 100,
            color=SERIES[0],
            linewidth=LINE,
            marker="o",
            markersize=MARKER * 0.6,
            label="Replayed traffic",
        )
        if weekly.expected.notna().any():
            right.plot(
                weekly.week,
                weekly.expected * 100,
                color=MUTED,
                linewidth=HAIRLINE * 1.5,
                linestyle="--",
                label="Release's evaluation month",
            )
        add_legend(right)
    else:
        right.text(
            0.02,
            0.5,
            "No mature frauds yet.",
            color=SECONDARY,
            fontsize=10,
            transform=right.transAxes,
        )
    style_axis(
        right,
        "Recall at the saved threshold, mature outcomes by week",
        "Week starting",
        "Frauds flagged (%)",
    )
    right.set_ylim(0, 102)
    date_axis(right)
    return _save(figure, directory, "monitoring.png", title)


def write_evaluation(
    directory, labels, *, candidate=None, incumbent=None, review_budget, title
):
    """Overlay frozen models; color follows the role, not which models were scored."""
    labels = np.asarray(labels, dtype=np.int64)
    models = []
    for entry, color in ((candidate, SERIES[0]), (incumbent, SERIES[1])):
        if entry is not None:
            label, scores, threshold = entry
            models.append((label, np.asarray(scores, dtype=float), threshold, color))
    figure = new_figure(14, 4.5)
    left, right = figure.subplots(1, 2)
    _precision_recall(left, labels, models)
    _review(right, labels, models, review_budget)
    if len(models) == 1:
        title = f"{title} · {models[0][0]}"
    return _save(figure, directory, "evaluation.png", title)


def write_load(directory, results, latency, steps, *, title):
    """Latency of every load-test request over the run, and server time by step.

    results: dicts with offset and seconds (since the run started) and status (None
    without a response). latency: client percentiles in ms for successful requests.
    steps: mean server milliseconds per request by step, including "other".
    """
    figure = new_figure(14, 4.5)
    left, right = figure.subplots(1, 2, width_ratios=(2, 1))
    # Request outcomes are states, so they wear the reserved status colors plus a
    # marker shape and a counted legend label; success keeps the first series slot.
    classes = (
        ("Succeeded", lambda s: s is not None and s < 400, SERIES[0], "o"),
        ("Rejected, 4xx", lambda s: s is not None and 400 <= s < 500, SERIOUS, "^"),
        ("Failed, 5xx or no response", lambda s: s is None or s >= 500, CRITICAL, "x"),
    )
    for label, member, color, marker in classes:
        points = [r for r in results if member(r["status"])]
        if points:
            left.plot(
                [r["offset"] for r in points],
                [r["seconds"] * 1000 for r in points],
                linestyle="none",
                marker=marker,
                markersize=MARKER * 0.6,
                color=color,
                alpha=0.5 if color == SERIES[0] else 1,
                label=f"{label} ({len(points):,})",
            )
    # Percentile labels sit outside the plot so dense points never cover them; p99
    # above its line and p95 below its line, so close values do not collide.
    for name, va in (("p50", "center"), ("p95", "top"), ("p99", "bottom")):
        if latency:
            left.axhline(
                latency[name], color=MUTED, linewidth=HAIRLINE * 1.5, linestyle="--"
            )
            left.annotate(
                f"{name} {latency[name]:,.1f} ms",
                (1, latency[name]),
                xycoords=left.get_yaxis_transform(),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va=va,
                color=SECONDARY,
                fontsize=8,
                annotation_clip=False,
            )
    left.set_yscale("log")
    # Plain milliseconds; minor ticks are labeled only when the range is narrow.
    left.yaxis.set_major_formatter(LogFormatter())
    left.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(1, 0.5)))
    add_legend(left)
    style_axis(
        left,
        "Client latency per request",
        "Seconds since the run started",
        "Latency (ms, log scale)",
    )
    left.tick_params(which="minor", colors=SECONDARY, labelsize=8, width=HAIRLINE)
    names = list(steps)
    right.barh(
        names,
        [steps[name] for name in names],
        color=[MUTED if name == "other" else SERIES[0] for name in names],
        height=0.6,
    )
    for row, name in enumerate(names):
        right.annotate(
            f"{steps[name]:,.1f} ms",
            (steps[name], row),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY,
            fontsize=8,
        )
    right.invert_yaxis()
    right.set_xlim(0, max(steps.values(), default=1) * 1.3 or 1)
    style_axis(
        right,
        "Server time per request by step (mean)",
        "Milliseconds",
        "",
    )
    right.grid(False, axis="y")
    return _save(figure, directory, "latency.png", title)
