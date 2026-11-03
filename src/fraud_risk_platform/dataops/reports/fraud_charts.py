"""Class exploration plots with explicit populations and support counts."""

import numpy as np

from fraud_risk_platform.dataops.reports.fraud import category_groups
from fraud_risk_platform.reporting.palette import (
    AXIS,
    FRAUD,
    LEGITIMATE,
    MUTED,
    SECONDARY,
    SERIES,
)
from fraud_risk_platform.reporting.style import (
    HAIRLINE,
    LINE,
    add_legend,
    label_color,
    new_figure,
    sequential_colormap,
    style_axis,
)

CLASS_STYLES = {
    "legitimate": ("Legitimate", LEGITIMATE),
    "fraud": ("Fraud", FRAUD),
}


def amount_distribution(stats):
    fig = new_figure(12, 4.5)
    axis = fig.subplots()
    x = np.arange(len(stats["amount_bin_labels"]))
    for offset, (name, (label, color)) in zip((-0.2, 0.2), CLASS_STYLES.items()):
        group = stats["classes"][name]
        if group["rows"]:
            axis.bar(
                x + offset,
                100 * np.asarray(group["amount_bins"]) / group["rows"],
                width=0.38,
                color=color,
                label=f"{label} (n={group['rows']:,})",
            )
    axis.set_xticks(x, stats["amount_bin_labels"], rotation=30, ha="right")
    style_axis(
        axis,
        "Amount distribution by class",
        "Signed amount bands (USD; lower bound included, upper excluded)",
        "Percent within each class",
        grid="y",
    )
    add_legend(axis)
    return fig


def amount_hour_scatter(stats):
    fig = new_figure(12, 5)
    axis = fig.subplots()
    for name, (label, color) in CLASS_STYLES.items():
        points = np.asarray(stats["scatter"]["points"][name])
        if not len(points):
            continue
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=9 if name == "legitimate" else 18,
            alpha=0.15 if name == "legitimate" else 0.65,
            color=color,
            linewidths=0,
            label=f"{label}: {len(points):,} shown / {stats['classes'][name]['rows']:,}",
        )
    axis.set_yscale("symlog", linthresh=10)
    axis.set(xlim=(0, 24), xticks=range(0, 25, 2))
    style_axis(
        axis,
        "Transaction amount versus time of day",
        "Source time of day (UTC convention; fractional hours)",
        "Amount (USD; symmetric log scale, linear between -10 and 10)",
    )
    add_legend(axis)
    # Faint points would make faint legend keys; the keys are solid and larger.
    for key in axis.get_legend().legend_handles:
        key.set_alpha(1)
        key.set_sizes([30])
    return fig


def amount_hour_heatmap(stats):
    frauds = np.asarray(stats["classes"]["fraud"]["amount_hour"])
    totals = frauds + np.asarray(stats["classes"]["legitimate"]["amount_hour"])
    rates = np.divide(
        100.0 * frauds, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0
    )
    supported = totals >= stats["minimum_rate_rows"]
    fig = new_figure(14, 5.5)
    axis = fig.subplots()
    maximum = float(rates[supported].max()) if supported.any() else 0
    plot = axis.imshow(
        np.ma.array(rates, mask=~supported),
        cmap=sequential_colormap(),
        vmin=0,
        vmax=maximum or 1,
        aspect="auto",
    )
    for band in range(len(totals)):
        for hour in range(24):
            n, f = totals[band, hour], frauds[band, hour]
            label = (
                f"{f}/{n}" + ("*" if n and not supported[band, hour] else "")
                if n
                else "—"
            )
            # Labels inside a filled cell take ink or white, whichever reads on it.
            color = (
                label_color(plot.cmap(plot.norm(rates[band, hour])))
                if supported[band, hour]
                else SECONDARY
            )
            axis.text(
                hour, band, label, ha="center", va="center", fontsize=6, color=color
            )
    axis.set(
        xticks=range(24),
        yticks=range(len(stats["amount_bin_labels"])),
        yticklabels=stats["amount_bin_labels"],
    )
    style_axis(
        axis,
        "Fraud rate by amount band and hour — cell labels: frauds / labeled transactions",
        "Source hour (UTC convention)",
        "Signed amount band (USD)",
        grid=None,
    )
    scale = fig.colorbar(plot, ax=axis)
    scale.set_label("Fraud / labeled transactions (%)", color=SECONDARY, fontsize=9)
    scale.ax.tick_params(colors=SECONDARY, labelsize=8, width=HAIRLINE)
    scale.outline.set_edgecolor(AXIS)
    scale.outline.set_linewidth(HAIRLINE)
    return fig


def category_rates(stats):
    fig = new_figure(14, 6.5)
    total = sum(group["rows"] for group in stats["classes"].values())
    baseline = 100 * stats["classes"]["fraud"]["rows"] / total
    for axis, key, limit, title in zip(
        fig.subplots(1, 2),
        ("payment_types", "mcc"),
        (None, 12),
        ("Payment channel", "Merchant category (12 largest by volume + Other)"),
    ):
        groups = category_groups(stats[key], limit)
        labels = []
        for position, (name, counts) in enumerate(groups):
            n, f = counts["rows"], counts["frauds"]
            p, z = f / n, 1.96
            middle = (p + z * z / (2 * n)) / (1 + z * z / n)
            radius = (
                z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            )
            # Too little support to read a rate: a hollow muted marker.
            low = n < stats["minimum_rate_rows"]
            color = MUTED if low else SERIES
            axis.hlines(
                position,
                100 * max(0, middle - radius),
                100 * min(1, middle + radius),
                color=color,
                linewidth=LINE,
            )
            axis.scatter(
                [100 * p],
                [position],
                facecolors="none" if low else color,
                edgecolors=color,
                linewidths=LINE,
                s=36,
                zorder=3,
            )
            labels.append(f"{name}{'*' if low else ''}   ({f:,} / {n:,})")
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.axvline(
            baseline,
            color=MUTED,
            linestyle="--",
            linewidth=HAIRLINE * 1.5,
            label=f"Training baseline: {baseline:.3f}%",
        )
        axis.set_xlim(0, None)
        style_axis(axis, title, "Fraud / labeled transactions (%)", grid="x")
        add_legend(axis)
    return fig


def fraud_figures(stats):
    """Yield each figure with its caption; callers embed it."""
    yield (
        amount_distribution(stats),
        (
            "All labeled training rows; each present class sums to 100% separately. "
            "These are class distributions, not the probability of fraud within an amount band."
        ),
    )
    yield (
        amount_hour_scatter(stats),
        (
            f"{stats['scatter']['method']}. Limits: "
            f"{stats['scatter']['limits']['legitimate']:,} legitimate and "
            f"{stats['scatter']['limits']['fraud']:,} fraudulent transactions. "
            "Class-specific display sampling changes the visible class ratio; it does not represent fraud prevalence. "
            "Negative and zero amounts are retained."
        ),
    )
    yield (
        amount_hour_heatmap(stats),
        (
            f"Exact counts from all labeled training rows, independent of scatter sampling. "
            f"Gray cells with * have fewer than {stats['minimum_rate_rows']:,} labeled transactions; "
            "their rates are hidden. A dash means no observations; supported zero-fraud cells have a 0% rate. "
            "This display cutoff is not a statistical-confidence guarantee."
        ),
    )
    yield (
        category_rates(stats),
        (
            "Labels show frauds / labeled transactions. Counts use all labeled training rows; remaining MCCs are combined as Other. "
            f"Hollow gray markers and * indicate fewer than {stats['minimum_rate_rows']:,} observations. "
            "Bars show descriptive 95% Wilson intervals assuming independent transactions; correlated fraud episodes can increase uncertainty."
        ),
    )
