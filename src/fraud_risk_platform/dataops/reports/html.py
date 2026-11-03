"""Offline HTML data reports with explicit populations, denominators, and provenance."""

from datetime import datetime

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fraud_risk_platform.dataops.reports.fraud_charts import fraud_figures
from fraud_risk_platform.dataops.reports.summary import AMOUNT_LABELS, calendar_series
from fraud_risk_platform.reporting.page import (
    badge,
    e,
    evidence,
    figure,
    money,
    number,
    page,
    percent,
    section,
    share,
    table,
    tiles,
)
from fraud_risk_platform.reporting.palette import AXIS, MUTED, ORDINAL, SERIES
from fraud_risk_platform.reporting.style import (
    LINE,
    add_legend,
    count_axis,
    date_axis,
    new_figure,
    png,
    style_axis,
)

# Badge tone of a delivery or snapshot status; any other status is in progress.
TONES = {"accepted": "good", "rejected": "critical", "failed": "critical"}


def charts(stats, *, start=None, end=None, population="Transactions"):
    if not stats or not stats.get("rows"):
        return "<p>No valid transactions were available for these charts.</p>"
    dates, counts = calendar_series(stats, start, end)
    a = np.asarray(counts)
    fig = new_figure(14, 8.5)
    (volume, amounts, hours), (absent, coverage, rate) = fig.subplots(2, 3)
    volume.plot(dates, a[:, 0], color=SERIES, linewidth=LINE)
    style_axis(volume, "Transactions by calendar day", ylabel="Transactions")
    count_axis(volume)
    date_axis(volume)
    bins = range(len(AMOUNT_LABELS))
    amounts.bar(bins, stats["amount_bins"], width=0.6, color=SERIES)
    amounts.set_xticks(bins, AMOUNT_LABELS, rotation=55, ha="right")
    style_axis(
        amounts, "Signed amount distribution (USD)", ylabel="Transactions", grid="y"
    )
    count_axis(amounts)
    hours.bar(range(24), stats["by_hour"], width=0.6, color=SERIES)
    style_axis(
        hours,
        "Activity by source hour",
        "Hour (UTC convention)",
        "Transactions",
        grid="y",
    )
    count_axis(hours)
    fields = ["merchant_state", "postal_code"]
    absent.bar(
        fields,
        [100 * stats.get("missing", {}).get(f, 0) / stats["rows"] for f in fields],
        width=0.3,
        color=SERIES,
    )
    style_axis(
        absent,
        "Absent optional location fields",
        ylabel="Percent of transactions",
        grid="y",
    )
    absent.set_ylim(0, 100)
    # Known labels are the measure; pending ones recede as the gap up to all rows.
    known, total = a[:, 1], a[:, 0]
    coverage.fill_between(dates, known, color=SERIES, alpha=0.12, linewidth=0)
    coverage.fill_between(dates, known, total, color=AXIS, linewidth=0)
    coverage.plot(dates, known, color=SERIES, linewidth=LINE)
    style_axis(coverage, "Label coverage by event date", ylabel="Transactions")
    count_axis(coverage)
    add_legend(
        coverage,
        handles=[Line2D([], [], color=SERIES, linewidth=LINE), Patch(color=AXIS)],
        labels=["Known", "Pending"],
    )
    date_axis(coverage)
    # Aggregate in non-overlapping seven-day groups. Partly mature dates are excluded.
    rate_dates, rates, lower, upper = [], [], [], []
    for offset in range(0, len(dates), 7):
        block = a[offset : offset + 7]
        mature = block[block[:, 0] == block[:, 1]]
        n, frauds = mature[:, 1].sum(), mature[:, 2].sum()
        rate_dates.append(dates[offset])
        if not n:
            rates.append(np.nan)
            lower.append(np.nan)
            upper.append(np.nan)
            continue
        p, z = frauds / n, 1.96
        middle = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rates.append(100 * p)
        lower.append(100 * (middle - radius))
        upper.append(100 * (middle + radius))
    rate.plot(rate_dates, rates, color=SERIES, linewidth=LINE)
    rate.fill_between(rate_dates, lower, upper, color=SERIES, alpha=0.12, linewidth=0)
    style_axis(rate, "Fraud rate on fully labeled days", ylabel="Fraud rate (%)")
    date_axis(rate)
    return figure(
        png(fig),
        f"{population}; n={stats['rows']:,}. Calendar gaps appear as zero volume. Fraud rates use seven-day groups of fully labeled event dates. Shading is a descriptive 95% Wilson interval assuming independent transactions; correlated fraud episodes can increase uncertainty.",
    )


def comparison_charts(left, right, *, labels, sampling=False):
    keys = (
        ("by_month", "amount_bins", "payment_types")
        if sampling
        else ("amount_bins", "payment_types", "mcc")
    )
    titles = (
        ("Share by transaction month", "Signed amount shares", "Payment-channel shares")
        if sampling
        else (
            "Signed amount shares",
            "Payment-channel shares",
            "Merchant-category shares",
        )
    )
    fig = new_figure(14, 4.5)
    for axis, key, title in zip(fig.subplots(1, 3), keys, titles):
        if key == "amount_bins":
            names = AMOUNT_LABELS
            lv, rv = left[key], right[key]
        else:
            lc, rc = left.get(key, {}), right.get(key, {})
            names = sorted(lc.keys() | rc.keys())
            if key == "mcc" and len(names) > 8:
                top = sorted(names, key=lambda k: (-(lc.get(k, 0) + rc.get(k, 0)), k))[
                    :8
                ]
                lv = [lc.get(k, 0) for k in top] + [
                    sum(lc.get(k, 0) for k in names if k not in top)
                ]
                rv = [rc.get(k, 0) for k in top] + [
                    sum(rc.get(k, 0) for k in names if k not in top)
                ]
                names = top + ["Other"]
            else:
                lv, rv = [lc.get(k, 0) for k in names], [rc.get(k, 0) for k in names]
        x = np.arange(len(names))
        # The cohort under examination is the series; the one it is compared with is a
        # muted baseline.
        axis.bar(
            x - 0.2,
            100 * np.array(lv) / left["rows"],
            width=0.38,
            label=labels[0],
            color=SERIES,
        )
        axis.bar(
            x + 0.2,
            100 * np.array(rv) / right["rows"],
            width=0.38,
            label=labels[1],
            color=MUTED,
        )
        axis.set_xticks(x, names, rotation=65, ha="right")
        style_axis(axis, title, ylabel="Percent of cohort", grid="y")
        add_legend(axis)
    return figure(
        png(fig),
        f"Normalized distributions: {labels[0]} n={left['rows']:,}; {labels[1]} n={right['rows']:,}. Differences are diagnostic signals, not automatic rejection or evidence of model degradation.",
    )


def timeline(report):
    windows = {
        "context": [
            report.get("context_start", report["windows"]["train"][0]),
            report["as_of"],
        ],
        **report["windows"],
    }
    start, end = map(datetime.fromisoformat, windows["context"])
    span = max((end - start).total_seconds(), 1)
    # Context is the backdrop; the supervised partitions step darker through time.
    fills = {
        "context": AXIS,
        "train": ORDINAL[0],
        "validation": ORDINAL[1],
        "evaluation": ORDINAL[2],
    }
    bars = []
    for i, (name, fill) in enumerate(fills.items()):
        begin, finish = map(datetime.fromisoformat, windows[name])
        x = 130 + 690 * (begin - start).total_seconds() / span
        width = 690 * (finish - begin).total_seconds() / span
        bars.append(
            f'<text x="0" y="{24 + i * 34}">{name.title()}</text><rect x="{x:.2f}" y="{8 + i * 34}" width="{max(width, 1):.2f}" height="22" rx="4" fill="{fill}"/>'
        )
    return (
        '<svg class="timeline" viewBox="0 0 840 160" role="img" aria-label="Temporal split timeline; context overlaps all supervised partitions">'
        + "".join(bars)
        + "</svg>"
    )


def split_section(report):
    if "splits" not in report:
        return ""
    rows = []
    purposes = {
        "context": "Historical pool; overlaps other partitions",
        "train": "Fit model parameters",
        "validation": "Select settings and decision threshold",
        "evaluation": "Assess the selected model on later data",
    }
    for name in ("context", "train", "validation", "evaluation"):
        counts = report["splits"][name]
        window = report.get("windows", {}).get(
            name, [report.get("context_start", "Unknown"), report["as_of"]]
        )
        known = counts["rows"] - counts["unavailable_labels"]
        rows.append(
            [
                name.title(),
                f"{window[0][:10]} → {window[1][:10]}",
                number(counts["rows"]),
                number(counts["frauds"]),
                percent(share(counts["frauds"], known)),
                number(counts["unavailable_labels"]),
                purposes[name],
            ]
        )
    body = "<p>Start dates are included; end dates are excluded. Context contains the supervised records: do not sum all four row counts. Pending labels are never treated as legitimate transactions.</p>"
    if report.get("windows"):
        body += timeline(report)
    body += table(
        [
            "Partition",
            "Period",
            "Rows",
            "Known frauds",
            "Fraud / known labels",
            "Pending",
            "Purpose",
        ],
        rows,
        align="llrrrrl",
    )
    ready = report.get("training_ready")
    body += (
        "<h3>"
        + (
            "Training data checks passed"
            if ready
            else "Training data checks need attention"
        )
        + "</h3>"
    )
    reasons = report.get("training_reasons", report.get("insufficient_splits", []))
    body += "".join(f'<p class="notice">{e(reason)}</p>' for reason in reasons)
    criteria = report.get("training_criteria", report.get("dataset_profile", {}))
    body += f"<p>Per supervised partition: at least {e(criteria.get('minimum_rows_per_split', 'unspecified'))} rows, {e(criteria.get('minimum_frauds_per_split', 'unspecified'))} frauds, and both classes. Reserved holdout begins {e(criteria.get('final_holdout_start', 'unspecified'))}. These are technical eligibility checks, not a guarantee of statistical power or model quality. Training enforces its own configured policy.</p>"
    return section("splits", "Time coverage and training partitions", body)


def sampling_section(report):
    sample = report.get("sampling")
    if not sample:
        if "eligible_training_rows" not in report:
            return ""
        return section(
            "sampling",
            "Training selection",
            f"<p>Selected {number(report['splits']['train']['rows'])} of {number(report['eligible_training_rows'])} eligible rows. Refresh this older report to compute distribution diagnostics.</p>",
        )
    left, right = sample["selected"], sample["eligible"]
    body = f'<p class="lead">Selected <strong>{number(left["rows"])}</strong> of <strong>{number(right["rows"])}</strong> eligible training rows ({percent(sample["fraction"])}).</p><p>{e(sample["method"])}. Seed: {e(sample["seed"])}. Validation, evaluation, and context are not sampled by the training cap.</p>'
    body += cohort_table(left, right, ["Selected training", "All eligible training"])
    body += cohort_quality_table(left, right, ["Selected", "Eligible"])
    if left["rows"] and right["rows"]:
        body += comparison_charts(
            left, right, labels=["Selected", "Eligible"], sampling=True
        )
    body += distance_table(sample["distribution_changes"])
    return section(
        "sampling", "Does the training sample represent its eligible history?", body
    )


def cohort_table(left, right, labels):
    rows = []
    for label, stats in zip(labels, (left, right)):
        amount = stats.get("amount", {})
        rows.append(
            [
                label,
                number(stats["rows"]),
                number(stats["known_labels"]),
                number(stats["frauds"]),
                percent(stats["mature_fraud_rate"]),
                percent(share(stats["unknown_labels"], stats["rows"])),
                number(stats["unique_customers"]),
                money(amount.get("quantiles", {}).get("p50")),
            ]
        )
    return (
        table(
            [
                "Population",
                "Rows",
                "Known labels",
                "Frauds",
                "Fraud / known",
                "Pending share",
                "Customers",
                "Median USD¹",
            ],
            rows,
            align="lrrrrrrr",
        )
        + '<p class="muted">¹ Medians are computed on all rows; the amount diagnostics and recorded JSON state the method.</p>'
    )


def cohort_quality_table(left, right, labels):
    rows = []
    for key, label in (
        ("merchant_state", "Absent merchant state"),
        ("postal_code", "Absent postal code"),
    ):
        rows.append(
            [
                label,
                *[
                    percent(share(s["missing"].get(key, 0), s["rows"]))
                    for s in (left, right)
                ],
            ]
        )
    rows.append(
        [
            "Negative amount share",
            *[
                percent(share(s["amount"]["negative_rows"], s["rows"]))
                for s in (left, right)
            ],
        ]
    )
    rows.append(
        [
            "No error recorded",
            *[percent(share(s["no_error_recorded"], s["rows"])) for s in (left, right)],
        ]
    )
    return table(["Measure (% of all cohort rows)", *labels], rows, align="lrr")


def distance_table(changes):
    names = {
        "amount_bins": "Amount bins",
        "payment_types": "Payment channel",
        "mcc": "Merchant category (MCC)",
        "by_month": "Transaction month",
    }
    return (
        table(
            ["Distribution", "Total variation distance"],
            [
                [names.get(k, k), "Unavailable" if v is None else f"{v:.4f}"]
                for k, v in changes.items()
            ],
            align="lr",
        )
        + '<p class="muted">Distance uses normalized category/bin shares: 0 means identical distributions; 1 means disjoint distributions. No automatic pass/fail threshold is applied.</p>'
    )


def comparison_section(report):
    comparison = report.get("comparison")
    if not comparison:
        return ""
    current, reference = comparison["current"], comparison["reference"]
    cw, rw = comparison["current_window"], comparison["reference_window"]
    body = f"<p>Current: <strong>{e(cw[0][:10])} → {e(cw[1][:10])}</strong>. Reference: <strong>{e(rw[0][:10])} → {e(rw[1][:10])}</strong>. Ends are exclusive.</p><p>{e(comparison['population'])}. {e(comparison['label_policy'])}.</p>"
    body += cohort_table(current, reference, ["Current period", "Reference period"])
    body += cohort_quality_table(current, reference, ["Current", "Reference"])
    body += f"<p>Customers first seen within the retained context: <strong>{number(comparison['new_customers'])}</strong>. {e(comparison['new_customer_definition'])}.</p>"
    if comparison["comparable"]:
        days = (datetime.fromisoformat(cw[1]) - datetime.fromisoformat(cw[0])).days
        body += f"<p>Mean daily volume: {current['rows'] / days:,.1f} current; {reference['rows'] / days:,.1f} reference. These are equal-duration windows ({days} days each).</p>"
        body += comparison_charts(current, reference, labels=["Current", "Reference"])
        body += distance_table(comparison["distribution_changes"])
    else:
        body += "".join(
            f'<p class="notice">{e(reason)}</p>' for reason in comparison["reasons"]
        )
    return section("comparison", "Change in the latest transaction cohort", body)


def quality_section(report):
    delivery = report if report.get("kind") == "Ingestion" else report.get("delivery")
    if not delivery:
        return section(
            "quality",
            "Delivery accounting",
            "<p>This older snapshot did not record a delivery breakdown in its report. Its transaction and partition counts remain available above.</p>",
        )
    accepted = delivery.get("accepted_records")
    if accepted is None and delivery.get("status") in ("accepted", "rejected"):
        accepted = delivery.get("valid", 0) if delivery["status"] == "accepted" else 0
    rows = [
        [label, number(delivery.get(key))]
        for key, label in (
            ("extracted_records", "Extracted from source before fault injection"),
            ("received", "Delivered records"),
            ("valid", "Passed row validation"),
            ("invalid", "Invalid records"),
            ("new_transactions", "Valid new transactions"),
            ("label_updates", "Valid newly available labels"),
            ("unchanged", "Valid unchanged observations"),
        )
    ]
    rows.append(
        ["Records accepted by the delivery validation policy", number(accepted)]
    )
    body = table(["Count", "Records"], rows, align="lr")
    body += "<p>Valid = new transactions + label updates + unchanged observations. Received = valid + invalid. The entire delivery is rejected if any record is invalid; valid records in a rejected delivery are not accepted. Label updates do not increase the unique transaction count. Publication completion is recorded separately by the ingestion ledger.</p>"
    if delivery.get("fault"):
        body += f'<p class="notice">Simulated corruption: {e(delivery["fault"])}. Extraction and delivery counts may differ because the fault is injected after reading the source.</p>'
    violations = delivery.get("violations", {})
    guidance = {
        "transaction_time": "Check extraction boundaries; keep transactions inside the declared interval.",
        "label_delay": "Check the source label-availability mapping against the configured delay.",
        "label_visibility": "Mask outcomes that are not available at this cutoff; retain available outcomes.",
        "duplicate_identity": "Remove repeated delivery records and retry the same pending interval.",
        "conflicting_record": "Investigate changed fields or labels for an existing source identity; corrections need a versioned contract.",
    }
    if violations:
        rows = []
        for code, count in sorted(violations.items()):
            advice = guidance.get(
                code,
                "Restore the field's required type/value according to the observation schema, then retry.",
            )
            rows.append(
                [
                    code,
                    "Error",
                    number(count),
                    percent(share(count, delivery.get("received", 0))),
                    ", ".join(map(str, delivery.get("examples", {}).get(code, []))),
                    advice,
                ]
            )
        body += table(
            [
                "Rule",
                "Severity",
                "Affected records",
                "% delivered",
                "Example row IDs",
                "Action",
            ],
            rows,
            align="llrrll",
        )
        body += "<p>Rule counts can overlap; their sum is not the number of invalid records. Examples are limited to 20 IDs per rule.</p>"
    else:
        body += "<p>No row violations recorded.</p>"
    if report.get("error"):
        body += f'<p class="notice">Processing failed: {e(report["error"])}. Counts may describe only a partial attempt.</p>'
    return section("quality", "Delivery accounting and validation findings", body)


def amount_section(stats):
    amount = stats.get("amount") if stats else None
    if not amount:
        return ""
    quantiles = amount["quantiles"]
    rows = [
        ["Minimum", money(amount["min"])],
        ["Mean", money(amount["mean"])],
        ["Maximum", money(amount["max"])],
    ]
    rows += [[label.upper(), money(value)] for label, value in quantiles.items()]
    rows += [
        [
            "Negative amounts",
            f"{number(amount['negative_rows'])} ({percent(share(amount['negative_rows'], stats['rows']))})",
        ],
        ["Zero amounts", number(amount["zero_rows"])],
        [
            "No error recorded",
            f"{number(stats['no_error_recorded'])} ({percent(share(stats['no_error_recorded'], stats['rows']))})",
        ],
        [
            "Top 10 customers' transaction share",
            percent(stats["top_10_customer_share"]),
        ],
    ]
    body = table(["Measure", "Value"], rows, align="lr")
    body += f"<p>Percentiles: {e(amount['quantile_method'])}; n={number(amount['quantile_sample_rows'])}. Min, max, mean, and sign counts use all records. Negative amounts are valid. A blank Errors? field means no error was recorded; optional location fields can be absent, including for online transactions.</p>"
    if stats["rows"]:
        fig = new_figure(14, 4.5)
        channels, categories, customers = fig.subplots(1, 3)
        for axis, key, title in (
            (channels, "payment_types", "Payment channels"),
            (categories, "mcc", "Top merchant categories (MCC)"),
        ):
            top = sorted(stats[key].items(), key=lambda item: (-item[1], item[0]))[:10]
            axis.barh(
                [k for k, _ in reversed(top)],
                [100 * v / stats["rows"] for _, v in reversed(top)],
                # Thin bars, even when only a few categories share the panel.
                height=0.6 if len(top) > 4 else 0.3,
                color=SERIES,
            )
            style_axis(axis, title, "Percent of all records", grid="x")
        top = stats["top_customers"][:10]
        customers.barh(
            [str(k) for k, _ in reversed(top)],
            [100 * v / stats["rows"] for _, v in reversed(top)],
            height=0.6,
            color=SERIES,
        )
        style_axis(
            customers,
            "Top customers by activity",
            "Percent of all records",
            "Source user ID",
            grid="x",
        )
        body += figure(
            png(fig),
            f"Population n={stats['rows']:,}; top-category/customer lists may not sum to 100%.",
        )
    return section(
        "distributions", "Amounts, payment channels, and customer activity", body
    )


def fraud_section(report):
    if report.get("kind") != "Snapshot":
        return ""
    stats = report.get("fraud_exploration")
    title = "Fraud versus legitimate transactions — training split only"
    if stats is None:
        return section(
            "fraud-exploration",
            title,
            "<p>Unavailable in this older report. Refresh the snapshot report to compute these plots.</p>",
        )
    classes = stats["classes"]
    body = f"<p>{e(stats['population'])}. Validation, evaluation, and extra context rows are excluded. Pending labels are never treated as legitimate transactions.</p>"
    body += table(
        [
            "Labeled legitimate",
            "Labeled fraud",
            "Fraud / labeled",
            "Excluded pending labels",
        ],
        [
            [
                number(classes["legitimate"]["rows"]),
                number(classes["fraud"]["rows"]),
                percent(
                    share(
                        classes["fraud"]["rows"],
                        sum(c["rows"] for c in classes.values()),
                    )
                ),
                number(stats["excluded_pending_labels"]),
            ]
        ],
        align="rrrr",
    )
    if not any(c["rows"] for c in classes.values()):
        return section(
            "fraud-exploration",
            title,
            body
            + "<p>No labeled training transactions are available for class comparisons.</p>",
        )
    if not all(c["rows"] for c in classes.values()):
        body += '<p class="notice">Only one class is present; comparisons between classes are unavailable.</p>'
    body += "<p>Use these plots to propose feature experiments. Confirm their value on later validation periods; apparent groupings do not establish model performance or hyperparameter choices.</p>"
    for fig, caption in fraud_figures(stats):
        body += figure(png(fig), caption)
    return section("fraud-exploration", title, body)


def render_report(report):
    stats = report.get("summary") or {}
    kind, status = report.get("kind", "Data"), report.get("status", "unknown")
    population = report.get("population", "valid delivered observations")
    chart_stats = (
        report.get("new_transaction_summary", stats) if kind == "Ingestion" else stats
    )
    chart_population = (
        "Valid new transactions in this delivery"
        if kind == "Ingestion" and "new_transaction_summary" in report
        else population
    )
    start = (
        report.get("context_start")
        if kind == "Snapshot"
        else report.get("previous_cutoff") or report.get("context_start")
    )
    body = (
        split_section(report)
        + quality_section(report)
        + sampling_section(report)
        + fraud_section(report)
        + comparison_section(report)
        + section(
            "coverage",
            "Transaction volume and label coverage",
            charts(
                chart_stats,
                start=start,
                end=report.get("as_of"),
                population=chart_population,
            ),
        )
        + amount_section(chart_stats)
    )
    as_of = report.get("as_of", "unknown")
    return page(
        area="Data",
        title=f"{kind} report — {as_of[:10]}",
        badge=badge(
            status.replace("_", " ").capitalize(), TONES.get(status, "warning")
        ),
        intro=f"<p>As of <strong>{e(as_of)}</strong> · IBM synthetic transactions · source timestamps use a UTC convention.</p><p>Population: {e(population)}. Pending outcomes are excluded from fraud-rate denominators. This is a data report, not a model-performance evaluation.</p>",
        tiles=tiles(
            [
                ("Records summarized", number(stats.get("rows"))),
                ("Known frauds", number(stats.get("frauds"))),
                ("Pending labels", number(stats.get("unknown_labels"))),
                ("Fraud / known labels", percent(stats.get("mature_fraud_rate"))),
            ]
        ),
        body=body,
        evidence=evidence(report),
    )
