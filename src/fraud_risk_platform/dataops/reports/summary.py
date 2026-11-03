"""Data summaries computed in DuckDB over whole populations, with exact quantiles."""

from collections import Counter
from datetime import date, timedelta

AMOUNT_EDGES = [-1000, -100, -10, 0, 10, 100, 1000, 10000]
AMOUNT_LABELS = [
    "<-1000",
    "-1000:-100",
    "-100:-10",
    "-10:0",
    "0:10",
    "10:100",
    "100:1000",
    "1000:10000",
    ">=10000",
]
QUANTILES = {"p01": 0.01, "p50": 0.5, "p95": 0.95, "p99": 0.99}
# An amount on an edge belongs to the higher bin, as bisect_right places it.
AMOUNT_BIN = " + ".join(f"(amount >= {edge})::INTEGER" for edge in AMOUNT_EDGES)


def labeled_rows(source: str, *, as_of=None) -> str:
    """Rows of `SELECT * FROM {source}` with label flags, bins, and hours.

    A label is known once supplied and, with as_of, released by `$as_of`.
    """
    released = " AND label_available_at <= $as_of" if as_of is not None else ""
    return (
        f"SELECT *, is_fraud IS NOT NULL{released} AS known, "
        f"coalesce(is_fraud, false){released} AS fraud, {AMOUNT_BIN} AS amount_bin, "
        f"hour(tx_datetime) AS tx_hour FROM {source}"
    )


def summary_counts(*, rows, known, frauds):
    """Label counts alone, for validation whose results are not rendered."""
    return {
        "rows": rows,
        "known_labels": known,
        "frauds": frauds,
        "unknown_labels": rows - known,
        "mature_fraud_rate": frauds / known if known else None,
    }


def summarize(db, source: str, parameters=None, *, as_of=None) -> dict:
    """Statistics of the rows `SELECT * FROM {source}` returns, in one population."""
    rows = f"({labeled_rows(source, as_of=as_of)})"
    values = dict(parameters or {}) | ({"as_of": as_of} if as_of is not None else {})

    def fetch(query):
        return db.execute(query, values).fetchall()

    bins = ", ".join(
        f"count(*) FILTER (WHERE amount_bin = {index})"
        for index in range(len(AMOUNT_LABELS))
    )
    hours = ", ".join(f"count(*) FILTER (WHERE tx_hour = {hour})" for hour in range(24))
    (
        count,
        known,
        frauds,
        no_state,
        no_postal_code,
        no_error,
        negative,
        zero,
        minimum,
        maximum,
        total,
        customers,
        quantiles,
        amount_bins,
        by_hour,
    ) = fetch(
        f"""SELECT count(*), count(*) FILTER (WHERE known), count(*) FILTER (WHERE fraud),
            count(*) FILTER (WHERE merchant_state IS NULL),
            count(*) FILTER (WHERE postal_code IS NULL),
            count(*) FILTER (WHERE errors IS NULL), count(*) FILTER (WHERE amount < 0),
            count(*) FILTER (WHERE amount = 0), min(amount), max(amount), sum(amount),
            count(DISTINCT user_id),
            quantile_cont(amount::DOUBLE, {list(QUANTILES.values())}), [{bins}], [{hours}]
        FROM {rows}"""
    )[0]
    days = {
        day: [n, k, f]
        for day, n, k, f in fetch(
            f"""SELECT strftime(tx_datetime, '%Y-%m-%d') AS day, count(*),
                count(*) FILTER (WHERE known), count(*) FILTER (WHERE fraud)
            FROM {rows} GROUP BY day ORDER BY day"""
        )
    }
    months = Counter()
    for day, values_by_day in days.items():
        months[day[:7]] += values_by_day[0]

    def groups(column):
        return dict(
            fetch(
                f"SELECT {column}, count(*) FROM {rows} GROUP BY {column} ORDER BY {column}"
            )
        )

    top = fetch(
        f"""SELECT user_id, count(*) AS n FROM {rows}
        GROUP BY user_id ORDER BY n DESC, user_id LIMIT 20"""
    )
    return {
        "rows": count,
        "known_labels": known,
        "frauds": frauds,
        "unknown_labels": count - known,
        "mature_fraud_rate": frauds / known if known else None,
        "by_day": days,
        "by_month": dict(sorted(months.items())),
        "by_hour": by_hour,
        "amount_bins": amount_bins,
        "amount_bin_labels": AMOUNT_LABELS,
        "missing": {"merchant_state": no_state, "postal_code": no_postal_code}
        if count
        else {},
        "no_error_recorded": no_error,
        "payment_types": groups("use_chip"),
        "mcc": groups("mcc"),
        "unique_customers": customers,
        "top_customers": top,
        "top_10_customer_share": sum(n for _, n in top[:10]) / count if count else None,
        "amount": {
            "min": str(minimum) if minimum is not None else None,
            "max": str(maximum) if maximum is not None else None,
            "mean": str(total / count) if count else None,
            "negative_rows": negative,
            "zero_rows": zero,
            "quantiles": dict(
                zip(QUANTILES, quantiles if quantiles else [None] * len(QUANTILES))
            ),
            "quantile_sample_rows": count,
            "quantile_method": "Exact on all rows",
        },
    }


def calendar_series(summary, start=None, end=None):
    """Include every day in [start, end); dates with no records get zero counts."""
    days = summary.get("by_day", {})
    if not days and (start is None or end is None):
        return [], []
    start = date.fromisoformat(start[:10]) if start else date.fromisoformat(min(days))
    end = (
        date.fromisoformat(end[:10])
        if end
        else date.fromisoformat(max(days)) + timedelta(days=1)
    )
    dates, counts = [], []
    while start < end:
        dates.append(start)
        counts.append(days.get(start.isoformat(), [0, 0, 0]))
        start += timedelta(days=1)
    return dates, counts


def distribution_distance(left, right):
    """Total variation distance: 0 identical shares, 1 disjoint distributions."""
    if not sum(left.values()) or not sum(right.values()):
        return None
    ltotal, rtotal = sum(left.values()), sum(right.values())
    return 0.5 * sum(
        abs(left.get(k, 0) / ltotal - right.get(k, 0) / rtotal)
        for k in left.keys() | right.keys()
    )


def distribution_changes(current, reference):
    values = {
        key: distribution_distance(current.get(key, {}), reference.get(key, {}))
        for key in ("payment_types", "mcc", "by_month")
    }
    values["amount_bins"] = distribution_distance(
        dict(enumerate(current["amount_bins"])),
        dict(enumerate(reference["amount_bins"])),
    )
    return values
