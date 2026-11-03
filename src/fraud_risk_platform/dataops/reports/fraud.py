"""Training-only class diagnostics; exact aggregates and bounded scatter samples."""

from fraud_risk_platform.dataops.reports.summary import AMOUNT_LABELS, labeled_rows

SCATTER_LIMITS = {"legitimate": 20_000, "fraud": 5_000}
MINIMUM_RATE_ROWS = 100
# Sample order: a hash of the row identity, independent of amount, time, and label.
SCATTER_ORDER = "sha256('fraud-scatter-v2:' || source_row::VARCHAR), source_row"


def fraud_exploration(db, source: str, parameters=None, *, cutoff) -> dict:
    """Class diagnostics for the rows `SELECT * FROM {source}` returns.

    Only labels released by the cutoff count; pending labels are excluded, not
    treated as legitimate.
    """
    rows = f"({labeled_rows(source, as_of=cutoff)})"
    labeled = f"(SELECT * FROM {rows} WHERE known)"
    values = dict(parameters or {}) | {"as_of": cutoff}

    def fetch(query):
        return db.execute(query, values).fetchall()

    classes = {
        name: {
            "rows": 0,
            "amount_bins": [0] * len(AMOUNT_LABELS),
            "amount_hour": [[0] * 24 for _ in AMOUNT_LABELS],
        }
        for name in SCATTER_LIMITS
    }
    for fraud, amount_bin, hour, count in fetch(
        f"SELECT fraud, amount_bin, tx_hour, count(*) FROM {labeled} GROUP BY ALL"
    ):
        counts = classes["fraud" if fraud else "legitimate"]
        counts["rows"] += count
        counts["amount_bins"][amount_bin] += count
        counts["amount_hour"][amount_bin][hour] += count
    categories = {
        key: {
            value: {"rows": count, "frauds": frauds}
            for value, count, frauds in fetch(
                f"""SELECT {column}, count(*), count(*) FILTER (WHERE fraud)
                FROM {labeled} GROUP BY {column} ORDER BY {column}"""
            )
        }
        for key, column in (("payment_types", "use_chip"), ("mcc", "mcc"))
    }
    limits = dict(SCATTER_LIMITS)
    points = {name: [] for name in limits}
    for fraud, hour, amount in fetch(
        f"""SELECT fraud,
            tx_hour + minute(tx_datetime)::DOUBLE / 60 + second(tx_datetime)::DOUBLE / 3600,
            amount::DOUBLE
        FROM {labeled}
        QUALIFY row_number() OVER (PARTITION BY fraud ORDER BY {SCATTER_ORDER})
            <= CASE WHEN fraud THEN {limits["fraud"]} ELSE {limits["legitimate"]} END
        ORDER BY fraud, {SCATTER_ORDER}"""
    ):
        points["fraud" if fraud else "legitimate"].append([hour, amount])
    return {
        "population": "Selected training split only; labels available at the snapshot cutoff",
        "excluded_pending_labels": fetch(
            f"SELECT count(*) FILTER (WHERE NOT known) FROM {rows}"
        )[0][0],
        "amount_bin_labels": AMOUNT_LABELS,
        "minimum_rate_rows": MINIMUM_RATE_ROWS,
        "classes": classes,
        **categories,
        "scatter": {
            "method": "Deterministic bottom-hash sample within each class (SHA-256 of "
            "the source row); independent of amount and time",
            "limits": limits,
            "columns": ["hour", "amount_usd"],
            "points": points,
        },
    }


def category_groups(groups, limit=None):
    """Choose categories by labeled volume, retaining the remainder as Other."""
    ordered = sorted(groups.items(), key=lambda item: (-item[1]["rows"], item[0]))
    if limit is None or len(ordered) <= limit:
        return ordered
    remainder = ordered[limit:]
    return ordered[:limit] + [
        (
            "Other",
            {
                key: sum(counts[key] for _, counts in remainder)
                for key in ("rows", "frauds")
            },
        )
    ]
