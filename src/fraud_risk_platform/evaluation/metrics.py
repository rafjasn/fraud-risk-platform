"""Validation-only threshold selection for a fixed review budget."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

# Reported review volumes as shares of scored transactions; the policy budget is added.
ALERT_RATES = (0.001, 0.005, 0.01)


def alert_name(rate: float) -> str:
    return f"top_{rate * 100:g}pct"


def review_rates(review_budget: float) -> tuple[float, ...]:
    return tuple(sorted({*ALERT_RATES, review_budget}))


def capacity(rows: int, rate: float) -> int:
    """Transactions reviewable at a rate; the epsilon absorbs float error."""
    return math.floor(rate * rows + 1e-9)


def select_threshold(scores, *, review_budget: float) -> float | None:
    """Lowest threshold that flags no more than the review budget.

    Recall only falls as the threshold rises, so this maximizes recall within the
    budget. A tie group straddling the budget is left unflagged; None when even the
    highest-scored tie group exceeds it.
    """
    ranked = np.sort(np.asarray(scores, dtype=float))[::-1]
    k = min(capacity(len(ranked), review_budget), len(ranked))
    if k == 0:
        return None
    boundary = ranked[k - 1]
    if k < len(ranked) and ranked[k] == boundary:
        above = ranked[ranked > boundary]
        return float(above[-1]) if above.size else None
    return float(boundary)


def ranking_metrics(labels, scores, *, review_budget: float) -> dict:
    """Threshold-free quality, defined for every fitted model."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    result = {
        "prevalence": positives / len(labels),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }
    ranked = np.sort(scores)[::-1]
    for rate in review_rates(review_budget):
        # Expected frauds among the top k scores; a tied boundary is split evenly.
        k = max(1, capacity(len(scores), rate))
        above = scores > ranked[k - 1]
        tied = scores == ranked[k - 1]
        hits = labels[above].sum() + (k - above.sum()) * labels[tied].mean()
        result[f"precision_at_{alert_name(rate)}"] = float(hits / k)
        result[f"recall_at_{alert_name(rate)}"] = float(hits / positives)
    return result


def bootstrap_recall(
    labels, scores: dict, *, review_budget: float, resamples=1000, seed=0
) -> dict:
    """Recall at the review budget over row resamples shared by every model.

    Sharing resamples makes model differences paired. Ties are broken by row order
    rather than split, and frauds cluster on cards, so row resampling understates
    the uncertainty: overlapping intervals are a tie, not a near miss.
    """
    labels = np.asarray(labels, dtype=np.float64)
    rows = len(labels)
    k = capacity(rows, review_budget)
    ranked = {
        name: np.argsort(-np.asarray(values, dtype=float), kind="stable")
        for name, values in scores.items()
    }
    generator = np.random.default_rng(seed)
    draws = {name: np.empty(resamples) for name in scores}
    for draw in range(resamples):
        weights = np.bincount(generator.integers(0, rows, rows), minlength=rows)
        frauds = weights @ labels
        for name, order in ranked.items():
            weight = weights[order].astype(np.float64)
            fraud = labels[order]
            cumulative = np.cumsum(weight)
            # Whole rows within capacity, then the boundary row's remaining share.
            cut = int(np.searchsorted(cumulative, k, side="right"))
            caught = weight[:cut] @ fraud[:cut]
            if cut < rows:
                caught += (k - (cumulative[cut - 1] if cut else 0.0)) * fraud[cut]
            draws[name][draw] = caught / frauds
    return draws


def interval(values) -> list[float]:
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def log_fields(values: dict, prefix: str) -> str:
    """Headline metrics as name=value log fields; MLflow and reports keep all."""
    return " ".join(
        f"{prefix}{key}={value:.6f}"
        for key, value in values.items()
        if key in {"precision", "recall", "alert_rate", "average_precision"}
        or key.startswith("recall_at_top_")
    )


def metrics(labels, scores, threshold: float) -> dict:
    tn, fp, fn, tp = (
        int(v)
        for v in confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    )
    return {
        "rows": len(labels),
        "frauds": int(sum(labels)),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "average_precision": float(average_precision_score(labels, scores)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "flagged_transactions": tp + fp,
        "alert_rate": (tp + fp) / len(labels),
    }


def recommendation(
    candidate: dict | None,
    incumbent: dict | None,
    *,
    review_budget: float,
    minimum_recall_gain: float,
) -> tuple[bool, str]:
    """Compare frozen models by evaluation recall at the same review volume."""
    key = f"recall_at_{alert_name(review_budget)}"
    if candidate is None:
        return False, "No candidate qualifies within the validation review budget."
    # Reviewing a random sample of the budget catches that share of fraud.
    if candidate[key] <= review_budget:
        return (
            False,
            "Validation winner does not beat random review within the budget on the later evaluation.",
        )
    if incumbent and candidate[key] <= incumbent[key] + minimum_recall_gain:
        return (
            False,
            "Validation winner does not exceed incumbent evaluation recall at the review budget by the required margin.",
        )
    return True, "Validation winner passes the later evaluation gate" + (
        " and improves incumbent recall at the review budget."
        if incumbent
        else "; explicit initial promotion is available."
    )
