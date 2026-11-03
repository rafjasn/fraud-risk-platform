"""Point-in-time card history shared by snapshot training and API serving.

A target transaction sees only its card's transactions with strictly earlier
timestamps in the preceding HISTORY_DAYS, and a fraud label only once released
(label_available_at <= the target's timestamp). Every lookup stays inside that
window, so one card's window from the source database reproduces the features a
snapshot's full context produces.
"""

import numpy as np
import pandas as pd

from fraud_risk_platform.contracts.features import HISTORY_DAYS

HISTORY_COLUMNS = [
    "user_id",
    "card_id",
    "tx_datetime",
    "amount",
    "merchant_id",
    "mcc",
    "is_fraud",
    "label_available_at",
]
WINDOW = HISTORY_DAYS * 86_400
COUNT_WINDOWS = {
    "card_count_1h": 3_600,
    "card_count_24h": 86_400,
    "card_count_7d": 7 * 86_400,
}
SPAN = 2**34  # seconds offset per group in sort keys (timestamps before 2514)


def _seconds(values) -> np.ndarray:
    seconds = np.asarray(pd.to_datetime(values), dtype="datetime64[s]").astype(np.int64)
    if len(seconds) and (seconds.min() <= WINDOW or seconds.max() >= SPAN):
        raise ValueError("History timestamps are outside the supported range")
    return seconds


def _codes(history_values, target_values):
    """Dense integer codes shared by history and targets.

    Each side is factorized alone and only the distinct values are merged, so
    millions of category strings are never materialized as Python objects.
    """
    history_codes, history_uniques = pd.factorize(history_values)
    target_codes, target_uniques = pd.factorize(target_values)
    if (history_codes < 0).any() or (target_codes < 0).any():
        raise ValueError("History identities and categories cannot be null")
    joint = pd.factorize(
        pd.Index(np.asarray(history_uniques, dtype=object)).append(
            pd.Index(np.asarray(target_uniques, dtype=object))
        )
    )[0].astype(np.int64)
    split = len(history_uniques)
    return joint[:split][history_codes], joint[split:][target_codes]


def _pair_codes(first, second):
    """Codes for value pairs, from each value's (history, targets) codes."""
    (first_h, first_t), (second_h, second_t) = first, second
    width = max(second_h.max(initial=0), second_t.max(initial=0)) + 1
    return _codes(first_h * width + second_h, first_t * width + second_t)


def _previous_gap(group_h, time_h, group_t, time_t):
    """Seconds since the latest strictly earlier row of the same group in the window."""
    order = np.lexsort((time_h, group_h))
    keys = group_h[order] * SPAN + time_h[order]
    previous = np.searchsorted(keys, group_t * SPAN + time_t, "left") - 1
    valid = previous >= 0
    valid[valid] = keys[previous[valid]] // SPAN == group_t[valid]
    gap = np.full(len(time_t), np.nan)
    gap[valid] = time_t[valid] - keys[previous[valid]] % SPAN
    gap[gap > WINDOW] = np.nan
    return gap


def _confirmed_frauds(card_h, time_h, released_h, fraud_h, card_t, time_t):
    counts = np.zeros(len(time_t))
    if not fraud_h.any():
        return counts
    targets = pd.Series(card_t).groupby(card_t).indices
    event_card = card_h[fraud_h]
    for card in np.unique(event_card):
        rows = targets.get(card)
        if rows is None:
            continue
        events = event_card == card
        occurred = time_h[fraud_h][events]
        released = released_h[fraud_h][events]
        at = time_t[rows][:, None]
        visible = (occurred >= at - WINDOW) & (occurred < at) & (released <= at)
        counts[rows] = visible.sum(axis=1)
    return counts


def build_history_features(
    targets: pd.DataFrame, history: pd.DataFrame, names: list[str]
) -> pd.DataFrame:
    """Return the requested history features, aligned with targets' index."""
    time_t = _seconds(targets.tx_datetime)
    time_h = _seconds(history.tx_datetime)
    card_h, card_t = _pair_codes(
        _codes(history.user_id, targets.user_id),
        _codes(history.card_id, targets.card_id),
    )
    order = np.lexsort((time_h, card_h))
    keys = card_h[order] * SPAN + time_h[order]
    target_keys = card_t * SPAN + time_t
    # Rows before `end` are strictly earlier; `start(s)` is the first row in [t - s, t).
    end = np.searchsorted(keys, target_keys, "left")

    def start(seconds):
        return np.searchsorted(keys, target_keys - seconds, "left")

    values = {
        name: (end - start(seconds)).astype(float)
        for name, seconds in COUNT_WINDOWS.items()
    }
    amounts = history.amount.to_numpy(dtype=float)[order]
    totals = np.concatenate([[0.0], np.cumsum(amounts)])
    day = start(86_400)
    values["card_amount_sum_24h"] = totals[end] - totals[day]
    month = start(WINDOW)
    count = end - month
    mean = np.divide(
        totals[end] - totals[month],
        count,
        out=np.full(len(time_t), np.nan),
        where=count > 0,
    )
    values["amount_vs_card_mean_30d"] = np.log1p(
        np.abs(targets.amount.to_numpy(dtype=float))
    ) - np.log1p(np.abs(mean))
    values["hours_since_card_last"] = (
        _previous_gap(card_h, time_h, card_t, time_t) / 3_600
    )
    for name, column in (
        ("new_merchant_for_card_30d", "merchant_id"),
        ("new_mcc_for_card_30d", "mcc"),
    ):
        if name in names:
            group_h, group_t = _pair_codes(
                (card_h, card_t), _codes(history[column], targets[column])
            )
            gap = _previous_gap(group_h, time_h, group_t, time_t)
            values[name] = np.isnan(gap).astype(float)
    if "card_confirmed_frauds_30d" in names:
        values["card_confirmed_frauds_30d"] = _confirmed_frauds(
            card_h,
            time_h,
            _seconds(history.label_available_at),
            history.is_fraud.eq(True).to_numpy(),
            card_t,
            time_t,
        )
    return pd.DataFrame(
        {name: values[name] for name in names}, index=targets.index, dtype=float
    )
