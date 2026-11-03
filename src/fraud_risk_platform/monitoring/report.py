"""Mature-outcome monitoring of replayed predictions at the simulation clock.

Outcomes count only once released by the clock (label_available_at <= clock); later
ones stay pending and are counted when a later snapshot advances the clock. Each
replayed month is compared with what its release achieved on its evaluation month.
"""

from pathlib import Path

import pandas as pd

from fraud_risk_platform.dataops.workflow import simulation_clock
from fraud_risk_platform.evaluation.charts import write_monitoring
from fraud_risk_platform.evaluation.metrics import alert_name, ranking_metrics
from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.database import connect, connect_source
from fraud_risk_platform.persistence.files import write_json
from fraud_risk_platform.reporting.page import (
    evidence,
    figure,
    page,
    percent,
    section,
    table,
    tiles,
)


def _outcomes(replay, clock, *, source_url, platform_url) -> pd.DataFrame:
    with connect(platform_url) as conn:
        predictions = pd.DataFrame(
            conn.execute(
                """SELECT transaction_id, tx_datetime, fraud_score, flagged_for_review
                FROM predictions WHERE replay_id=%s""",
                (replay["replay_id"],),
            ).fetchall(),
            columns=[
                "transaction_id",
                "tx_datetime",
                "fraud_score",
                "flagged_for_review",
            ],
        )
    with connect_source(source_url) as conn:
        released = pd.DataFrame(
            conn.execute(
                """SELECT source_row AS transaction_id, tx_datetime AS source_datetime,
                is_fraud FROM ibm_complete_transactions WHERE import_id=%s
                AND tx_datetime >= %s AND tx_datetime < %s AND label_available_at <= %s""",
                (
                    replay["source_import_id"],
                    replay["window_start"],
                    replay["window_end"],
                    clock,
                ),
            ).fetchall(),
            columns=["transaction_id", "source_datetime", "is_fraud"],
        )
    joined = predictions.merge(released, on="transaction_id", how="left")
    matched = joined.source_datetime.notna()
    if (joined.source_datetime[matched] != joined.tx_datetime[matched]).any():
        raise ValueError("Replayed predictions disagree with their source transactions")
    joined["fraud"] = joined.is_fraud.astype("float")
    return joined


def _window_metrics(frame, budget) -> dict:
    mature = frame[frame.fraud.notna()]
    frauds = int(mature.fraud.sum())
    flagged = mature.flagged_for_review
    caught = int((flagged & (mature.fraud == 1)).sum())
    result = {
        "transactions": len(frame),
        "flagged": int(frame.flagged_for_review.sum()),
        "alert_rate": float(frame.flagged_for_review.mean()) if len(frame) else 0.0,
        "mature": len(mature),
        "pending": len(frame) - len(mature),
        "frauds": frauds,
        "caught": caught,
        "precision": caught / int(flagged.sum()) if flagged.any() else None,
        "recall": caught / frauds if frauds else None,
    }
    if budget and frauds and frauds < len(mature):
        ranking = ranking_metrics(
            mature.fraud.astype(int), mature.fraud_score, review_budget=budget
        )
        result["average_precision"] = ranking["average_precision"]
        result["recall_at_budget"] = ranking[f"recall_at_{alert_name(budget)}"]
    return result


def monitor(
    root: Path,
    settings,
    *,
    review_budget: float,
    source_url=None,
    platform_url=None,
) -> dict:
    clock = simulation_clock(settings, source_url=source_url, platform_url=platform_url)
    now = clock["clock"]
    with connect(platform_url) as conn:
        replays = conn.execute(
            "SELECT * FROM replays WHERE simulation_id=%s ORDER BY window_start",
            (clock["simulation_id"],),
        ).fetchall()
    if not replays:
        raise ValueError(
            "No replayed traffic in this simulation; run make replay first"
        )
    windows, daily, weekly = [], [], []
    for replay in replays:
        manifest, _ = load_release(root, replay["release_id"])
        budget = manifest.get("review_budget", review_budget)
        frame = _outcomes(replay, now, source_url=source_url, platform_url=platform_url)
        expected = (manifest.get("evaluation") or {}).get("candidate") or {}
        windows.append(
            {
                "replay_id": str(replay["replay_id"]),
                "window": [
                    replay["window_start"].isoformat(),
                    replay["window_end"].isoformat(),
                ],
                "release_id": replay["release_id"],
                "recipe_name": manifest.get("recipe_name"),
                "recorded_at_clock": replay["clock"].isoformat(),
                "rejected": replay["rejected"],
                "review_budget": budget,
                **_window_metrics(frame, budget),
                "expected": {
                    "precision": expected.get("precision"),
                    "recall": expected.get("recall"),
                    "alert_rate": expected.get("alert_rate"),
                    "average_precision": expected.get("average_precision"),
                    "recall_at_budget": expected.get(f"recall_at_{alert_name(budget)}"),
                },
            }
        )
        day = frame.tx_datetime.dt.floor("D")
        daily.append(
            frame.groupby(day)
            .flagged_for_review.mean()
            .rename("alert_rate")
            .reset_index()
            .rename(columns={"tx_datetime": "day"})
        )
        # Weekly recall at the threshold from mature outcomes; daily counts are too small.
        week = replay["window_start"] + pd.to_timedelta(
            (frame.tx_datetime - replay["window_start"]).dt.days // 7 * 7, unit="D"
        )
        mature = frame.fraud.notna()
        for start, group in frame[mature].groupby(week[mature]):
            frauds = int(group.fraud.sum())
            if frauds:
                weekly.append(
                    {
                        "week": start,
                        "recall": float(
                            (group.flagged_for_review & (group.fraud == 1)).sum()
                            / frauds
                        ),
                        "frauds": frauds,
                        "expected": expected.get("recall"),
                    }
                )
    output = root / "monitoring" / f"{now:%Y-%m-%d}"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "simulation_clock": now.isoformat(),
        "simulation_id": clock["simulation_id"],
        "windows": windows,
        "plots": [
            write_monitoring(
                output,
                pd.concat(daily, ignore_index=True),
                pd.DataFrame(weekly, columns=["week", "recall", "frauds", "expected"]),
                review_budget=windows[-1]["review_budget"],
                title=f"Replayed traffic at simulated clock {now:%Y-%m-%d}",
            )
        ],
    }
    write_json(output / "report.json", report)
    (output / "report.html").write_text(_html(output, report))
    return {"report": str(output / "report.html"), **report}


def _percent(value):
    return percent(value, 1, missing="–")


def _html(directory: Path, report: dict) -> str:
    windows = report["windows"]

    def versus(window, key):
        # Recall at the budget is absent until both classes have matured.
        return f"{_percent(window.get(key))} ({_percent(window['expected'].get(key))})"

    # One column per replayed month, so months compare side by side.
    measures = [
        (
            "Release",
            lambda w: f"{w['recipe_name'] or ''} {w['release_id'][:8]}".strip(),
        ),
        ("Recorded at clock", lambda w: w["recorded_at_clock"][:10]),
        ("Scored", lambda w: f"{w['transactions']:,}"),
        ("Rejected (missing fields)", lambda w: f"{sum(w['rejected'].values()):,}"),
        ("Flagged", lambda w: f"{w['flagged']:,}"),
        ("Alert rate (evaluation month)", lambda w: versus(w, "alert_rate")),
        ("Mature / pending outcomes", lambda w: f"{w['mature']:,} / {w['pending']:,}"),
        ("Mature frauds", lambda w: f"{w['frauds']:,}"),
        ("Precision at threshold (evaluation month)", lambda w: versus(w, "precision")),
        ("Recall at threshold (evaluation month)", lambda w: versus(w, "recall")),
        (
            "Recall at review budget (evaluation month)",
            lambda w: versus(w, "recall_at_budget"),
        ),
    ]
    months = table(
        [
            "Traffic window",
            *(f"{w['window'][0][:10]} → {w['window'][1][:10]}" for w in windows),
        ],
        [[label, *(value(w) for w in windows)] for label, value in measures],
        align="l" + "r" * len(windows),
    )
    latest = windows[-1]
    expected = latest["expected"]
    figures = "".join(
        figure(
            (directory / name).read_bytes(),
            alt="Daily share of replayed transactions flagged, and weekly recall at the saved threshold on mature outcomes",
        )
        for name in report["plots"]
    )
    body = section(
        "months",
        "Replayed months",
        months
        + "<p>Precision and recall at the threshold use the release's saved threshold on mature outcomes. Recall at the review budget ranks mature transactions only, so it is comparable with the evaluation month's value; it is computed once both classes have matured. Rejected transactions lacked a required field and would have received HTTP 422 from the API. Batch replay measures scoring throughput, not API latency.</p>",
    ) + section("trends", "Review volume and recall over time", figures)
    return page(
        area="Monitoring",
        title=f"Replayed traffic — simulated clock {report['simulation_clock'][:10]}",
        intro="<p>Each month of traffic was scored by the release deployed at the month's start, with the API's feature code and point-in-time card history, and recorded before the next snapshot released any of the month's outcomes. Outcomes count only once released by the simulated clock; the rest are pending and will count in a later report. Values in parentheses are what the same release achieved on its evaluation month.</p>",
        tiles=tiles(
            [
                (
                    "Latest month scored",
                    f"{latest['transactions']:,}",
                    f"{latest['window'][0][:10]} to {latest['window'][1][:10]}",
                ),
                (
                    "Flagged for review",
                    _percent(latest["alert_rate"]),
                    f"budget {latest['review_budget']:.1%}; evaluation month {_percent(expected['alert_rate'])}",
                ),
                (
                    "Recall at threshold, mature outcomes",
                    _percent(latest["recall"]),
                    f"evaluation month {_percent(expected['recall'])}",
                ),
                (
                    "Mature / pending outcomes",
                    f"{latest['mature']:,} / {latest['pending']:,}",
                    f"{latest['frauds']:,} mature frauds",
                ),
            ]
        ),
        body=body,
        evidence=evidence(report),
    )
