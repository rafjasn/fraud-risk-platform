"""Every report shares one palette and one page layout."""

import re
from pathlib import Path

import pytest

import fraud_risk_platform
from fraud_risk_platform.dataops.reports.data import render_report
from fraud_risk_platform.monitoring import load
from fraud_risk_platform.monitoring import report as monitoring
from fraud_risk_platform.reporting import palette
from fraud_risk_platform.reporting.page import STYLE, badge, table
from fraud_risk_platform.reporting.style import label_color
from fraud_risk_platform.training.report import write_report

COLOR = re.compile(r"#[0-9a-fA-F]{6}\b")
RANKING = {
    "prevalence": 0.002,
    "average_precision": 0.41,
    "roc_auc": 0.93,
    **{
        f"{kind}_at_top_{rate}pct": value
        for rate in ("0.1", "0.5", "1")
        for kind, value in (("precision", 0.3), ("recall", 0.6))
    },
}
AT_THRESHOLD = {
    "precision": 0.4,
    "recall": 0.6,
    "alert_rate": 0.005,
    "average_precision": 0.41,
    "true_positives": 3,
    "false_positives": 4,
    "false_negatives": 2,
}


def training_report(recommendation="initial_promotion"):
    trial = {
        "number": 0,
        "recipe": {
            "estimator": "logistic",
            "parameters": {"C": 0.49, "solver": "newton-cholesky"},
        },
        "qualified": True,
        "validation": AT_THRESHOLD,
        "validation_ranking": RANKING,
        "validation_average_precision": 0.41,
        "train_average_precision": 0.45,
        "training_seconds": 2.5,
    }
    return {
        "run_id": "run",
        "as_of": "2019-07-08",
        "snapshot_id": "a" * 64,
        "release_id": "b" * 64,
        "recommendation": recommendation,
        "reason": "Validation winner passes <the> gate.",
        "resolved_config": {
            "review_budget": 0.005,
            "executed_recipes": {
                "logistic_history": {
                    "feature_set": "card_history",
                    "feature_spec": {"numeric": ["amount"], "categorical": ["mcc"]},
                    "search_space": {"C": {"kind": "float", "low": 0.1, "high": 100}},
                }
            },
        },
        "validation_winner": "logistic_history",
        "validation_summary": {
            "source": "selected_candidate",
            "train_average_precision": 0.45,
            **RANKING,
        },
        "comparison": {
            "logistic_history": {
                "trial": 0,
                "qualified": True,
                "validation": AT_THRESHOLD,
                "validation_ranking": RANKING,
                "train_average_precision": 0.45,
                "recall_interval": [0.5, 0.7],
            }
        },
        "evaluation": {"candidate": {**AT_THRESHOLD, **RANKING}, "incumbent": None},
        "searches": {"logistic_history": {"trials": [trial], "best": trial}},
        "plots": [],
    }


def monitoring_report():
    window = {
        "window": ["2019-07-08T00:00:00", "2019-08-08T00:00:00"],
        "release_id": "c" * 64,
        "recipe_name": "xgboost_history",
        "recorded_at_clock": "2019-07-08T00:00:00",
        "rejected": {},
        "review_budget": 0.005,
        "transactions": 146_000,
        "flagged": 700,
        "alert_rate": 0.0048,
        "mature": 120_000,
        "pending": 26_000,
        "frauds": 150,
        "precision": 0.114,
        "recall": 0.533,
        "recall_at_budget": 0.55,
        "expected": {
            "precision": 0.2,
            "recall": 0.52,
            "alert_rate": 0.0039,
            "recall_at_budget": 0.53,
        },
    }
    return {
        "simulation_clock": "2019-08-08T00:00:00",
        "windows": [window],
        "plots": [],
    }


def load_report():
    results = [
        {"offset": 0.0, "lag": 0.0, "seconds": 0.012, "status": 200},
        {"offset": 0.05, "lag": 0.001, "seconds": 0.02, "status": 503},
    ]
    return {
        "release_id": "c" * 64,
        "release_changed_to": None,
        "traffic_window": ["2019-07-08T00:00:00", "2019-08-08T00:00:00"],
        "sample": "2 transactions",
        "target_rate": 20.0,
        "concurrency": 1,
        "started_at": "2026-10-18T10:00:00+00:00",
        "finished_at": "2026-10-18T10:00:01+00:00",
        "dashboard": "http://127.0.0.1:3000/d/fraud-api/fraud-api?from=1&to=2",
        "client": load.client_summary(results, 0.1),
        "server": load.server_summary("", ""),
        "plots": [],
    }


def test_colors_are_defined_only_in_the_palette():
    package = Path(fraud_risk_platform.__file__).resolve().parent
    defined = Path(palette.__file__).resolve()
    stray = {
        str(path.relative_to(package)): COLOR.findall(path.read_text())
        for path in package.rglob("*.py")
        if path.resolve() != defined
    }
    assert {path: colors for path, colors in stray.items() if colors} == {}


def test_every_report_uses_the_shared_page(tmp_path):
    write_report(
        tmp_path,
        {
            "run_id": "run",
            "as_of": "unknown",
            "snapshot_id": "s",
            "release_id": None,
            "recommendation": "failed",
            "reason": "Cycle did not complete",
            "searches": {},
            "comparison": {},
        },
    )
    pages = {
        "data": render_report(
            {"kind": "Ingestion", "status": "failed", "error": "Source unavailable"}
        ),
        "training": (tmp_path / "report.html").read_text(),
        "monitoring": monitoring._html(tmp_path, monitoring_report()),
        "load": load._html(tmp_path, load_report()),
    }
    for name, html in pages.items():
        assert html.startswith("<!doctype html>"), name
        assert f"<style>{STYLE}</style>" in html, name
        assert '<details id="provenance">' in html, name
    assert 'class="badge critical"' in pages["training"]
    assert "50.00% server errors" in pages["load"]
    assert (
        'href="http://127.0.0.1:3000/d/fraud-api/fraud-api?from=1&amp;to=2"'
        in (pages["load"])
    )
    assert "Latest month scored" in pages["monitoring"]


def test_training_report_renders_every_section(tmp_path):
    write_report(tmp_path, training_report())
    html = (tmp_path / "report.html").read_text()
    for text in (
        "Initial promotion",
        "Validation summary",
        "Validation selection",
        "Later evaluation",
        "Search trials",
        "Input features",
        "Definitions and limitations",
        "50.0% to 70.0%",
        "Validation winner",
        "&lt;the&gt;",
    ):
        assert text in html
    # Trials list the searched settings, not the fixed ones.
    assert "C=0.49" in html and "solver=newton-cholesky" not in html


def test_keeping_the_incumbent_is_neutral_not_a_warning(tmp_path):
    write_report(tmp_path, training_report("keep_current"))
    assert 'class="badge neutral"' in (tmp_path / "report.html").read_text()


def test_tables_escape_cells_and_align_columns():
    html = table(["Name", "Rows"], [["<b>", "1,234"]], align="lr")
    assert "&lt;b&gt;" in html and '<td class="r">1,234</td>' in html
    with pytest.raises(ValueError, match="alignment"):
        table(["Name", "Rows"], [], align="l")


def test_badges_pair_color_with_an_icon_and_a_label():
    html = badge("Rejected <x>", "critical")
    assert 'class="badge critical"' in html and "×" in html
    assert "Rejected &lt;x&gt;" in html
    with pytest.raises(ValueError, match="tone"):
        badge("Done", "blue")


def test_labels_inside_fills_stay_readable():
    assert label_color(palette.SEQUENTIAL[0]) == palette.INK
    assert label_color(palette.SEQUENTIAL[-1]) == palette.ON_DARK
