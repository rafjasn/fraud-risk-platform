"""Load tests send the coming month through the API and report what each side saw."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from fraud_risk_platform.contracts.api import PredictionRequest
from fraud_risk_platform.monitoring.load import (
    client_summary,
    payload,
    run_load,
    server_summary,
)
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.serving.app import create_app
from fraud_risk_platform.serving.metrics import ApiMetrics
from fraud_risk_platform.training.cycle import run_cycle


class ClientApi:
    """The load generator's view of an in-process API."""

    def __init__(self, client):
        self.client = client

    def post(self, path, body):
        return self.client.post(path, json=body).status_code

    def get(self, path):
        response = self.client.get(path)
        return response.status_code, response.text


def test_client_summary_separates_rejections_from_failures():
    outcomes = [(10, 200), (20, 200), (30, 200), (500, 422), (1000, 503), (2000, None)]
    results = [
        {"offset": i / 10, "lag": i / 100, "seconds": ms / 1000, "status": status}
        for i, (ms, status) in enumerate(outcomes)
    ]
    summary = client_summary(results, seconds=2.0)
    assert summary["statuses"] == {"200": 3, "422": 1, "503": 1, "none": 1}
    # No response counts as a server failure; a 422 is the client's.
    assert summary["server_error_rate"] == pytest.approx(2 / 6)
    assert summary["client_error_rate"] == pytest.approx(1 / 6)
    assert summary["achieved_rate"] == 3
    assert summary["max_lag_ms"] == pytest.approx(50)
    latency = summary["latency_ms"]
    assert (latency["count"], latency["p50"], latency["max"]) == (
        3,
        pytest.approx(20),
        pytest.approx(30),
    )


def test_server_summary_is_the_change_between_scrapes():
    metrics = ApiMetrics()
    metrics.request(endpoint="/predict", status=200, release="r", seconds=5, cause=None)
    before = generate_latest(metrics.registry).decode()
    for _ in range(2):
        metrics.steps.labels("scoring", "r").observe(0.1)
        metrics.steps.labels("persistence", "r").observe(0.2)
        metrics.request(
            endpoint="/predict", status=200, release="r", seconds=0.4, cause=None
        )
    metrics.steps.labels("scoring", "r").observe(0.05)
    metrics.request(
        endpoint="/predict", status=503, release="r", seconds=0.1, cause="storage"
    )
    metrics.request(endpoint="/health", status=200, release="r", seconds=9, cause=None)
    summary = server_summary(before, generate_latest(metrics.registry).decode())
    assert summary["requests"] == 3
    assert summary["statuses"] == {"200": 2, "503": 1}
    assert summary["errors"] == {"storage": 1}
    assert summary["mean_ms"] == pytest.approx(300)
    scoring, persistence = summary["steps"]["scoring"], summary["steps"]["persistence"]
    assert (scoring["count"], persistence["count"]) == (3, 2)
    assert scoring["mean_ms"] == pytest.approx(250 / 3)
    assert persistence["ms_per_request"] == pytest.approx(400 / 3)
    assert "history" not in summary["steps"]
    assert summary["other_ms_per_request"] == pytest.approx(300 - 650 / 3)


def test_payload_is_a_valid_api_request_with_source_nulls():
    row = {
        "source_row": 42,
        "tx_datetime": datetime(2019, 8, 9, 12, 30),
        "amount": -10.5,
        "use_chip": "Online Transaction",
        "mcc": "5411",
        "merchant_city": "ONLINE",
        "merchant_state": None,
        "postal_code": None,
        "user_id": 3,
        "card_id": 0,
        "merchant_id": -7,
    }
    request = PredictionRequest.model_validate_json(json.dumps(payload(row)))
    assert (request.transaction_id, request.timestamp) == (42, row["tx_datetime"])
    assert request.merchant_state is None and request.postal_code is None


@pytest.mark.integration
def test_load_sends_the_coming_month_through_the_api(
    tmp_path, simulate, card_policy, databases
):
    settings, urls, advance = simulate(date(2019, 3, 20))
    snapshot = advance(initial=True)
    with TestClient(create_app(root=tmp_path / "empty")) as client:
        with pytest.raises(ValueError, match="not ready"):
            run_load(
                tmp_path,
                settings,
                requests=5,
                rate=1000,
                concurrency=1,
                api=ClientApi(client),
                **urls,
            )
    release = run_cycle(
        root=tmp_path,
        snapshot=snapshot["snapshot_id"],
        policy=card_policy,
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )["release_id"]
    app = create_app(
        root=tmp_path,
        database_url=databases["platform"],
        source_database_url=databases["source"],
        release_id=release,
    )
    with TestClient(app) as client:
        result = run_load(
            tmp_path,
            settings,
            requests=30,
            rate=1000,
            concurrency=1,
            api=ClientApi(client),
            **urls,
        )
    assert result["release_id"] == release and result["release_changed_to"] is None
    client_view, server = result["client"], result["server"]
    assert client_view["statuses"] == {"200": 30}
    assert client_view["latency_ms"]["count"] == 30
    assert (server["requests"], server["errors"]) == (30, {})
    # A card-history release runs every step for every request.
    assert {step: values["count"] for step, values in server["steps"].items()} == {
        "history": 30,
        "scoring": 30,
        "persistence": 30,
    }
    start, end = map(datetime.fromisoformat, result["traffic_window"])
    with connect(databases["platform"]) as conn:
        stored = conn.execute(
            "SELECT tx_datetime, replay_id FROM predictions"
        ).fetchall()
    assert len(stored) == 30
    assert all(start <= row["tx_datetime"] < end for row in stored)
    assert all(row["replay_id"] is None for row in stored)
    report = Path(result["report"])
    assert (report.parent / "latency.png").exists()
    assert "Open this period in Grafana" in report.read_text()
