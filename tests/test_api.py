from contextlib import contextmanager
from uuid import UUID

import psycopg
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.serving.app import create_app
from fraud_risk_platform.serving.predictor import FraudPredictor

PAYLOAD = {
    "transaction_id": 9000001,
    "amount": 500.0,
    "timestamp": "2018-06-01T12:00:00",
}


def metric(client, name, **labels):
    """One sample from the API's /metrics, or 0 if it has not been recorded."""
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0


def test_health_without_release(tmp_path):
    with TestClient(create_app(root=tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=PAYLOAD).status_code == 503


@pytest.mark.integration
def test_real_api_artifact_and_persistence(
    tmp_path, release_factory, database, source_database
):
    release = release_factory()
    with TestClient(
        create_app(
            root=tmp_path,
            database_url=database,
            source_database_url=source_database,
            release_id=release["release_id"],
        )
    ) as client:
        assert client.get("/ready").status_code == 200
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 200, response.text
        body = response.json()
        assert UUID(body["prediction_id"]).version == 4
        assert body["release_id"] == release["release_id"]
        assert body["flagged_for_review"] == (body["fraud_score"] >= body["threshold"])
        with connect(database) as conn:
            saved = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id=%s",
                (body["prediction_id"],),
            ).fetchone()
        assert saved["release_id"] == body["release_id"]
        assert saved["transaction_id"] == PAYLOAD["transaction_id"]
        assert saved["fraud_score"] == body["fraud_score"]
        release_id = release["release_id"]
        flagged = str(body["flagged_for_review"]).lower()
        assert (
            metric(
                client,
                "fraud_api_predictions_total",
                release=release_id,
                flagged=flagged,
            )
            == 1
        )
        for step in ("scoring", "persistence"):
            assert (
                metric(
                    client,
                    "fraud_api_step_seconds_count",
                    step=step,
                    release=release_id,
                )
                == 1
            )
        assert (
            metric(
                client,
                "fraud_api_request_seconds_count",
                endpoint="/predict",
                status="200",
                release=release_id,
            )
            == 1
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"transaction_id": 0},
        {"amount": True},
        {"amount": "500"},
        {"timestamp": "2018-06-01T12:00:00Z"},
        {"unexpected": 1},
        {"transaction_id": -1},
        {"transaction_id": True},
        {"transaction_id": "123"},
    ],
)
def test_rejects_invalid_requests(tmp_path, changes):
    with TestClient(create_app(root=tmp_path)) as client:
        assert client.post("/predict", json={**PAYLOAD, **changes}).status_code == 422


def test_invalid_artifact_is_unready(tmp_path, release_factory):
    release = release_factory()
    (tmp_path / "releases" / release["release_id"] / "model.joblib").write_bytes(
        b"broken"
    )
    with TestClient(
        create_app(root=tmp_path, release_id=release["release_id"])
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_storage_outage_does_not_return_unrecorded_prediction(
    tmp_path, release_factory
):
    release = release_factory()
    with TestClient(
        create_app(
            root=tmp_path,
            database_url="postgresql://nobody@127.0.0.1:1/no?connect_timeout=1",
            release_id=release["release_id"],
        )
    ) as client:
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=PAYLOAD).status_code == 503
        release_id = release["release_id"]
        assert (
            metric(
                client,
                "fraud_api_errors_total",
                endpoint="/predict",
                cause="storage",
                release=release_id,
            )
            == 1
        )
        # The failed write is timed; a prediction that was never recorded is not counted.
        assert (
            metric(
                client,
                "fraud_api_step_seconds_count",
                step="persistence",
                release=release_id,
            )
            == 1
        )
        for flagged in ("true", "false"):
            assert (
                metric(
                    client,
                    "fraud_api_predictions_total",
                    release=release_id,
                    flagged=flagged,
                )
                == 0
            )


def test_metrics_count_requests_and_name_each_error_cause(tmp_path):
    with TestClient(create_app(root=tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json=PAYLOAD).status_code == 503
        assert (
            client.post("/predict", json={**PAYLOAD, "unexpected": 1}).status_code
            == 422
        )
        assert client.get("/predict/1").status_code == 404
        assert metric(client, "fraud_api_release_info", release="none") == 1
        assert (
            metric(
                client,
                "fraud_api_request_seconds_count",
                endpoint="/health",
                status="200",
                release="none",
            )
            == 1
        )
        for cause, endpoint in (
            ("no_model", "/predict"),
            ("validation", "/predict"),
            # Unknown paths share one label, so request paths cannot add new series.
            ("http_404", "unmatched"),
        ):
            assert (
                metric(
                    client,
                    "fraud_api_errors_total",
                    endpoint=endpoint,
                    cause=cause,
                    release="none",
                )
                == 1
            )
        assert 'endpoint="/metrics"' not in client.get("/metrics").text


def test_unhandled_error_is_counted_as_a_server_error(
    tmp_path, release_factory, monkeypatch
):
    release = release_factory()

    def broken(self, **inputs):
        raise RuntimeError("scoring bug")

    monkeypatch.setattr(FraudPredictor, "predict", broken)
    app = create_app(root=tmp_path, release_id=release["release_id"])
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/predict", json=PAYLOAD).status_code == 500
        assert (
            metric(
                client,
                "fraud_api_errors_total",
                endpoint="/predict",
                cause="unhandled",
                release=release["release_id"],
            )
            == 1
        )
        assert (
            metric(client, "fraud_api_release_info", release=release["release_id"]) == 1
        )


@pytest.mark.integration
def test_recorded_predictions_join_only_available_matching_outcomes(
    tmp_path, release_factory, database, source_database, ibm_history
):
    from datetime import datetime

    from fraud_risk_platform.monitoring.outcomes import query_outcomes

    release = release_factory()
    payload = {
        "transaction_id": 1,
        "amount": -10.50,
        "timestamp": "2019-01-01T12:00:00",
    }
    with TestClient(
        create_app(
            root=tmp_path,
            database_url=database,
            source_database_url=source_database,
            release_id=release["release_id"],
        )
    ) as client:
        assert client.post("/predict", json=payload).status_code == 200
        assert (
            client.post("/predict", json={**payload, "amount": 11.0}).status_code == 409
        )
    assert (
        query_outcomes(datetime(2019, 1, 1), database, source_url=source_database) == []
    )
    pending = query_outcomes(datetime(2019, 1, 2), database, source_url=source_database)
    assert len(pending) == 1 and pending[0]["fraud"] is None
    mature = query_outcomes(datetime(2019, 1, 4), database, source_url=source_database)
    assert len(mature) == 1 and mature[0]["fraud"] == 0

    # Exercise more than one lookup batch, including observations recorded before
    # source arrival whose identities do not match the eventual source record.
    with connect(database) as conn:
        conn.execute(
            """INSERT INTO predictions
            (prediction_id,source_import_id,transaction_id,tx_datetime,amount,fraud_score,
             flagged_for_review,threshold,release_id,source_run_id,feature_version)
            SELECT gen_random_uuid(),p.source_import_id,p.transaction_id,p.tx_datetime,p.amount + 1,
                   p.fraud_score,p.flagged_for_review,p.threshold,p.release_id,
                   p.source_run_id,p.feature_version
            FROM predictions p CROSS JOIN generate_series(1,1001)"""
        )
    outcomes = query_outcomes(
        datetime(2019, 1, 4), database, source_url=source_database
    )
    assert len(outcomes) == 1002
    assert sum(row["fraud"] is not None for row in outcomes) == 1
    assert all(
        row["label_available_at"] is None for row in outcomes if row["fraud"] is None
    )


class FakePool:
    """Stands in for a connection pool: counts checkouts and its open state."""

    def __init__(self, url):
        self.url, self.state, self.checkouts = url, "created", 0

    def __enter__(self):
        self.state = "open"
        return self

    def __exit__(self, *exc):
        self.state = "closed"

    @contextmanager
    def connection(self):
        assert self.state == "open"
        self.checkouts += 1
        yield self

    def execute(self, query, parameters=None):
        return self

    def fetchone(self):
        return None  # Not a known source transaction.


def test_requests_reuse_one_pool_per_database(tmp_path, release_factory, monkeypatch):
    pools = {}

    def fake_pool(url, *, name):
        pools[name] = FakePool(url)
        return pools[name]

    monkeypatch.setattr("fraud_risk_platform.serving.app.pool", fake_pool)
    release = release_factory()
    app = create_app(
        root=tmp_path,
        database_url="postgresql://platform",
        source_database_url="postgresql://source",
        release_id=release["release_id"],
    )
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        for transaction_id in (9000001, 9000002):
            response = client.post(
                "/predict", json={**PAYLOAD, "transaction_id": transaction_id}
            )
            assert response.status_code == 200, response.text
    # Created once for the app's lifetime, and closed when it stops.
    assert {name: (p.url, p.state) for name, p in pools.items()} == {
        "platform": ("postgresql://platform", "closed"),
        "source": ("postgresql://source", "closed"),
    }
    # One checkout from each for /ready, and from each for every prediction.
    assert pools["platform"].checkouts == pools["source"].checkouts == 3


def test_card_history_outage_is_a_storage_error(tmp_path, release_factory, monkeypatch):
    def unavailable(self, **inputs):
        raise psycopg.OperationalError("source database unavailable")

    monkeypatch.setattr(FraudPredictor, "predict", unavailable)
    release = release_factory()
    with TestClient(
        create_app(root=tmp_path, release_id=release["release_id"])
    ) as client:
        assert client.post("/predict", json=PAYLOAD).status_code == 503
        assert (
            metric(
                client,
                "fraud_api_errors_total",
                endpoint="/predict",
                cause="storage",
                release=release["release_id"],
            )
            == 1
        )
