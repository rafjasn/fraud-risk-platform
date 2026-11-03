"""Point-in-time card history: leakage rules and training/serving parity."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_risk_platform.contracts.features import HISTORY_DAYS, FeatureSet
from fraud_risk_platform.contracts.training import Recipe, TrainingPolicy
from fraud_risk_platform.dataops.reader import SnapshotReader
from fraud_risk_platform.features.history import (
    HISTORY_COLUMNS,
    build_history_features,
)
from fraud_risk_platform.features.matrices import feature_matrices
from fraud_risk_platform.features.transaction import build_transaction_features
from fraud_risk_platform.models.adapters import SklearnAdapter
from fraud_risk_platform.models.releases import save_release
from fraud_risk_platform.monitoring.outcomes import query_outcomes
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.serving.app import create_app
from fraud_risk_platform.serving.predictor import FraudPredictor
from fraud_risk_platform.serving.storage import card_history
from fraud_risk_platform.training.cycle import run_cycle

T = datetime(2019, 6, 1, 12)
NAMES = [
    "card_count_1h",
    "card_count_24h",
    "card_count_7d",
    "card_amount_sum_24h",
    "amount_vs_card_mean_30d",
    "hours_since_card_last",
    "new_merchant_for_card_30d",
    "new_mcc_for_card_30d",
    "card_confirmed_frauds_30d",
]
HISTORY_SPEC = FeatureSet(
    version="history-v1", numeric=["amount", *NAMES], categorical=["use_chip"]
)


def transaction(
    when, *, amount=10.0, merchant=1, mcc="5411", card=0, fraud=None, released=None
):
    return {
        "user_id": 7,
        "card_id": card,
        "tx_datetime": when,
        "amount": amount,
        "merchant_id": merchant,
        "mcc": mcc,
        "is_fraud": fraud,
        "label_available_at": released or when + timedelta(days=7),
    }


def features(history, target, names=NAMES):
    return build_history_features(
        pd.DataFrame([target]), pd.DataFrame(history, columns=HISTORY_COLUMNS), names
    ).iloc[0]


def test_only_strictly_earlier_transactions_count():
    history = [
        transaction(T - timedelta(hours=2), amount=20.0),
        transaction(T - timedelta(minutes=30), amount=30.0),
        transaction(T, amount=999.0),  # same timestamp: not "before"
        transaction(T + timedelta(hours=1), amount=999.0),  # future
        transaction(T - timedelta(minutes=10), amount=999.0, card=1),  # other card
    ]
    row = features(history, transaction(T, amount=50.0))
    assert row.card_count_1h == 1
    assert row.card_count_24h == 2
    assert row.card_amount_sum_24h == 50.0
    assert row.hours_since_card_last == pytest.approx(0.5)
    assert row.amount_vs_card_mean_30d == pytest.approx(np.log1p(50) - np.log1p(25))


def test_labels_count_only_once_released_and_inside_the_window():
    history = [
        transaction(T - timedelta(days=10), fraud=True, released=T - timedelta(days=3)),
        transaction(T - timedelta(days=2), fraud=True, released=T + timedelta(days=5)),
        transaction(
            T - timedelta(days=40), fraud=True, released=T - timedelta(days=33)
        ),
        transaction(T - timedelta(days=5), fraud=False, released=T - timedelta(days=1)),
    ]
    assert features(history, transaction(T)).card_confirmed_frauds_30d == 1


def test_novelty_and_empty_history():
    history = [
        transaction(T - timedelta(days=10), merchant=1, mcc="5411"),
        transaction(T - timedelta(days=40), merchant=2, mcc="5812"),
    ]
    seen = features(history, transaction(T, merchant=1, mcc="5411"))
    assert seen.new_merchant_for_card_30d == 0 and seen.new_mcc_for_card_30d == 0
    stale = features(history, transaction(T, merchant=2, mcc="5812"))
    assert stale.new_merchant_for_card_30d == 1 and stale.new_mcc_for_card_30d == 1
    empty = features([], transaction(T))
    assert empty.card_count_7d == 0 and empty.card_confirmed_frauds_30d == 0
    assert np.isnan(empty.hours_since_card_last)
    assert np.isnan(empty.amount_vs_card_mean_30d)
    assert empty.new_merchant_for_card_30d == 1


def test_batch_training_features_match_one_card_window_lookups():
    """Serving sees one card's [t - 30 days, t) window with labels masked at t."""
    generator = np.random.default_rng(3)
    rows = []
    for card in range(3):
        times = T + pd.to_timedelta(
            np.sort(generator.integers(0, 60 * 1440, 150)), "min"
        )
        for when in times:
            rows.append(
                transaction(
                    when.to_pydatetime(),
                    amount=float(generator.choice([-5.0, 12.5, 80.0, 900.0])),
                    merchant=int(generator.integers(0, 6)),
                    mcc=str(generator.choice(["5411", "5812", "5999"])),
                    card=card,
                    fraud=bool(generator.random() < 0.15),
                    released=when.to_pydatetime()
                    + timedelta(days=int(generator.integers(1, 20))),
                )
            )
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    batch = build_history_features(history, history, NAMES)
    for index in generator.choice(len(history), 120, replace=False):
        target = history.iloc[index]
        window = history[
            (history.card_id == target.card_id)
            & (history.tx_datetime >= target.tx_datetime - timedelta(days=HISTORY_DAYS))
            & (history.tx_datetime < target.tx_datetime)
        ].astype({"is_fraud": object})
        window.loc[window.label_available_at > target.tx_datetime, "is_fraud"] = None
        single = build_history_features(target.to_frame().T, window, NAMES).iloc[0]
        np.testing.assert_allclose(single.to_numpy(), batch.iloc[index].to_numpy())


def test_history_feature_contract():
    assert {"user_id", "card_id", "merchant_id", "mcc"} <= set(HISTORY_SPEC.inputs)
    assert HISTORY_SPEC.history_features == NAMES
    with pytest.raises(ValidationError, match="history-v1"):
        FeatureSet(numeric=["amount", "card_count_24h"])
    with pytest.raises(ValidationError, match="history-v1"):
        FeatureSet(version="history-v1", numeric=["amount"])
    with pytest.raises(ValidationError):
        FeatureSet(version="history-v1", categorical=["card_count_24h"])


def test_predictor_fetches_the_card_window_and_requires_card_identity(tmp_path):
    history = [
        transaction(
            T - timedelta(days=d), fraud=d % 2 == 0, released=T - timedelta(days=d - 1)
        )
        for d in range(2, 20)
    ]
    targets = pd.DataFrame(
        [
            {
                **transaction(T - timedelta(hours=h), amount=float(h)),
                "use_chip": "Chip Transaction",
            }
            for h in range(1, 41)
        ]
    )
    X = build_transaction_features(
        targets, HISTORY_SPEC, history=pd.DataFrame(history, columns=HISTORY_COLUMNS)
    )
    recipe = Recipe(
        estimator="logistic",
        parameters={"max_iter": 1000},
        feature_set="card",
        feature_spec=HISTORY_SPEC,
    )
    model = SklearnAdapter().fit(recipe, X, pd.Series([0, 1] * 20))
    release = save_release(
        tmp_path,
        model,
        {
            "recipe": recipe.model_dump(),
            "source_run_id": "fixture-run",
            "dataset": "ibm-tabformer",
            "source_import_id": "a" * 64,
            "threshold": 0.5,
            "promotion_eligible": False,
        },
        X.iloc[:3],
    )
    with pytest.raises(ValueError, match="card-history source"):
        FraudPredictor(root=tmp_path, release_id=release["release_id"])
    queries = []

    def lookup(**query):
        queries.append(query)
        return [
            row
            for row in history
            if query["start"] <= row["tx_datetime"] < query["end"]
        ]

    predictor = FraudPredictor(
        root=tmp_path, release_id=release["release_id"], history=lookup
    )
    inputs = {
        k: v for k, v in transaction(T, amount=3.0).items() if k in HISTORY_SPEC.inputs
    }
    inputs["timestamp"] = inputs.pop("tx_datetime")
    result = predictor.predict(**inputs, use_chip="Chip Transaction")
    assert queries == [
        {
            "import_id": "a" * 64,
            "user_id": 7,
            "card_id": 0,
            "start": T - timedelta(days=HISTORY_DAYS),
            "end": T,
        }
    ]
    expected = SklearnAdapter().scores(
        model,
        build_transaction_features(
            pd.DataFrame(
                [{**transaction(T, amount=3.0), "use_chip": "Chip Transaction"}]
            ),
            HISTORY_SPEC,
            history=pd.DataFrame(history, columns=HISTORY_COLUMNS),
        ),
    )[0]
    assert result["fraud_score"] == pytest.approx(expected)
    assert result["feature_version"] == "history-v1"
    with pytest.raises(
        ValueError, match="Missing required transaction fields: card_id"
    ):
        predictor.predict(
            **{k: v for k, v in inputs.items() if k != "card_id"},
            use_chip="Chip Transaction",
        )


@pytest.fixture
def history_snapshot(snapshot_job, dataset_profile):
    """Fixture data from 2019-01-01: 30 days of context before training starts."""
    profile = dataset_profile.model_copy(
        update={
            "train_start": date(2019, 1, 31),
            "context_days": HISTORY_DAYS,
            "final_holdout_start": date(2019, 3, 20),
        }
    )
    return snapshot_job(as_of="2019-03-10", profile=profile)


@pytest.mark.integration
def test_snapshot_features_match_source_database_lookups(
    history_snapshot, object_store, source_database, ibm_history
):
    with SnapshotReader(object_store, history_snapshot["snapshot_id"]) as reader:
        X, _ = feature_matrices(reader, ("validation",), feature_spec=HISTORY_SPEC)
        rows = pa.Table.from_batches(
            list(reader.iter_batches("validation"))
        ).to_pandas()
    trained = X["validation"][NAMES].to_numpy(dtype=float)
    assert trained[:, NAMES.index("card_confirmed_frauds_30d")].max() > 0
    for index in range(0, len(rows), 5):
        row = rows.iloc[index]
        history = pd.DataFrame(
            card_history(
                lambda: connect(source_database),
                import_id=ibm_history,
                user_id=int(row.user_id),
                card_id=int(row.card_id),
                start=row.tx_datetime - timedelta(days=HISTORY_DAYS),
                end=row.tx_datetime,
            ),
            columns=HISTORY_COLUMNS,
        )
        served = build_transaction_features(
            rows.iloc[[index]], HISTORY_SPEC, history=history
        )
        np.testing.assert_allclose(
            served[NAMES].to_numpy(dtype=float)[0], trained[index]
        )


@pytest.mark.integration
def test_history_model_trains_serves_and_joins_outcomes(
    tmp_path, history_snapshot, policy, database, source_database, monkeypatch
):
    data = policy.model_dump()
    data["feature_sets"] = {"card": HISTORY_SPEC.model_dump()}
    data["recipes"]["logistic_card"] = {
        "estimator": "logistic",
        "parameters": {"max_iter": 1000},
        "feature_set": "card",
    }
    data["initial_recipe"], data["challengers"] = "logistic_card", []
    report = run_cycle(
        root=tmp_path,
        snapshot=history_snapshot["snapshot_id"],
        policy=TrainingPolicy.model_validate(data),
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
    )
    trial = report["searches"]["logistic_card"]["trials"][0]
    assert trial["feature_version"] == "history-v1"
    assert report["release_id"], report["reason"]
    # Fixture source row 487: day 60, user 3 card 0, a fraud with 28 earlier released frauds.
    payload = {
        "transaction_id": 487,
        "amount": 1000.0,
        "timestamp": "2019-03-02T12:06:00",
        "use_chip": "Online Transaction",
        "merchant_id": -7,
        "mcc": "5411",
        "user_id": 3,
        "card_id": 0,
    }
    with TestClient(
        create_app(
            root=tmp_path,
            database_url=database,
            source_database_url=source_database,
            release_id=report["release_id"],
        )
    ) as client:
        assert client.get("/ready").status_code == 200
        missing = {k: v for k, v in payload.items() if k != "card_id"}
        assert client.post("/predict", json=missing).status_code == 422
        assert (
            client.post("/predict", json={**payload, "card_id": 1}).status_code == 409
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["feature_version"] == "history-v1"
    with connect(database) as conn:
        stored = conn.execute(
            "SELECT feature_inputs FROM predictions WHERE transaction_id=487"
        ).fetchone()["feature_inputs"]
    assert stored == {
        "user_id": 3,
        "card_id": 0,
        "use_chip": "Online Transaction",
        "merchant_id": -7,
        "mcc": "5411",
    }
    outcomes = query_outcomes(
        datetime(2019, 3, 5), database, source_url=source_database
    )
    assert [row["fraud"] for row in outcomes] == [1]
