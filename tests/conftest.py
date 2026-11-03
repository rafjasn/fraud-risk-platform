import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import boto3
import pandas as pd
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from fraud_risk_platform.contracts.features import (
    HISTORY_DAYS,
    FeatureSet,
    basic_feature_set,
)
from fraud_risk_platform.contracts.training import TrainingPolicy
from fraud_risk_platform.dataops.dataset import DatasetProfile, DataSettings
from fraud_risk_platform.dataops.ingestion import create_snapshot
from fraud_risk_platform.dataops.workflow import monthly_snapshot
from fraud_risk_platform.features.transaction import build_transaction_features
from fraud_risk_platform.models.recipes import build_model
from fraud_risk_platform.models.releases import save_release
from fraud_risk_platform.persistence.database import connect, initialize
from fraud_risk_platform.persistence.files import sha256, write_json
from fraud_risk_platform.persistence.objects import ObjectStore


@pytest.fixture
def policy():
    return TrainingPolicy.model_validate(
        {
            # Synthetic fixtures are 25% fraud; the budget must cover a perfect ranking.
            "review_budget": 0.3,
            "minimum_recall_gain": 0.0,
            "minimum_rows_per_split": 2,
            "minimum_frauds_per_split": 1,
            "initial_recipe": "logistic_basic",
            "challengers": ["boosting_basic"],
            "recipes": {
                "logistic_basic": {
                    "estimator": "logistic",
                    "parameters": {"max_iter": 1000, "random_state": 42},
                },
                "boosting_basic": {
                    "estimator": "hist_gradient_boosting",
                    "parameters": {
                        "max_iter": 5,
                        "min_samples_leaf": 2,
                        "early_stopping": False,
                        "random_state": 42,
                    },
                },
            },
        }
    )


@pytest.fixture
def databases():
    base = os.environ.get("TEST_DATABASE_URL")
    if not base:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL database")
    suffix = uuid4().hex
    names = {"platform": "test_platform_" + suffix, "source": "test_source_" + suffix}
    urls = {name: make_conninfo(base, dbname=db) for name, db in names.items()}
    created = []
    with psycopg.connect(base, autocommit=True) as conn:
        try:
            for name in names.values():
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
                created.append(name)
            initialize(urls["platform"], source_url=urls["source"])
            yield urls
        finally:
            for name in reversed(created):
                conn.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                        sql.Identifier(name)
                    )
                )


@pytest.fixture
def database(databases):
    return databases["platform"]


@pytest.fixture
def source_database(databases):
    return databases["source"]


@pytest.fixture
def release_factory(tmp_path, policy):
    frame = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 500.0, 600.0],
            "tx_datetime": pd.to_datetime(["2018-04-01T00:00:00"] * 4),
        }
    )
    X = build_transaction_features(frame, basic_feature_set())
    model = build_model(policy.recipes["logistic_basic"])
    model.fit(X, [0, 0, 1, 1])

    def make(*, baseline=None, eligible=True, threshold=0.5):
        run_id = uuid4().hex
        manifest = save_release(
            tmp_path,
            model,
            {
                "source_run_id": run_id,
                "snapshot_id": "fixture",
                "dataset": "ibm-tabformer",
                "source_import_id": "a" * 64,
                "as_of": "2018-05-01",
                "training_end_exclusive": "2018-04-20T00:00:00",
                "information_end_exclusive": "2018-04-29T00:00:00",
                "recipe_name": "logistic_basic",
                "recipe": policy.recipes["logistic_basic"].model_dump(),
                "threshold": threshold,
                "incumbent_release_id": baseline,
                "promotion_eligible": eligible,
                "recommendation": "promote" if baseline else "initial_promotion",
            },
            X,
        )
        directory = tmp_path / "cycles" / run_id
        directory.mkdir(parents=True)
        write_json(
            directory / "report.json",
            {
                "release_id": manifest["release_id"],
                "recommendation": manifest["recommendation"],
            },
        )
        write_json(
            directory / "complete.json",
            {"report_sha256": sha256(directory / "report.json")},
        )
        return manifest

    return make


@pytest.fixture
def object_store(monkeypatch):
    endpoint = os.environ.get("TEST_S3_ENDPOINT_URL")
    if not endpoint:
        pytest.skip("Set TEST_S3_ENDPOINT_URL to disposable LocalStack S3")
    bucket = "fraud-test-" + uuid4().hex
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("FRAUD_DATA_URI", f"s3://{bucket}")
    client = boto3.client("s3", endpoint_url=endpoint)
    client.create_bucket(Bucket=bucket)
    try:
        yield ObjectStore(client=client)
    finally:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            keys = [{"Key": row["Key"]} for row in page.get("Contents", [])]
            if keys:
                client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        client.delete_bucket(Bucket=bucket)


@pytest.fixture
def ibm_history(source_database):
    import_id = "a" * 64
    start = datetime(2019, 1, 1, 12)
    count = 80 * 8
    manifest = {
        "dataset": "ibm-tabformer",
        "sha256": "b" * 64,
        "revision": "c" * 40,
        "row_count": count,
    }
    with connect(source_database) as conn:
        conn.execute(
            """INSERT INTO ibm_source_imports
            (import_id,source_manifest,adapter_version,label_delay_days,status,scanned_rows,imported_rows)
            VALUES (%s,%s,1,2,'complete',%s,%s)""",
            (import_id, Jsonb(manifest), count, count),
        )
        with conn.cursor().copy("""COPY ibm_transactions
            (import_id,source_row,user_id,card_id,tx_datetime,amount,use_chip,merchant_id,
             merchant_city,merchant_state,postal_code,mcc,errors,is_fraud,label_available_at)
             FROM STDIN""") as copy:
            for day in range(80):
                for sample in range(8):
                    timestamp = start + timedelta(days=day, minutes=sample)
                    fraud = sample >= 6
                    copy.write_row(
                        (
                            import_id,
                            day * 8 + sample + 1,
                            sample // 2,
                            sample % 2,
                            timestamp,
                            Decimal("1000.00")
                            if fraud
                            else Decimal("-10.50")
                            if sample == 0
                            else Decimal("10.50"),
                            "Online Transaction",
                            -7,
                            "ONLINE",
                            None,
                            None,
                            "5411",
                            None,
                            fraud,
                            timestamp + timedelta(days=2),
                        )
                    )
    return import_id


@pytest.fixture
def dataset_profile():
    return DatasetProfile(
        train_start="2019-01-02",
        context_days=1,
        validation_days=7,
        evaluation_days=7,
        final_holdout_start="2019-03-01",
        max_train_rows=64,
        minimum_rows_per_split=8,
        minimum_frauds_per_split=1,
    )


@pytest.fixture
def snapshot_job(object_store, ibm_history, dataset_profile, databases, tmp_path):
    def run(**overrides):
        return create_snapshot(
            **{
                "store": object_store,
                "profile": dataset_profile,
                "as_of": "2019-02-10",
                "source_url": databases["source"],
                "platform_url": databases["platform"],
                "import_id": ibm_history,
                "chunk_size": 23,
                "work_dir": tmp_path,
                **overrides,
            }
        )

    return run


SPEC = FeatureSet(
    version="history-v1",
    numeric=["amount", "card_count_24h", "card_confirmed_frauds_30d"],
    categorical=["use_chip"],
)


@pytest.fixture
def simulate(object_store, ibm_history, dataset_profile, databases, tmp_path):
    """Fixture traffic from 2019-01-01, eight transactions a day, two of them fraud."""

    def setup(final_holdout_start, initial_snapshot=date(2019, 3, 10)):
        settings = DataSettings(
            initial_snapshot=initial_snapshot,
            label_delay_days=2,
            dataset=dataset_profile.model_copy(
                update={
                    "train_start": date(2019, 1, 31),
                    "context_days": HISTORY_DAYS,
                    "final_holdout_start": final_holdout_start,
                }
            ),
        )
        urls = {
            "source_url": databases["source"],
            "platform_url": databases["platform"],
        }

        def advance(initial=False):
            return monthly_snapshot(
                initial=initial,
                settings=settings,
                store=object_store,
                work_dir=tmp_path / "work",
                **urls,
            )

        return settings, urls, advance

    return setup


@pytest.fixture
def card_policy(policy):
    """One logistic recipe on card-history features."""
    data = policy.model_dump()
    data["feature_sets"] = {"card": SPEC.model_dump()}
    data["recipes"]["logistic_card"] = {
        "estimator": "logistic",
        "parameters": {"max_iter": 1000},
        "feature_set": "card",
    }
    data["initial_recipe"], data["challengers"] = "logistic_card", []
    return TrainingPolicy.model_validate(data)
