import pytest

from fraud_risk_platform.persistence.database import (
    connect,
    connect_source,
    initialize,
)


@pytest.mark.integration
def test_environment_routes_to_separate_databases(
    database, source_database, monkeypatch, ibm_history, snapshot_job
):
    monkeypatch.setenv("DATABASE_URL", database)
    monkeypatch.setenv("SOURCE_DATABASE_URL", source_database)
    initialize()
    initialize()
    assert snapshot_job()["splits"]["context"]["rows"] == 320
    with connect() as platform, connect_source() as source:
        platform_name = platform.execute("SELECT current_database() AS name").fetchone()
        source_name = source.execute("SELECT current_database() AS name").fetchone()
        assert platform_name != source_name
        assert (
            platform.execute("SELECT count(*) AS n FROM predictions").fetchone()["n"]
            == 0
        )
        assert (
            source.execute("SELECT count(*) AS n FROM ibm_transactions").fetchone()["n"]
            == 80 * 8
        )
        assert (
            platform.execute(
                "SELECT to_regclass('ibm_transactions') AS name"
            ).fetchone()["name"]
            is None
        )
        assert (
            source.execute("SELECT to_regclass('predictions') AS name").fetchone()[
                "name"
            ]
            is None
        )
        for connection in (platform, source):
            assert (
                connection.execute(
                    "SELECT 1 FROM pg_namespace WHERE nspname = 'mlflow'"
                ).fetchone()
                is None
            )
