"""Smoke-check the installed ingestion image, without databases or model extras."""

import importlib
import io
from datetime import datetime, timedelta
from decimal import Decimal
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from fraud_risk_platform.cli import parser
from fraud_risk_platform.dataops.dataset import Observation, arrow_schema
from fraud_risk_platform.dataops.history_store import HistoryStore
from fraud_risk_platform.dataops.reports.data import render_report
from fraud_risk_platform.persistence.files import canonical


def main():
    packages = {d.metadata["Name"].lower().replace("_", "-") for d in distributions()}
    forbidden = {
        p
        for p in packages
        if p in {"torch", "triton", "scikit-learn", "joblib"}
        or p.startswith(("mlflow", "nvidia-", "cuda-"))
    }
    assert not forbidden, (
        f"Model dependencies leaked into ingestion: {sorted(forbidden)}"
    )
    for module in (
        "seeding.ibm",
        "dataops.workflow",
        "dataops.ingestion",
        "dataops.reader",
        "persistence.database",
    ):
        importlib.import_module(f"fraud_risk_platform.{module}")
    for command in ("init-db", "seed-source", "initial-snapshot", "next-snapshot"):
        assert parser().parse_args([command]).command == command
    timestamp = datetime(2019, 1, 1, 12)
    row = Observation(
        source_row=1,
        user_id=1,
        card_id=0,
        tx_datetime=timestamp,
        amount=Decimal("-10.50"),
        use_chip="Online Transaction",
        merchant_id=7,
        merchant_city="ONLINE",
        merchant_state=None,
        postal_code=None,
        mcc="5411",
        errors=None,
        is_fraud=False,
        label_available_at=timestamp + timedelta(days=7),
    )
    buffer = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist([row.model_dump()], schema=arrow_schema()), buffer
    )
    buffer.seek(0)
    assert Observation.model_validate(pq.read_table(buffer).to_pylist()[0]) == row
    with TemporaryDirectory() as temp:
        delivery = Path(temp) / "raw.jsonl"
        delivery.write_bytes(canonical(row.model_dump(mode="json")) + b"\n")
        context = HistoryStore(Path(temp) / "history.duckdb")
        try:
            findings = context.apply(
                [delivery],
                start=timestamp,
                cutoff=timestamp + timedelta(days=8),
                label_delay_days=7,
            )
            assert findings["valid"] == 1 and not findings["violations"], findings
            (table,) = context.rows()
            assert table.schema.equals(arrow_schema()) and table.num_rows == 1
        finally:
            context.close()
    report = render_report(
        {"kind": "Ingestion", "status": "accepted", "summary": findings["summary"]}
    )
    assert "data:image/png;base64" in report
    print(
        "Ingestion image passed: CLI/imports, DuckDB validation, Parquet round trip, "
        "embedded report charts; no model dependencies."
    )


if __name__ == "__main__":
    main()
