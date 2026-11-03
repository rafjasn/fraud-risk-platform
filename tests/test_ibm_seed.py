import csv
import io
import json
import sys
import tarfile
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from fraud_risk_platform.cli import main
from fraud_risk_platform.persistence.database import connect
from fraud_risk_platform.persistence.files import sha256, write_json
from fraud_risk_platform.seeding import ibm


@pytest.fixture
def ibm_source(tmp_path):
    def make(rows=None, header=None):
        records = (
            rows
            if rows is not None
            else [
                [
                    "0",
                    "0",
                    "2019",
                    "1",
                    "1",
                    "06:21",
                    "$134.09",
                    "Swipe Transaction",
                    "3527213246127876953",
                    "La Verne",
                    "CA",
                    "91750.0",
                    "5300",
                    "",
                    "No",
                ],
                [
                    "0",
                    "1",
                    "2019",
                    "2",
                    "1",
                    "12:00",
                    "$-61.00",
                    "Online Transaction",
                    "-727612092139916043",
                    "ONLINE",
                    "",
                    "",
                    "5411",
                    "Insufficient Balance,",
                    "Yes",
                ],
                [
                    "1",
                    "0",
                    "2020",
                    "2",
                    "29",
                    "23:59",
                    "$0.00",
                    "Chip Transaction",
                    "3",
                    "Paris",
                    "France",
                    "",
                    "5411",
                    "Bad PIN,Technical Glitch",
                    "No",
                ],
                [
                    "2",
                    "0",
                    "2018",
                    "12",
                    "31",
                    "00:00",
                    "$10.10",
                    "Swipe Transaction",
                    "4",
                    "City",
                    "ST",
                    "00123.0",
                    "5300",
                    "",
                    "No",
                ],
                [
                    "0",
                    "0",
                    "2020",
                    "3",
                    "1",
                    "01:02",
                    "$9.99",
                    "Swipe Transaction",
                    "4",
                    "City",
                    "ST",
                    "00123.0",
                    "5300",
                    "",
                    "No",
                ],
            ]
        )
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(ibm.HEADER if header is None else header)
        writer.writerows(records)
        data = stream.getvalue().encode()
        archive = tmp_path / "transactions.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            member = tarfile.TarInfo("card_transaction.v1.csv")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        manifest = {
            "schema_version": 1,
            "dataset": "ibm-tabformer",
            "revision": "a" * 40,
            "url": "https://example.invalid/transactions.tgz",
            "filename": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "csv_member": "card_transaction.v1.csv",
            "csv_bytes": len(data),
            "row_count": len(records),
        }
        lock = tmp_path / "ibm.lock.json"
        write_json(lock, manifest)
        return lock, archive, records

    return make


def seed_args(ibm_source, tmp_path, source_database, **overrides):
    lock, archive, _ = ibm_source()
    return {
        "source_lock": lock,
        "raw_dir": tmp_path / "raw",
        "archive": archive,
        "label_delay_days": 7,
        "chunk_size": 2,
        "url": source_database,
        **overrides,
    }


@pytest.mark.integration
def test_ibm_types_provenance_and_repeatable_import(
    ibm_source, tmp_path, source_database
):
    args = seed_args(ibm_source, tmp_path, source_database)
    result = ibm.seed_ibm(**args)
    assert result["status"] == "complete"
    assert (
        result["inserted_rows"]
        == result["imported_rows"]
        == result["scanned_rows"]
        == 5
    )
    assert result["frauds"] == 1
    repeated = ibm.seed_ibm(**{**args, "chunk_size": 3})
    assert repeated["import_id"] == result["import_id"]
    assert repeated["inserted_rows"] == 0
    with connect(source_database) as conn:
        rows = conn.execute(
            "SELECT * FROM ibm_complete_transactions ORDER BY source_row"
        ).fetchall()
        assert rows[1]["amount"] == Decimal("-61.00")
        assert rows[1]["merchant_id"] == -727612092139916043
        assert rows[1]["merchant_state"] is None and rows[1]["postal_code"] is None
        assert rows[1]["errors"] == "Insufficient Balance,"
        assert rows[2]["tx_datetime"] == datetime(2020, 2, 29, 23, 59)
        assert rows[3]["postal_code"] == "00123.0"
        assert all(
            row["label_available_at"] == row["tx_datetime"] + timedelta(days=7)
            for row in rows
        )
        state = conn.execute("SELECT * FROM ibm_source_imports").fetchone()
        assert state["source_manifest"]["sha256"] == sha256(args["archive"])
        assert (
            conn.execute("SELECT to_regclass('transactions') AS name").fetchone()[
                "name"
            ]
            is None
        )


@pytest.mark.integration
def test_ibm_customer_selection_keeps_entire_histories(
    ibm_source, tmp_path, source_database
):
    args = seed_args(ibm_source, tmp_path, source_database, customer_limit=1)
    result = ibm.seed_ibm(**args)
    assert result["scanned_rows"] == 5 and result["imported_rows"] == 3
    with connect(source_database) as conn:
        rows = conn.execute(
            "SELECT source_row,user_id FROM ibm_complete_transactions ORDER BY source_row"
        ).fetchall()
    assert [row["source_row"] for row in rows] == [1, 2, 5]
    assert {row["user_id"] for row in rows} == {0}
    different = ibm.seed_ibm(**{**args, "label_delay_days": 2})
    assert different["import_id"] != result["import_id"]


@pytest.mark.integration
def test_interruption_is_hidden_and_resumes_without_duplicates(
    ibm_source, tmp_path, source_database, monkeypatch
):
    args = seed_args(ibm_source, tmp_path, source_database)
    real_write = ibm._write_chunk

    def interrupted(conn, import_id, scanned_rows, rows):
        if scanned_rows > 2:
            raise RuntimeError("simulated worker failure")
        return real_write(conn, import_id, scanned_rows, rows)

    monkeypatch.setattr(ibm, "_write_chunk", interrupted)
    with pytest.raises(RuntimeError, match="worker failure"):
        ibm.seed_ibm(**args)
    with connect(source_database) as conn:
        state = conn.execute("SELECT * FROM ibm_source_imports").fetchone()
        assert (
            state["status"] == "failed"
            and state["scanned_rows"] == state["imported_rows"] == 2
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM ibm_complete_transactions"
            ).fetchone()["n"]
            == 0
        )
    monkeypatch.setattr(ibm, "_write_chunk", real_write)
    result = ibm.seed_ibm(**{**args, "chunk_size": 3})
    assert result["inserted_rows"] == 3 and result["imported_rows"] == 5
    assert ibm.seed_ibm(**args)["inserted_rows"] == 0


@pytest.mark.integration
def test_invalid_row_rejects_chunk_and_keeps_import_hidden(
    ibm_source, tmp_path, source_database
):
    _, _, rows = ibm_source()
    rows[3][14] = "Unknown"
    lock, archive, _ = ibm_source(rows)
    with pytest.raises(ValueError, match="source row 4"):
        ibm.seed_ibm(
            source_lock=lock,
            archive=archive,
            raw_dir=tmp_path,
            label_delay_days=7,
            chunk_size=2,
            url=source_database,
        )
    with connect(source_database) as conn:
        state = conn.execute(
            "SELECT status,scanned_rows,imported_rows FROM ibm_source_imports"
        ).fetchone()
        assert state == {"status": "failed", "scanned_rows": 2, "imported_rows": 2}
        assert (
            conn.execute("SELECT count(*) AS n FROM ibm_transactions").fetchone()["n"]
            == 2
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM ibm_complete_transactions"
            ).fetchone()["n"]
            == 0
        )


def test_bad_archive_is_rejected_before_database_access(
    ibm_source, tmp_path, monkeypatch
):
    lock, archive, _ = ibm_source()
    corrupted = bytearray(archive.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    archive.write_bytes(corrupted)  # Same size: this must fail the checksum check.
    monkeypatch.delenv("SOURCE_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="checksum/size"):
        ibm.seed_ibm(
            source_lock=lock, archive=archive, raw_dir=tmp_path, label_delay_days=7
        )


def test_header_is_checked_before_database_access(ibm_source, tmp_path):
    lock, archive, _ = ibm_source(header=[*ibm.HEADER[:-1], "wrong label"])
    with pytest.raises(ValueError, match="header"):
        ibm.seed_ibm(
            source_lock=lock, archive=archive, raw_dir=tmp_path, label_delay_days=7
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("Amount", "$NaN"),
        ("Amount", "$1.234"),
        ("User", "1.2"),
        ("Time", "24:00"),
        ("Month", "13"),
        ("Use Chip", "unknown"),
        ("Is Fraud?", ""),
        ("MCC", "abcd"),
    ],
)
def test_invalid_typed_values_are_not_silently_coerced(ibm_source, column, value):
    _, _, rows = ibm_source()
    rows[0][ibm.HEADER.index(column)] = value
    with pytest.raises(ValueError, match="source row 1"):
        ibm.parse_transaction(rows[0], 1, 7)


def test_verified_download_cache_and_corruption(ibm_source, tmp_path, monkeypatch):
    lock, archive, _ = ibm_source()
    manifest = ibm.load_manifest(lock)
    calls = []

    def download(request, timeout):
        calls.append(request.full_url)
        return archive.open("rb")

    monkeypatch.setattr(ibm, "urlopen", download)
    destination = ibm.prepare_archive(manifest, tmp_path / "cache")
    assert destination.read_bytes() == archive.read_bytes()
    assert ibm.prepare_archive(manifest, tmp_path / "cache") == destination
    assert len(calls) == 1
    destination.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="checksum/size"):
        ibm.prepare_archive(manifest, tmp_path / "cache")
    assert destination.read_bytes() == b"corrupt cache" and len(calls) == 1


@pytest.mark.integration
def test_cli_runs_real_ibm_seeding(
    ibm_source, tmp_path, source_database, monkeypatch, capsys
):
    lock, archive, _ = ibm_source()
    monkeypatch.setenv("SOURCE_DATABASE_URL", source_database)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraud-risk-platform",
            "seed-source",
            "--source-lock",
            str(lock),
            "--archive",
            str(archive),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--customer-limit",
            "1",
            "--chunk-size",
            "2",
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["dataset"] == "ibm-tabformer" and report["imported_rows"] == 3


@pytest.mark.integration
def test_completed_import_with_missing_rows_is_hidden(
    ibm_source, tmp_path, source_database
):
    args = seed_args(ibm_source, tmp_path, source_database)
    result = ibm.seed_ibm(**args)
    with connect(source_database) as conn:
        conn.execute(
            "DELETE FROM ibm_transactions WHERE import_id=%s AND source_row=1",
            (result["import_id"],),
        )
    with pytest.raises(ValueError, match="row count differs"):
        ibm.seed_ibm(**args)
    with connect(source_database) as conn:
        assert (
            conn.execute("SELECT status FROM ibm_source_imports").fetchone()["status"]
            == "failed"
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM ibm_complete_transactions"
            ).fetchone()["n"]
            == 0
        )


@pytest.mark.integration
def test_second_worker_cannot_take_the_same_import(
    ibm_source, tmp_path, source_database
):
    args = seed_args(ibm_source, tmp_path, source_database)
    result = ibm.seed_ibm(**args)
    with connect(source_database) as conn:
        conn.execute(
            "SELECT pg_advisory_lock(%s)", (int(result["import_id"][:15], 16),)
        )
        with pytest.raises(ValueError, match="already running"):
            ibm.seed_ibm(**args)
