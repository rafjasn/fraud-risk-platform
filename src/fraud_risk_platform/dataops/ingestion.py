"""Incremental IBM deliveries, immutable evidence, and accumulated S3 snapshots."""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pyarrow.compute as pc
import pyarrow.parquet as pq
from psycopg.types.json import Jsonb

from fraud_risk_platform.contracts.base import midnight
from fraud_risk_platform.dataops.history_store import HistoryStore
from fraud_risk_platform.dataops.reports.data import (
    REPORT_VERSION,
    publish_report,
    read_report,
    snapshot_report,
    snapshot_statistics,
)
from fraud_risk_platform.persistence.database import connect, connect_source
from fraud_risk_platform.persistence.files import (
    canonical,
    identity,
    sha256,
    write_json,
)

logger = logging.getLogger(__name__)


def source_import(url=None, import_id=None):
    with connect_source(url) as conn:
        rows = conn.execute(
            "SELECT * FROM ibm_source_imports WHERE status='complete'"
            + (" AND import_id=%s" if import_id else "")
            + " ORDER BY (customer_limit IS NOT NULL),completed_at DESC NULLS LAST,import_id LIMIT 1",
            (import_id,) if import_id else (),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                "No completed IBM source import; run make seed-source first"
            )
        result = rows[0]
        latest = conn.execute(
            "SELECT tx_datetime,label_available_at FROM ibm_transactions WHERE import_id=%s ORDER BY tx_datetime DESC,source_row DESC LIMIT 1",
            (result["import_id"],),
        ).fetchone()
        if not latest:
            raise ValueError("Completed source import has no transactions")
        return {
            "import_id": result["import_id"],
            "manifest": result["source_manifest"],
            "customer_limit": result["customer_limit"],
            "label_delay_days": result["label_delay_days"],
            "end_exclusive": (
                latest["label_available_at"].replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                + timedelta(days=1)
            ).isoformat(),
        }


def extract(source, *, start, previous, cutoff, chunk_size, url):
    with connect_source(url) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        with conn.cursor(name="ibm_delivery") as cursor:
            query = """SELECT source_row,user_id,card_id,tx_datetime,amount,use_chip,
                merchant_id,merchant_city,merchant_state,postal_code,mcc,errors,
                CASE WHEN label_available_at<=%(cutoff)s THEN is_fraud ELSE NULL END AS is_fraud,
                label_available_at FROM ibm_complete_transactions
                WHERE import_id=%(import_id)s AND tx_datetime>=%(start)s AND tx_datetime<%(cutoff)s"""
            if previous is not None:
                query += " AND (tx_datetime>=%(previous)s OR (label_available_at>%(previous)s AND label_available_at<=%(cutoff)s))"
            query += " ORDER BY tx_datetime,source_row"
            cursor.execute(
                query,
                {
                    "cutoff": cutoff,
                    "start": start,
                    "previous": previous,
                    "import_id": source["import_id"],
                },
            )
            while records := cursor.fetchmany(chunk_size):
                yield [
                    {
                        key: value.isoformat()
                        if isinstance(value, datetime)
                        else str(value)
                        if isinstance(value, Decimal)
                        else value
                        for key, value in row.items()
                    }
                    for row in records
                ]


def corrupt(records, fault):
    if not records or not fault:
        return records
    records = [dict(row) for row in records]
    if fault == "missing-amount":
        records[0].pop("amount")
    elif fault == "invalid-amount":
        records[0]["amount"] = "broken"
    elif fault == "duplicate":
        records.append(dict(records[0]))
    elif fault == "conflict":
        records.append({**records[0], "amount": "999999.00"})
    else:
        raise ValueError("Unknown corruption scenario")
    return records


def raw_files(store, files, directory):
    """Verified local copies of delivery parts, in order; each is read before the next."""
    for file in files:
        path = directory / "raw.jsonl"
        store.verify(file, path)
        yield path


def restore(context, store, batches, directory, *, start, delay):
    previous = None
    for batch in batches:
        manifest = batch["manifest"]
        if batch["previous_cutoff"] != previous:
            raise ValueError("Accepted delivery chain is incomplete")
        if store.read_json(manifest["manifest_key"]) != manifest:
            raise ValueError(
                "Accepted batch manifest is missing or changed; rebuild the disposable simulation"
            )
        result = context.apply(
            raw_files(store, manifest["files"], directory),
            start=start,
            cutoff=batch["as_of"],
            label_delay_days=delay,
            reporting=False,
        )
        if result["invalid"] or result["received"] != manifest["received"]:
            raise ValueError(
                "Previously accepted delivery failed integrity/semantic verification"
            )
        previous = batch["as_of"]


@contextmanager
def stream_lock(platform_url, configuration):
    stream_id = identity(configuration)
    with connect(platform_url) as conn:
        conn.autocommit = True
        lock_id = int(stream_id[:15], 16)
        locked = conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS locked", (lock_id,)
        ).fetchone()["locked"]
        if not locked:
            raise ValueError(
                "Another job is processing this ingestion stream; retry later"
            )
        try:
            conn.execute(
                "INSERT INTO ingestion_streams(stream_id,configuration) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (stream_id, Jsonb(configuration)),
            )
            yield conn, stream_id
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))


def ingest_delivery(
    conn,
    stream_id,
    context,
    store,
    directory,
    source,
    *,
    start,
    previous,
    cutoff,
    chunk_size,
    source_url,
    fault,
):
    attempt_id = uuid4().hex
    prefix = f"batches/{stream_id}/{attempt_id}"
    report_prefix = f"reports/ingestion/{attempt_id}"
    conn.execute(
        "INSERT INTO ingestion_attempts(attempt_id,stream_id,as_of,status) VALUES (%s,%s,%s,'running')",
        (attempt_id, stream_id, cutoff),
    )
    report = {
        "report_version": REPORT_VERSION,
        "kind": "Ingestion",
        "status": "running",
        "context_start": start.isoformat(),
        "previous_cutoff": previous.isoformat() if previous else None,
        "source": source,
        "as_of": cutoff.isoformat(),
        "attempt_id": attempt_id,
        "stream_id": stream_id,
        "fault": fault,
        "population": "valid records in this delivery, including delayed label updates",
    }
    files = []
    extracted_records = 0
    rejection_recorded = False
    try:
        for index, records in enumerate(
            extract(
                source,
                start=start,
                previous=previous,
                cutoff=cutoff,
                chunk_size=chunk_size,
                url=source_url,
            )
        ):
            extracted_records += len(records)
            if index == 0:
                records = corrupt(records, fault)
            path = directory / "delivery.jsonl"
            with path.open("wb") as stream:
                for record in records:
                    stream.write(canonical(record) + b"\n")
            files.append(
                {
                    **store.put_file(
                        f"{prefix}/part-{index:06}.jsonl", path, "application/x-ndjson"
                    ),
                    "rows": len(records),
                }
            )
        store.put_json(
            f"{prefix}/raw-manifest.json",
            {
                "files": files,
                "as_of": cutoff.isoformat(),
                "source": source,
                "fault": fault,
            },
        )
        findings = context.apply(
            raw_files(store, files, directory),
            start=start,
            cutoff=cutoff,
            label_delay_days=source["label_delay_days"],
        )
        report.update(
            findings,
            status="rejected" if findings["invalid"] else "accepted",
            extracted_records=extracted_records,
            accepted_records=0 if findings["invalid"] else findings["valid"],
        )
        reports = publish_report(store, report_prefix, report, directory)
        if findings["invalid"]:
            store.put_json(
                f"quarantine/{attempt_id}.json",
                {
                    "raw_files": files,
                    "reports": reports,
                    "policy": "reject-whole-batch",
                },
            )
            conn.execute(
                "UPDATE ingestion_attempts SET status='rejected',report_key=%s,finished_at=now() WHERE attempt_id=%s",
                (reports["json"]["key"], attempt_id),
            )
            rejection_recorded = True
            raise ValueError(
                f"Delivery rejected: {findings['invalid']} invalid records; report: {store.uri(reports['html']['key'])}"
            )
        manifest = {
            "schema_version": 1,
            "status": "accepted",
            "stream_id": stream_id,
            "as_of": cutoff.isoformat(),
            "previous_cutoff": previous.isoformat() if previous else None,
            "source": source,
            "files": files,
            "received": findings["received"],
            "reports": reports,
            "manifest_key": f"{prefix}/manifest.json",
        }
        batch_id = identity(manifest)
        store.put_json(manifest["manifest_key"], manifest)
        # Artifacts are complete before the cursor and accepted ledger advance atomically.
        with conn.transaction():
            conn.execute(
                "INSERT INTO ingestion_batches VALUES (%s,%s,%s,%s,%s)",
                (batch_id, stream_id, previous, cutoff, Jsonb(manifest)),
            )
            conn.execute(
                "UPDATE ingestion_streams SET accepted_cutoff=%s WHERE stream_id=%s",
                (cutoff, stream_id),
            )
            conn.execute(
                "UPDATE ingestion_attempts SET status='accepted',report_key=%s,finished_at=now() WHERE attempt_id=%s",
                (reports["json"]["key"], attempt_id),
            )
        return {
            "batch_id": batch_id,
            "previous_cutoff": previous,
            "as_of": cutoff,
            "manifest": manifest,
        }
    except Exception as error:
        if not rejection_recorded:
            report.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                partial_files=files,
            )
            write_json(directory / "failure.json", report)
            try:
                store.put_json(f"{report_prefix}/failure.json", report)
            except Exception:
                logger.exception(
                    "Could not publish failure report; see job logs and ingestion_attempts"
                )
            conn.execute(
                "UPDATE ingestion_attempts SET status='failed',error=%s,finished_at=now() WHERE attempt_id=%s",
                (str(error), attempt_id),
            )
        raise


def publish_snapshot(context, store, directory, source, profile, cutoff, batches):
    configuration = profile.model_dump(mode="json")
    request = {
        "dataset_profile": configuration,
        "as_of": cutoff.isoformat(),
        "batch_ids": [b["batch_id"] for b in batches],
    }
    request_key = f"snapshot-requests/{identity(request)}.json"
    if store.exists(request_key):
        from fraud_risk_platform.dataops.reader import SnapshotReader

        with SnapshotReader(
            store, store.read_json(request_key)["snapshot_id"]
        ) as reader:
            reader.verify()
            return reader.manifest
    windows = profile.windows(cutoff.date().isoformat(), source["label_delay_days"])
    context.select(profile, windows, source["manifest"]["sha256"])
    files, counts = {}, {}
    last_batch = batches[-1]
    delivery = read_report(store, last_batch["manifest"]["reports"]["json"], directory)
    delivery = {
        key: value
        for key, value in delivery.items()
        if key not in ("summary", "new_transaction_summary", "source")
    }
    delivery["batch_id"] = last_batch["batch_id"]
    delivery["previous_cutoff"] = (
        last_batch["previous_cutoff"].isoformat()
        if last_batch["previous_cutoff"]
        else None
    )
    statistics = snapshot_statistics(
        context.db,
        profile=profile,
        cutoff=cutoff,
        windows=windows,
        previous=last_batch["previous_cutoff"],
    )
    for split in ("context", "train", "validation", "evaluation"):
        descriptors = []
        rows = frauds = pending = 0
        for table in context.rows(split, windows):
            path = directory / f"{split}.parquet"
            pq.write_table(table, path, compression="zstd", row_group_size=10_000)
            digest = sha256(path)
            descriptors.append(
                {
                    **store.put_file(f"objects/{digest}.parquet", path),
                    "rows": table.num_rows,
                }
            )
            rows += table.num_rows
            frauds += pc.sum(table["is_fraud"]).as_py() or 0
            pending += table["is_fraud"].null_count
        files[split] = descriptors
        counts[split] = {"rows": rows, "frauds": frauds, "unavailable_labels": pending}
    report = snapshot_report(
        statistics,
        profile=profile,
        cutoff=cutoff,
        windows=windows,
        splits=counts,
        source=source,
        batch_ids=request["batch_ids"],
        delivery=delivery,
    )
    eligible = report["eligible_training_rows"]
    if eligible != context.count_eligible(windows["train"]):
        raise ValueError("Report training population disagrees with selection")
    reports = publish_report(
        store,
        f"reports/snapshots/{identity(request)}/v{REPORT_VERSION}",
        report,
        directory,
    )
    manifest = {
        "schema_version": 2,
        "dataset": "ibm-tabformer",
        "record_version": "ibm-observation-v1",
        "status": "complete",
        "as_of": cutoff.isoformat(),
        "timezone": "naive source timestamps interpreted as UTC",
        "context_start": profile.context_start.isoformat(),
        "source": source,
        "dataset_profile": configuration,
        "profile_sha256": identity(configuration),
        "label_delay_days": source["label_delay_days"],
        "windows": windows,
        "selection": {
            "method": "sha256-source-seed-row-v1",
            "eligible_rows": eligible,
            "selected_rows": counts["train"]["rows"],
        },
        "batch_ids": request["batch_ids"],
        "files": files,
        "splits": counts,
        "reports": reports,
    }
    manifest["snapshot_id"] = identity(manifest)
    store.put_json(f"snapshots/{manifest['snapshot_id']}/manifest.json", manifest)
    store.put_json(request_key, {"snapshot_id": manifest["snapshot_id"]})
    return manifest


def create_snapshot(
    *,
    store,
    profile,
    as_of,
    source_url=None,
    platform_url=None,
    import_id=None,
    simulation="default",
    chunk_size=50_000,
    fault=None,
    work_dir=None,
):
    if not 1 <= chunk_size <= 100_000:
        raise ValueError("Extraction chunk size must be between 1 and 100000")
    source = source_import(source_url, import_id)
    cutoff = midnight(as_of)
    profile.windows(as_of, source["label_delay_days"])
    if cutoff > datetime.fromisoformat(source["end_exclusive"]):
        raise ValueError("Cutoff exceeds the source transaction/label timeline")
    configuration = {
        "version": 1,
        "source": source,
        "context_start": profile.context_start.isoformat(),
        "simulation": simulation,
    }
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with (
        stream_lock(platform_url, configuration) as (conn, stream_id),
        TemporaryDirectory(prefix="ibm-snapshot-", dir=work_dir) as temp,
    ):
        directory = Path(temp)
        context = HistoryStore(directory / "history.duckdb")
        try:
            previous = conn.execute(
                "SELECT accepted_cutoff FROM ingestion_streams WHERE stream_id=%s",
                (stream_id,),
            ).fetchone()["accepted_cutoff"]
            batches = conn.execute(
                "SELECT * FROM ingestion_batches WHERE stream_id=%s AND as_of<=%s ORDER BY as_of",
                (stream_id, cutoff),
            ).fetchall()
            if previous and cutoff >= previous:
                if not batches or batches[-1]["as_of"] != previous:
                    raise ValueError("Accepted delivery ledger and cursor do not agree")
            restore(
                context,
                store,
                batches,
                directory,
                start=profile.context_start,
                delay=source["label_delay_days"],
            )
            if not batches or batches[-1]["as_of"] != cutoff:
                if previous and cutoff <= previous:
                    raise ValueError(
                        "This cutoff was not accepted before the stream advanced; use a new --simulation for a backfill"
                    )
                batches.append(
                    ingest_delivery(
                        conn,
                        stream_id,
                        context,
                        store,
                        directory,
                        source,
                        start=profile.context_start,
                        previous=previous,
                        cutoff=cutoff,
                        chunk_size=chunk_size,
                        source_url=source_url,
                        fault=fault,
                    )
                )
            elif fault:
                raise ValueError(
                    "This delivery is already accepted; inject a fault at a new cutoff or in a new simulation"
                )
            return publish_snapshot(
                context, store, directory, source, profile, cutoff, batches
            )
        finally:
            context.close()
