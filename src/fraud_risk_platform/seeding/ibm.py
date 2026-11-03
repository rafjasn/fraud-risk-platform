"""Populate the IBM upstream source with verified, resumable imports."""

import csv
import io
import logging
import re
import tarfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from psycopg.types.json import Jsonb

from fraud_risk_platform.persistence.database import connect_source
from fraud_risk_platform.persistence.files import identity, read_json, sha256

logger = logging.getLogger(__name__)
ADAPTER_VERSION = 1
HEADER = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Time",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
    "Errors?",
    "Is Fraud?",
]
PAYMENT_TYPES = {"Swipe Transaction", "Chip Transaction", "Online Transaction"}
INTEGER = re.compile(r"-?[0-9]+\Z")
AMOUNT = re.compile(r"\$(-?[0-9]+\.[0-9]{2})\Z")
COPY_SQL = """COPY ibm_transactions
    (import_id,source_row,user_id,card_id,tx_datetime,amount,use_chip,merchant_id,
     merchant_city,merchant_state,postal_code,mcc,errors,is_fraud,label_available_at)
    FROM STDIN"""


def load_manifest(path: Path) -> dict:
    manifest = read_json(path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset") != "ibm-tabformer"
    ):
        raise ValueError("Unsupported IBM source manifest")
    for key, pattern in (("sha256", r"[0-9a-f]{64}"), ("revision", r"[0-9a-f]{40}")):
        if not re.fullmatch(pattern, str(manifest.get(key, ""))):
            raise ValueError(f"Invalid source {key}")
    for key in ("filename", "csv_member"):
        name = manifest.get(key, "")
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Unsafe source {key}")
    for key in ("bytes", "csv_bytes"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError(f"Invalid source {key}")
    if "row_count" in manifest and (
        type(manifest["row_count"]) is not int or manifest["row_count"] <= 0
    ):
        raise ValueError("Invalid source row_count")
    if not str(manifest.get("url", "")).startswith("https://"):
        raise ValueError("Source download URL must use HTTPS")
    return manifest


def verify_archive(path: Path, manifest: dict) -> None:
    if path.stat().st_size != manifest["bytes"] or sha256(path) != manifest["sha256"]:
        raise ValueError(
            f"IBM archive checksum/size mismatch: {path}; existing files are not overwritten"
        )


def prepare_archive(manifest: dict, raw_dir: Path, archive: Path | None = None) -> Path:
    if archive is not None:
        verify_archive(archive, manifest)
        return archive
    folder = raw_dir / "ibm-tabformer" / manifest["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / manifest["filename"]
    if path.exists():
        verify_archive(path, manifest)
        return path
    temporary = folder / f".{manifest['filename']}.{uuid4().hex}.part"
    try:
        logger.info("Downloading %s (%s bytes)", manifest["url"], manifest["bytes"])
        request = Request(
            manifest["url"], headers={"User-Agent": "fraud-risk-platform/0.1"}
        )
        with urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > manifest["bytes"]:
                    raise ValueError("IBM download exceeds the pinned size")
                output.write(block)
        verify_archive(temporary, manifest)
        try:
            path.hardlink_to(temporary)
        except FileExistsError:
            verify_archive(path, manifest)
    except URLError as error:
        raise RuntimeError(
            "IBM download failed. Retry or supply --archive with the original "
            "transactions.tgz from the official distribution; the same checksum is required."
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


@contextmanager
def csv_rows(path: Path, manifest: dict):
    # Read the member directly; do not extract arbitrary archive paths to disk.
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name != manifest["csv_member"]:
                continue
            if not member.isfile() or member.size != manifest["csv_bytes"]:
                raise ValueError("Invalid IBM CSV archive member")
            with (
                archive.extractfile(member) as binary,
                io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream,
            ):
                reader = csv.reader(stream, strict=True)
                if next(reader, None) != HEADER:
                    raise ValueError(
                        "IBM CSV header does not match the versioned source contract"
                    )
                yield reader
            return
    raise ValueError("IBM CSV member missing from archive")


def integer(value: str, field: str, minimum: int, maximum: int) -> int:
    if not INTEGER.fullmatch(value):
        raise ValueError(f"{field} must be an integer")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{field} is out of range")
    return number


def parse_transaction(row: list[str], source_row: int, label_delay_days: int) -> tuple:
    """Keep source identity and exact signed money; do not fit feature encoders."""
    try:
        if len(row) != len(HEADER):
            raise ValueError(f"Expected {len(HEADER)} fields, received {len(row)}")
        user = integer(row[0], "User", 0, 2**31 - 1)
        card = integer(row[1], "Card", 0, 32767)
        if not re.fullmatch(r"[0-9]{2}:[0-9]{2}", row[5]):
            raise ValueError("Time must be HH:MM")
        hour, minute = map(int, row[5].split(":"))
        timestamp = datetime(
            integer(row[2], "Year", 1, 9999),
            integer(row[3], "Month", 1, 12),
            integer(row[4], "Day", 1, 31),
            hour,
            minute,
        )
        amount_match = AMOUNT.fullmatch(row[6])
        if not amount_match:
            raise ValueError(
                "Amount must be a dollar-prefixed decimal with two fractional digits"
            )
        amount = Decimal(amount_match[1])
        if abs(amount) >= Decimal("10000000000000000"):
            raise ValueError("Amount exceeds numeric(18,2)")
        if row[7] not in PAYMENT_TYPES:
            raise ValueError("Unknown Use Chip value")
        merchant = integer(row[8], "Merchant Name", -(2**63), 2**63 - 1)
        if not row[9] or not re.fullmatch(r"[0-9]{4}", row[12]):
            raise ValueError(
                "Merchant City is required and MCC must contain four digits"
            )
        if row[14] not in {"Yes", "No"}:
            raise ValueError("Is Fraud? must be Yes or No")
        return (
            source_row,
            user,
            card,
            timestamp,
            amount,
            row[7],
            merchant,
            row[9],
            row[10] or None,
            row[11] or None,
            row[12],
            row[13] or None,
            row[14] == "Yes",
            timestamp + timedelta(days=label_delay_days),
        )
    except (ValueError, OverflowError) as error:
        raise ValueError(f"IBM source row {source_row}: {error}") from error


def _write_chunk(conn, import_id: str, scanned_rows: int, rows: list[tuple]) -> None:
    # Rows and cursor commit together, including chunks with no selected customers.
    with conn.transaction():
        if rows:
            with conn.cursor().copy(COPY_SQL) as copy:
                for row in rows:
                    copy.write_row((import_id, *row))
        conn.execute(
            """UPDATE ibm_source_imports SET scanned_rows=%s,
            imported_rows=imported_rows+%s,updated_at=now() WHERE import_id=%s""",
            (scanned_rows, len(rows), import_id),
        )
    logger.info(
        "IBM import %s: scanned=%s, inserted_chunk=%s",
        import_id[:12],
        scanned_rows,
        len(rows),
    )


def _summary(conn, import_id: str, inserted_rows: int) -> dict:
    state = conn.execute(
        "SELECT * FROM ibm_source_imports WHERE import_id=%s", (import_id,)
    ).fetchone()
    counts = conn.execute(
        """SELECT count(*) AS rows,count(*) FILTER (WHERE is_fraud) AS frauds,
        min(tx_datetime)::text AS first_transaction,max(tx_datetime)::text AS last_transaction
        FROM ibm_transactions WHERE import_id=%s""",
        (import_id,),
    ).fetchone()
    if counts["rows"] != state["imported_rows"]:
        conn.execute(
            """UPDATE ibm_source_imports SET status='failed',
            last_error='Stored row count differs from the import ledger',updated_at=now()
            WHERE import_id=%s""",
            (import_id,),
        )
        raise ValueError("IBM stored row count differs from the import ledger")
    return {
        "dataset": "ibm-tabformer",
        "import_id": import_id,
        "status": state["status"],
        "scanned_rows": state["scanned_rows"],
        "imported_rows": counts.pop("rows"),
        "inserted_rows": inserted_rows,
        "customer_limit": state["customer_limit"],
        **counts,
    }


def seed_ibm(
    *,
    source_lock: Path,
    raw_dir: Path,
    label_delay_days: int,
    archive: Path | None = None,
    customer_limit: int | None = None,
    chunk_size: int = 200000,
    url: str | None = None,
) -> dict:
    if type(chunk_size) is not int or not 1 <= chunk_size <= 1000000:
        raise ValueError("chunk_size must be between 1 and 1000000")
    if type(label_delay_days) is not int or not 0 <= label_delay_days <= 36500:
        raise ValueError("label_delay_days must be between 0 and 36500")
    if customer_limit is not None and (
        type(customer_limit) is not int or not 1 <= customer_limit <= 2**31 - 1
    ):
        raise ValueError("customer_limit must be a positive integer")
    manifest = load_manifest(source_lock)
    path = prepare_archive(manifest, raw_dir, archive)
    # Chunk size is an execution detail, so it can change when resuming a job.
    import_id = identity(
        {
            "manifest": manifest,
            "adapter_version": ADAPTER_VERSION,
            "customer_limit": customer_limit,
            "label_delay_days": label_delay_days,
        }
    )
    with csv_rows(path, manifest) as reader, connect_source(url) as conn:
        conn.autocommit = True
        locked = conn.execute(
            "SELECT pg_try_advisory_lock(%s) AS locked", (int(import_id[:15], 16),)
        ).fetchone()["locked"]
        if not locked:
            raise ValueError(
                "This IBM import is already running; retry after it finishes"
            )
        state = conn.execute(
            "SELECT * FROM ibm_source_imports WHERE import_id=%s", (import_id,)
        ).fetchone()
        if state and state["status"] == "complete":
            return _summary(conn, import_id, 0)
        if state:
            _summary(conn, import_id, 0)
            conn.execute(
                "UPDATE ibm_source_imports SET status='loading',last_error=NULL,updated_at=now() WHERE import_id=%s",
                (import_id,),
            )
        else:
            conn.execute(
                """INSERT INTO ibm_source_imports
                (import_id,source_manifest,adapter_version,customer_limit,label_delay_days,status)
                VALUES (%s,%s,%s,%s,%s,'loading')""",
                (
                    import_id,
                    Jsonb(manifest),
                    ADAPTER_VERSION,
                    customer_limit,
                    label_delay_days,
                ),
            )
        checkpoint = state["scanned_rows"] if state else 0
        inserted = scanned = pending = 0
        rows = []
        try:
            for scanned, row in enumerate(reader, start=1):
                if scanned <= checkpoint:
                    continue
                if len(row) != len(HEADER):
                    raise ValueError(f"IBM source row {scanned}: incorrect field count")
                user = integer(row[0], f"User at source row {scanned}", 0, 2**31 - 1)
                if customer_limit is None or user < customer_limit:
                    rows.append(parse_transaction(row, scanned, label_delay_days))
                pending += 1
                if pending == chunk_size:
                    _write_chunk(conn, import_id, scanned, rows)
                    inserted += len(rows)
                    rows, pending = [], 0
            if scanned < checkpoint or scanned == 0:
                raise ValueError(
                    "IBM CSV is empty or shorter than the committed checkpoint"
                )
            if (
                manifest.get("row_count") is not None
                and scanned != manifest["row_count"]
            ):
                raise ValueError(
                    "IBM CSV row count differs from the pinned source manifest"
                )
            if pending:
                _write_chunk(conn, import_id, scanned, rows)
                inserted += len(rows)
            result = _summary(conn, import_id, inserted)
            if result["imported_rows"] == 0:
                raise ValueError("IBM selection contains no transactions")
            conn.execute(
                "UPDATE ibm_source_imports SET status='complete',completed_at=now(),updated_at=now() WHERE import_id=%s",
                (import_id,),
            )
            result["status"] = "complete"
            return result
        except Exception as error:
            # Committed chunks remain recoverable but the complete-only view hides them.
            conn.execute(
                "UPDATE ibm_source_imports SET status='failed',last_error=%s,updated_at=now() WHERE import_id=%s",
                (str(error)[:2000], import_id),
            )
            raise
