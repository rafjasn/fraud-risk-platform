"""Disk-backed accepted history, cross-batch consistency, and deterministic selection.

DuckDB validates whole deliveries as tables. The rules mirror the Observation
contract, with stricter text forms: timestamps must use ISO 8601 with a "T", and
amounts exactly two decimal places, as the extractor writes them and the IBM source
contract requires. Violation codes and counts match pydantic's for other inputs.
"""

import json
import os
from datetime import datetime
from typing import NamedTuple, get_args

import duckdb
import pyarrow as pa

from fraud_risk_platform.dataops.dataset import Observation, arrow_schema
from fraud_risk_platform.dataops.reports.summary import summarize, summary_counts


class Column(NamedTuple):
    sql_type: str
    json_types: tuple[str, ...]
    nullable: bool = False
    rule: str | None = None  # SQL on the typed value, written as {v}
    text: str | None = None  # Full-match pattern for the JSON text form


INTEGER, TEXT = ("UBIGINT", "BIGINT"), ("VARCHAR",)
TIMESTAMP_TEXT = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?"
PAYMENT_TYPES = ", ".join(
    f"'{value}'" for value in get_args(Observation.model_fields["use_chip"].annotation)
)
COLUMNS = {
    "source_row": Column("BIGINT", INTEGER, rule="{v} > 0"),
    "user_id": Column("INTEGER", INTEGER, rule="{v} >= 0"),
    "card_id": Column("SMALLINT", INTEGER, rule="{v} >= 0"),
    "tx_datetime": Column("TIMESTAMP", TEXT, text=TIMESTAMP_TEXT),
    "amount": Column("DECIMAL(18,2)", TEXT, text=r"-?[0-9]{1,16}\.[0-9]{2}"),
    "use_chip": Column("VARCHAR", TEXT, rule=f"{{v}} IN ({PAYMENT_TYPES})"),
    "merchant_id": Column("BIGINT", INTEGER),
    "merchant_city": Column("VARCHAR", TEXT, rule="length({v}) >= 1"),
    "merchant_state": Column("VARCHAR", TEXT, nullable=True),
    "postal_code": Column("VARCHAR", TEXT, nullable=True),
    "mcc": Column("VARCHAR", TEXT, rule="regexp_full_match({v}, '[0-9]{4}')"),
    "errors": Column("VARCHAR", TEXT, nullable=True),
    "is_fraud": Column("BOOLEAN", ("BOOLEAN",), nullable=True),
    "label_available_at": Column("TIMESTAMP", TEXT, text=TIMESTAMP_TEXT),
}
FIELDS = list(COLUMNS)
SELECTED = ", ".join(FIELDS)


def _sql_list(values) -> str:
    return "[" + ", ".join(f"'{value}'" for value in values) + "]"


def _value(name: str) -> str:
    """The field's JSON value: SQL NULL when absent, JSON null when null."""
    return f"v[{FIELDS.index(name) + 1}]"


def _json_valid(name: str, column: Column) -> str:
    kind = f"json_type({_value(name)})"
    text = f"json_extract_string({_value(name)}, '$')"
    value = f"TRY_CAST({text} AS {column.sql_type})"
    checks = [
        f"{kind} IN ({', '.join(repr(t) for t in column.json_types)})",
        f"{value} IS NOT NULL",
    ]
    if column.text:
        checks.append(f"regexp_full_match({text}, '{column.text}')")
    if column.rule:
        checks.append(column.rule.replace("{v}", value))
    # Every field is required; null is a value only nullable fields accept.
    return (
        f"CASE WHEN {kind} IS NULL THEN false WHEN {kind} = 'NULL' THEN "
        f"{str(column.nullable).lower()} ELSE {' AND '.join(checks)} END"
    )


def _typed_valid(name: str, column: Column) -> str:
    rule = column.rule.replace("{v}", name) if column.rule else "true"
    return (
        f"CASE WHEN {name} IS NULL THEN {str(column.nullable).lower()} ELSE {rule} END"
    )


def _schema_codes(valid) -> str:
    return (
        "["
        + ", ".join(
            f"CASE WHEN ({valid(name, column)}) IS NOT TRUE THEN 'schema.{name}' END"
            for name, column in COLUMNS.items()
        )
        + "]"
    )


STAGED = f"""CREATE OR REPLACE TABLE staged (
    line BIGINT, reference JSON, schema_codes VARCHAR[],
    {", ".join(f"{name} {column.sql_type}" for name, column in COLUMNS.items())})"""
# Delivery lines, numbered from each file so nothing depends on scan order.
LINES = "CREATE OR REPLACE TABLE lines (line BIGINT, j VARCHAR)"
LINE_INSERT = """INSERT INTO lines SELECT $offset + position, text FROM (
    SELECT unnest(lines) AS text, generate_subscripts(lines, 1) AS position
    FROM (SELECT CASE WHEN content = '' THEN []::VARCHAR[]
            ELSE string_split(regexp_replace(content, chr(10) || '$', ''), chr(10))
        END AS lines FROM read_text($path)))"""
# One parallel pass over all lines that parses each line once; a malformed line
# fails the whole delivery.
JSON_STAGE = f"""CREATE OR REPLACE TABLE staged AS SELECT line,
    CASE WHEN kind = 'OBJECT' AND {_value("source_row")} IS NOT NULL
        THEN {_value("source_row")} ELSE to_json(line) END AS reference,
    CASE WHEN kind = 'OBJECT' THEN list_filter(
        {_schema_codes(_json_valid)} || list_transform(
            list_filter(keys, k -> NOT list_contains({_sql_list(FIELDS)}, k)),
            k -> 'schema.' || k),
        code -> code IS NOT NULL)
    -- pydantic reports a non-object record at the empty location.
    ELSE ['schema.'] END AS schema_codes,
    {", ".join(f"TRY_CAST(json_extract_string({_value(n)}, '$') AS {c.sql_type}) AS {n}" for n, c in COLUMNS.items())}
    FROM (SELECT line, json_type(j) AS kind, json_keys(j) AS keys,
        json_extract(j, [{", ".join(f"'$.{n}'" for n in FIELDS)}]) AS v FROM lines)"""
# Parquet rows in file order, then row order within each file.
PARQUET_ROWS = f"""SELECT row_number() OVER (
        ORDER BY list_position($files, filename), file_row_number) AS line, {SELECTED}
    FROM read_parquet($files, filename = true, file_row_number = true)"""
TYPED_STAGE = f"""CREATE OR REPLACE TABLE staged AS
    SELECT line, to_json(source_row) AS reference,
    list_filter({_schema_codes(_typed_valid)}, code -> code IS NOT NULL) AS schema_codes,
    {SELECTED} FROM ({PARQUET_ROWS})"""


def _matches(row: str, previous: str) -> str:
    """The same observation, keeping any label the previous version released."""
    same = " AND ".join(
        f"{row}.{name} IS NOT DISTINCT FROM {previous}.{name}"
        for name in FIELDS
        if name != "is_fraud"
    )
    return (
        f"({same} AND ({previous}.is_fraud IS NULL "
        f"OR {previous}.is_fraud IS NOT DISTINCT FROM {row}.is_fraud))"
    )


# A later copy of a record is compared with the first copy when that copy was
# accepted (it then replaced the history row), and with the history otherwise.
CHECK = f"""CREATE OR REPLACE TABLE checked AS
WITH ranked AS (
    SELECT s.*,
        row_number() OVER (PARTITION BY s.source_row ORDER BY s.line) AS copy,
        NOT (s.tx_datetime >= $start AND s.tx_datetime < $cutoff) AS time_error,
        s.label_available_at <> s.tx_datetime + to_days($delay) AS delay_error,
        (s.label_available_at <= $cutoff) <> (s.is_fraud IS NOT NULL) AS visibility_error,
        h.source_row IS NOT NULL AS in_history,
        h.is_fraud AS history_label,
        h.source_row IS NOT NULL AND NOT {_matches("s", "h")} AS history_conflict
    FROM staged s LEFT JOIN observations h ON h.source_row = s.source_row
    WHERE len(s.schema_codes) = 0),
resolved AS (
    SELECT r.*,
        CASE WHEN r.copy = 1 OR f.time_error OR f.delay_error OR f.visibility_error
            OR f.history_conflict THEN r.history_conflict
            ELSE NOT {_matches("r", "f")} END AS conflict
    FROM ranked r JOIN ranked f ON f.source_row = r.source_row AND f.copy = 1)
SELECT line, reference, in_history, history_label, {SELECTED}, list_filter([
        CASE WHEN time_error THEN 'transaction_time' END,
        CASE WHEN delay_error THEN 'label_delay' END,
        CASE WHEN visibility_error THEN 'label_visibility' END,
        CASE WHEN copy > 1 THEN 'duplicate_identity' END,
        CASE WHEN conflict THEN 'conflicting_record' END
    ], code -> code IS NOT NULL) AS codes
FROM resolved"""
FINDINGS = """CREATE OR REPLACE TABLE findings AS
SELECT line, reference, unnest(codes) AS code, generate_subscripts(codes, 1) AS position
FROM (SELECT line, reference, schema_codes AS codes FROM staged WHERE len(schema_codes) > 0
      UNION ALL SELECT line, reference, codes FROM checked WHERE len(codes) > 0)"""


class HistoryStore:
    """A snapshot's context (accepted history) in DuckDB: validates deliveries, selects training rows."""

    def __init__(self, path, *, memory_limit: str | None = None):
        self.db = duckdb.connect(
            str(path),
            config={
                # Line numbers are explicit, so large inserts need not keep scan
                # order; this lets them stream within the memory limit.
                "preserve_insertion_order": False,
                "memory_limit": memory_limit
                or os.environ.get("FRAUD_DUCKDB_MEMORY_LIMIT", "2GB"),
            },
        )
        columns = ", ".join(
            f"{name} {column.sql_type}{'' if column.nullable else ' NOT NULL'}"
            for name, column in COLUMNS.items()
        )
        self.db.execute(f"CREATE TABLE observations ({columns})")

    def close(self):
        self.db.close()

    def apply(self, files, *, start, cutoff, label_delay_days, reporting=True):
        """Validate one delivery's raw JSONL parts, in order, and accept its valid rows."""
        self.db.execute(LINES)
        offset = 0
        for path in files:
            offset += self.db.execute(
                LINE_INSERT, {"path": str(path), "offset": offset}
            ).fetchone()[0]
        self.db.execute(JSON_STAGE)
        self.db.execute("DROP TABLE lines")
        return self._check(start, cutoff, label_delay_days, reporting=reporting)

    def apply_parquet(self, files, *, start, cutoff, label_delay_days):
        """Validate a published snapshot's typed context, as apply() does a delivery."""
        if files:
            self.db.execute(TYPED_STAGE, {"files": [str(path) for path in files]})
        else:
            self.db.execute(STAGED)
        return self._check(start, cutoff, label_delay_days, reporting=False)

    def _check(self, start, cutoff, label_delay_days, *, reporting):
        self.db.execute(
            CHECK, {"start": start, "cutoff": cutoff, "delay": label_delay_days}
        )
        self.db.execute(FINDINGS)
        # Codes in order of first occurrence, as rows were reported before.
        violations = dict(
            self.db.execute(
                """SELECT code, count(*) OVER (PARTITION BY code) FROM findings
                QUALIFY row_number() OVER (PARTITION BY code ORDER BY line, position) = 1
                ORDER BY line, position"""
            ).fetchall()
        )
        examples = {code: [] for code in violations}
        for code, reference in self.db.execute(
            """SELECT code, reference FROM findings QUALIFY row_number() OVER (
                PARTITION BY code ORDER BY line, position) <= 20
            ORDER BY line, position"""
        ).fetchall():
            examples[code].append(json.loads(reference))
        received = self.db.execute("SELECT count(*) FROM staged").fetchone()[0]
        invalid = self.db.execute(
            "SELECT count(DISTINCT line) FROM findings"
        ).fetchone()[0]
        new, updated, known, frauds = self.db.execute(
            """SELECT count(*) FILTER (WHERE NOT in_history),
                count(*) FILTER (WHERE in_history AND history_label IS NULL
                    AND is_fraud IS NOT NULL),
                count(*) FILTER (WHERE is_fraud IS NOT NULL),
                count(*) FILTER (WHERE is_fraud)
            FROM checked WHERE len(codes) = 0"""
        ).fetchone()
        valid = received - invalid
        if reporting:
            accepted = "checked WHERE len(codes) = 0"
            summary = summarize(self.db, accepted)
            new_summary = summarize(self.db, accepted + " AND NOT in_history")
        else:
            summary = summary_counts(rows=valid, known=known, frauds=frauds)
        # Valid rows are accepted even beside invalid ones; callers discard a context
        # whose delivery was rejected.
        self.db.execute(
            """DELETE FROM observations WHERE source_row IN
            (SELECT source_row FROM checked WHERE len(codes) = 0)"""
        )
        self.db.execute(
            f"INSERT INTO observations SELECT {SELECTED} FROM checked WHERE len(codes) = 0"
        )
        return {
            "received": received,
            "valid": valid,
            "invalid": invalid,
            "new_transactions": new,
            "label_updates": updated,
            "unchanged": valid - new - updated,
            "violations": violations,
            "examples": examples,
            "summary": summary,
            "new_transaction_summary": new_summary if reporting else None,
        }

    def select(self, profile, windows, source_sha):
        start, end = map(datetime.fromisoformat, windows["train"])
        query = """CREATE OR REPLACE TABLE selected AS SELECT source_row FROM observations
            WHERE tx_datetime >= $start AND tx_datetime < $end AND is_fraud IS NOT NULL"""
        parameters = {"start": start, "end": end}
        if profile.max_train_rows is not None:
            # Hex digests of f"{source_sha}:{seed}:{row}", ordered like the Python hashes.
            query += " ORDER BY sha256($prefix || source_row::VARCHAR), source_row LIMIT $limit"
            parameters |= {
                "prefix": f"{source_sha}:{profile.selection_seed}:",
                "limit": profile.max_train_rows,
            }
        self.db.execute(query, parameters)
        self._training_view()

    def _training_view(self):
        self.db.execute(
            f"""CREATE OR REPLACE VIEW training AS SELECT {SELECTED} FROM observations
            WHERE source_row IN (SELECT source_row FROM selected)"""
        )

    def load(self, files, *, training):
        """Published context and training Parquet as history and selection, unvalidated."""
        if files:
            self.db.execute(
                f"INSERT INTO observations SELECT {SELECTED} FROM read_parquet($files)",
                {"files": [str(path) for path in files]},
            )
        self.db.execute("CREATE OR REPLACE TABLE selected (source_row BIGINT)")
        if training:
            self.db.execute(
                "INSERT INTO selected SELECT source_row FROM read_parquet($files)",
                {"files": [str(path) for path in training]},
            )
        self._training_view()

    def _split(self, split, windows):
        query = f"SELECT {SELECTED} FROM observations"
        if split == "context":
            return query, {}
        if split == "train":
            return f"SELECT {SELECTED} FROM training", {}
        start, end = map(datetime.fromisoformat, windows[split])
        return (
            query + " WHERE tx_datetime >= $start AND tx_datetime < $end"
            " AND is_fraud IS NOT NULL",
            {"start": start, "end": end},
        )

    def rows(self, split="context", windows=None, batch_size=50_000):
        """The split as Arrow tables in (time, source row) order, typed as snapshots are."""
        query, parameters = self._split(split, windows)
        schema = arrow_schema()
        reader = self.db.execute(
            query + " ORDER BY tx_datetime, source_row", parameters
        ).to_arrow_reader(batch_size)
        for batch in reader:
            if batch.num_rows:
                yield pa.Table.from_batches([batch]).cast(schema)

    def count_eligible(self, window):
        start, end = map(datetime.fromisoformat, window)
        return self.db.execute(
            """SELECT count(*) FROM observations
            WHERE tx_datetime >= $start AND tx_datetime < $end AND is_fraud IS NOT NULL""",
            {"start": start, "end": end},
        ).fetchone()[0]

    def compare(self, split, windows, files):
        """Stored Parquet versus the derived split: same rows, in (time, row) order."""
        query, parameters = self._split(split, windows)
        if files:
            self.db.execute(
                f"CREATE OR REPLACE TABLE stored AS {PARQUET_ROWS}",
                {"files": [str(path) for path in files]},
            )
        else:
            self.db.execute(
                "CREATE OR REPLACE TABLE stored AS SELECT 0::BIGINT AS line, "
                f"{SELECTED} FROM observations WHERE false"
            )
        differing = self.db.execute(
            f"""SELECT count(*) FROM (
                (({query}) EXCEPT ALL (SELECT {SELECTED} FROM stored))
                UNION ALL ((SELECT {SELECTED} FROM stored) EXCEPT ALL ({query})))""",
            parameters,
        ).fetchone()[0]
        unordered = self.db.execute(
            """SELECT count(*) FROM (
                SELECT tx_datetime, source_row,
                    lag(tx_datetime) OVER (ORDER BY line) AS previous_time,
                    lag(source_row) OVER (ORDER BY line) AS previous_row
                FROM stored)
            WHERE previous_time > tx_datetime
                OR (previous_time = tx_datetime AND previous_row >= source_row)"""
        ).fetchone()[0]
        rows, frauds, pending = self.db.execute(
            """SELECT count(*), count(*) FILTER (WHERE is_fraud),
                count(*) FILTER (WHERE is_fraud IS NULL) FROM stored"""
        ).fetchone()
        return {
            "identical": not differing and not unordered,
            "rows": rows,
            "frauds": frauds,
            "unavailable_labels": pending,
        }
