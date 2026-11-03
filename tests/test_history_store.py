from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from fraud_risk_platform.dataops.dataset import (
    DatasetProfile,
    Observation,
    arrow_schema,
)
from fraud_risk_platform.dataops.history_store import FIELDS, HistoryStore
from fraud_risk_platform.persistence.files import canonical

START = datetime(2019, 1, 1)
CUTOFF = datetime(2019, 1, 10)
DELAY = 2


def row(source_row=1, *, day=0, amount="10.50", fraud=False, **changes):
    timestamp = datetime(2019, 1, 1, 12) + timedelta(days=day)
    return {
        "source_row": source_row,
        "user_id": source_row % 4,
        "card_id": 0,
        "tx_datetime": timestamp.isoformat(),
        "amount": amount,
        "use_chip": "Online Transaction",
        "merchant_id": -7,
        "merchant_city": "ONLINE",
        "merchant_state": None,
        "postal_code": None,
        "mcc": "5411",
        "errors": None,
        "is_fraud": fraud,
        "label_available_at": (timestamp + timedelta(days=DELAY)).isoformat(),
        **changes,
    }


@pytest.fixture
def context(tmp_path):
    result = HistoryStore(tmp_path / "history.duckdb")
    yield result
    result.close()


def deliver(context, tmp_path, rows, *, cutoff=CUTOFF, name="raw.jsonl"):
    path = tmp_path / name
    path.write_bytes(
        b"".join((r if isinstance(r, bytes) else canonical(r)) + b"\n" for r in rows)
    )
    return context.apply(
        [path], start=START, cutoff=cutoff, label_delay_days=DELAY, reporting=False
    )


def pydantic_codes(raw):
    try:
        Observation.model_validate(raw)
    except ValidationError as error:
        return ["schema." + ".".join(map(str, e["loc"])) for e in error.errors()]
    return []


def test_columns_follow_the_observation_contract():
    assert FIELDS == list(Observation.model_fields) == arrow_schema().names


MISSING = object()
SCHEMA_CASES = [
    ("source_row", "1"),
    ("source_row", 1.0),
    ("source_row", True),
    ("source_row", 0),
    ("source_row", 2**63),
    ("source_row", MISSING),
    ("user_id", "5"),
    ("user_id", -1),
    ("user_id", 2**31),
    ("user_id", None),
    ("card_id", 32768),
    ("card_id", -1),
    ("tx_datetime", 1546344000),
    ("tx_datetime", "2019-02-30T00:00:00"),
    ("tx_datetime", "2019-01-01T12:00:00Z"),
    ("tx_datetime", None),
    ("amount", 10.5),
    ("amount", "1.234"),
    ("amount", "broken"),
    ("amount", "12345678901234567.00"),
    ("amount", MISSING),
    ("use_chip", "Tap Transaction"),
    ("use_chip", 1),
    ("merchant_id", "7"),
    ("merchant_id", 2**63),
    ("merchant_id", 1.5),
    ("merchant_city", ""),
    ("merchant_city", None),
    ("merchant_state", 5),
    ("merchant_state", MISSING),
    ("postal_code", 91750),
    ("mcc", "541"),
    ("mcc", 5411),
    ("errors", 3),
    ("is_fraud", 0),
    ("is_fraud", "true"),
    ("is_fraud", MISSING),
    ("label_available_at", "bad"),
    ("note", "an extra field"),
]


@pytest.mark.parametrize("field,value", SCHEMA_CASES)
def test_schema_violations_match_pydantic(context, tmp_path, field, value):
    raw = row()
    if value is MISSING:
        del raw[field]
    else:
        raw[field] = value
    expected = pydantic_codes(raw)
    assert expected, "each case must be invalid under the Observation contract"
    result = deliver(context, tmp_path, [raw])
    assert list(result["violations"].items()) == list(Counter(expected).items())
    assert (result["received"], result["invalid"], result["valid"]) == (1, 1, 0)


def test_several_violations_are_reported_in_pydantic_order(context, tmp_path):
    raw = {**row(), "user_id": "5", "mcc": "541", "note": "x"}
    result = deliver(context, tmp_path, [raw])
    assert list(result["violations"]) == pydantic_codes(raw)


@pytest.mark.parametrize(
    "line",
    [b"[1, 2]", b"5", b'"text"', b"null"],
    ids=["array", "number", "string", "null"],
)
def test_non_object_records_are_rejected_like_pydantic(context, tmp_path, line):
    result = deliver(context, tmp_path, [row(1), line])
    assert result["violations"] == {"schema.": 1}
    # Without a source_row, the reference is the record's position in the delivery.
    assert result["examples"] == {"schema.": [2]}


@pytest.mark.parametrize(
    "field,value",
    [("tx_datetime", "2019-01-01 12:00:00"), ("amount", "10.5")],
)
def test_text_forms_are_stricter_than_pydantic(context, tmp_path, field, value):
    raw = {**row(), field: value}
    assert pydantic_codes(raw) == []
    assert f"schema.{field}" in deliver(context, tmp_path, [raw])["violations"]


def test_valid_delivery_is_accepted_and_counted(context, tmp_path):
    result = deliver(
        context, tmp_path, [row(1), row(2, fraud=True), row(3, day=8, fraud=None)]
    )
    assert result["violations"] == {} and result["invalid"] == 0
    assert result["new_transactions"] == result["valid"] == 3
    assert result["summary"]["rows"] == 3
    assert result["summary"]["frauds"] == 1
    assert result["summary"]["unknown_labels"] == 1
    assert result["new_transaction_summary"] is None


@pytest.mark.parametrize(
    "raw,code",
    [
        (row(day=-1), "transaction_time"),
        (row(day=9, fraud=None), "transaction_time"),
        ({**row(), "label_available_at": "2019-01-02T12:00:00"}, "label_delay"),
        (row(fraud=None), "label_visibility"),
        (row(day=8), "label_visibility"),
    ],
    ids=["before-start", "at-cutoff", "delay", "hidden-label", "future-label"],
)
def test_time_and_label_rules(context, tmp_path, raw, code):
    assert deliver(context, tmp_path, [raw])["violations"] == {code: 1}


def test_duplicates_and_conflicts_within_a_delivery(context, tmp_path):
    result = deliver(context, tmp_path, [row(1), row(1), row(1, amount="999999.00")])
    assert result["violations"] == {"duplicate_identity": 2, "conflicting_record": 1}
    assert (result["valid"], result["invalid"]) == (1, 2)


def test_later_copies_of_an_invalid_first_copy_compare_with_history(context, tmp_path):
    result = deliver(
        context, tmp_path, [row(1, day=-1), row(1, day=-1, amount="999999.00")]
    )
    assert result["violations"] == {"transaction_time": 2, "duplicate_identity": 1}


def test_history_conflicts_label_updates_and_retractions(context, tmp_path):
    first = deliver(context, tmp_path, [row(1), row(2, day=7, fraud=None)])
    assert first["invalid"] == 0
    later = datetime(2019, 1, 20)
    conflict = deliver(
        context, tmp_path, [row(1, amount="1.00")], cutoff=later, name="b.jsonl"
    )
    assert conflict["violations"] == {"conflicting_record": 1}
    retracted = deliver(
        context, tmp_path, [row(1, fraud=None)], cutoff=later, name="c.jsonl"
    )
    assert retracted["violations"] == {
        "label_visibility": 1,
        "conflicting_record": 1,
    }
    update = deliver(
        context, tmp_path, [row(1), row(2, day=7, fraud=True)], cutoff=later
    )
    assert update["invalid"] == 0
    assert (update["label_updates"], update["unchanged"]) == (1, 1)
    assert update["new_transactions"] == 0
    assert context.count_eligible(["2019-01-01T00:00:00", "2019-01-20T00:00:00"]) == 2


def test_examples_are_capped_and_follow_delivery_order(context, tmp_path):
    rows = [row(1, label_available_at="2019-01-05T00:00:00")]
    rows += [row(n, amount="bad") for n in range(2, 30)]
    result = deliver(context, tmp_path, rows)
    assert list(result["violations"]) == ["label_delay", "schema.amount"]
    assert result["violations"]["schema.amount"] == 28
    assert result["examples"]["schema.amount"] == list(range(2, 22))


def test_reporting_summarizes_valid_and_new_rows(context, tmp_path):
    deliver(context, tmp_path, [row(1, day=7, fraud=None)])
    path = tmp_path / "next.jsonl"
    path.write_bytes(b"".join(canonical(r) + b"\n" for r in [row(1, day=7), row(2)]))
    result = context.apply(
        [path], start=START, cutoff=datetime(2019, 1, 20), label_delay_days=DELAY
    )
    assert result["summary"]["rows"] == 2
    assert result["new_transaction_summary"]["rows"] == 1
    assert result["summary"]["amount"]["min"] == "10.50"


def test_selection_matches_the_python_hash_order(context, tmp_path):
    deliver(context, tmp_path, [row(n, day=n % 7) for n in range(1, 101)])
    profile = DatasetProfile(
        train_start="2019-01-01",
        final_holdout_start="2019-03-01",
        max_train_rows=10,
        selection_seed=7,
    )
    windows = {"train": ["2019-01-01T00:00:00", "2019-01-08T00:00:00"]}
    source = "b" * 64
    context.select(profile, windows, source)
    selected = sorted(
        r["source_row"] for t in context.rows("train", windows) for r in t.to_pylist()
    )
    expected = sorted(
        range(1, 101),
        key=lambda n: (sha256(f"{source}:7:{n}".encode()).hexdigest(), n),
    )[:10]
    assert selected == sorted(expected)


def test_rows_are_typed_ordered_batches(context, tmp_path):
    deliver(context, tmp_path, [row(n, day=(7 - n) % 7) for n in range(1, 8)])
    batches = list(context.rows("context", batch_size=3))
    assert [b.num_rows for b in batches] == [3, 3, 1]
    assert all(b.schema.equals(arrow_schema()) for b in batches)
    ordered = [
        (r["tx_datetime"], r["source_row"]) for b in batches for r in b.to_pylist()
    ]
    assert ordered == sorted(ordered)
    empty = {"validation": ["2018-01-01T00:00:00", "2018-01-02T00:00:00"]}
    assert list(context.rows("validation", empty)) == []


def test_compare_detects_changed_missing_and_reordered_rows(context, tmp_path):
    deliver(context, tmp_path, [row(n, day=n % 5) for n in range(1, 21)])
    windows = {"validation": ["2019-01-01T00:00:00", "2019-01-10T00:00:00"]}
    parts = list(context.rows("validation", windows, batch_size=8))
    whole = pa.concat_tables(parts)

    def matches(*tables):
        paths = []
        for table in tables:
            paths.append(tmp_path / f"stored-{len(list(tmp_path.glob('stored-*')))}")
            pq.write_table(table, paths[-1])
        return context.compare("validation", windows, paths)

    assert matches(*parts) == {
        "identical": True,
        "rows": 20,
        "frauds": 0,
        "unavailable_labels": 0,
    }
    reversed_rows = whole.take(list(reversed(range(whole.num_rows))))
    city = whole.schema.get_field_index("merchant_city")
    changed = whole.set_column(
        city,
        whole.schema.field(city),
        pa.array(["CHANGED", *whole["merchant_city"].to_pylist()[1:]]),
    )
    for table in (whole.slice(1), reversed_rows, changed):
        assert not matches(table)["identical"]


def test_published_context_is_revalidated(context, tmp_path):
    deliver(context, tmp_path, [row(1), row(2, day=8, fraud=None)])
    paths = []
    for index, table in enumerate(context.rows("context", batch_size=1)):
        paths.append(tmp_path / f"context-{index}.parquet")
        pq.write_table(table, paths[-1])
    replay = HistoryStore(tmp_path / "verify.duckdb")
    try:
        result = replay.apply_parquet(
            paths, start=START, cutoff=CUTOFF, label_delay_days=DELAY
        )
        assert (result["received"], result["invalid"]) == (2, 0)
        assert result["summary"]["unknown_labels"] == 1
        # At an earlier cutoff, one label was not yet released and one row is too late.
        early = replay.apply_parquet(
            paths, start=START, cutoff=datetime(2019, 1, 3), label_delay_days=DELAY
        )
        assert early["violations"] == {"label_visibility": 1, "transaction_time": 1}
    finally:
        replay.close()
