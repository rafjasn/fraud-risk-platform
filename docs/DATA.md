# Data

IBM's synthetic card transactions are loaded once into a PostgreSQL database that plays the bank's system. From there they are delivered month by month: each delivery is validated as a whole, then frozen with all earlier accepted history into an immutable S3 snapshot, which is what training reads.

## Source

[IBM's TabFormer repository](https://github.com/IBM/TabFormer) distributes the data. [config/ibm-tabformer.lock.json](../config/ibm-tabformer.lock.json) pins the upstream revision, download URL, compressed and uncompressed sizes, CSV member name, row count, and SHA-256, taken from the upstream Git LFS pointer, and links the [license](https://github.com/IBM/TabFormer/blob/ebb7cd68ee1897599568107740bc452104bbbaf8/LICENSE). The archive is about 279 MB and is not stored in Git.

The verified archive holds 24,386,900 transactions from 2,000 users, dated 1991 to 2020, in 15 columns. Merchant state, postal code, and errors can be empty. The 1,244,689 negative amounts are kept.

| CSV field | Database field | Mapping |
| --- | --- | --- |
| None | `source_row` | 1-based CSV row number; the transaction's identity, since IBM has no transaction ID |
| Import configuration | `import_id` | Hash of the source manifest, loader version, customer limit, and label delay |
| `User`, `Card` | `user_id`, `card_id` | Non-negative integers; a card is identified by both together |
| `Year`, `Month`, `Day`, `Time` | `tx_datetime` | One timestamp to the minute, without a time zone and treated as UTC |
| `Amount` | `amount` | Dollar sign removed, stored as an exact signed `numeric(18,2)`, never rounded |
| `Use Chip` | `use_chip` | `Swipe Transaction`, `Chip Transaction`, or `Online Transaction` |
| `Merchant Name` | `merchant_id` | Signed 64-bit integer; negative IDs are valid |
| `Merchant City`, `Merchant State`, `Zip` | `merchant_city`, `merchant_state`, `postal_code` | Text as given, including `ONLINE` and the `.0` suffix; empty becomes NULL |
| `MCC` | `mcc` | Four-digit text code |
| `Errors?` | `errors` | Original text; empty becomes NULL |
| `Is Fraud?` | `is_fraud` | Strictly `Yes` or `No` |
| Simulation | `label_available_at` | Transaction time plus the label delay (7 days), for both classes |

The source gives no time zone and no investigation date. The UTC convention and the 7-day label delay are simulation assumptions, not observed facts.

## Seeding

`make seed-source` downloads the archive to a temporary file, checks its size and SHA-256, and caches it in `data/raw/ibm-tabformer/`. It then streams the compressed CSV into `fraud_source.ibm_transactions` with PostgreSQL `COPY`, 200,000 rows per chunk. Each chunk commits together with the import's progress cursor, so a crash loses at most one chunk, and a PostgreSQL advisory lock refuses a second loader. A rerun reads the stream up to the cursor and continues; a completed import is reused and inserts nothing. If GitHub's LFS download is unavailable, pass the original archive from [IBM's Box distribution](https://ibm.box.com/v/tabformer-data) with `--archive`; it must match the same checksum.

An import becomes `complete` only when its row count matches the pinned manifest. Downstream queries use the `ibm_complete_transactions` view, which hides incomplete imports, and an import whose stored rows later stop matching its count is marked failed. A malformed record fails its chunk; it is never dropped silently.

`SEED_ARGS="--customer-limit 20"` loads users 0–19 only (295,448 rows) as a loader test. It is too small for the default training splits. [ibm-seed-verification.json](ibm-seed-verification.json) records the import checks: all 24,386,900 rows parsed (29,757 frauds), and a repeated subset import inserted no duplicates.

## Timeline

[config/data.json](../config/data.json) is the only data configuration:

| Setting | Value |
| --- | --- |
| First cutoff | 2019-07-08; each later snapshot advances one calendar month |
| Training start | 2018-01-01, expanding every month, with 30 days of earlier context for card history |
| Validation and evaluation | 30 days each, ending 7 days before the cutoff, so all their labels are released |
| Label delay | 7 days for both classes |
| Training rows | At most 1,000,000 per snapshot (`max_train_rows`; `null` keeps every eligible row) |
| Final holdout | From 2019-10-01; no training snapshot's evaluation window may reach it |

The dates follow the source's class availability, not model results. [ibm-monthly-profile.json](ibm-monthly-profile.json) records the monthly counts: there are frauds through October 2019 and none afterwards. The training cap orders eligible rows by a SHA-256 hash of the source checksum, a seed, and the row number. The sample is deterministic and independent of the label, and validation, evaluation, and context are never capped.

## Monthly deliveries

`make initial-snapshot` publishes the first snapshot, or verifies and returns the existing one. If S3 has lost it, it starts a fresh stream. `make next-snapshot` delivers the next month: its new transactions plus the labels of earlier transactions that matured in it. It skips empty months, shortens the last interval at the end of the data, and then reports `no_new_data` without changing anything. A snapshot command takes minutes; nothing waits for a real month. An advisory lock refuses overlapping commands.

Each delivery goes through these steps:

1. Read the month from a consistent, read-only view of the source with a server-side cursor, masking unreleased labels in SQL.
2. Save the delivered JSONL and its manifest to S3 before validating anything.
3. Rebuild accepted history in a temporary DuckDB file by replaying every earlier accepted delivery, then validate the new one against it as a table, counting every violation per rule with example rows.
4. Reject the whole delivery if any record fails. Its evidence and reports are kept, and a quarantine record points at them.
5. Otherwise write the batch manifest, then record the accepted batch and advance the cursor in one database transaction.
6. Export the snapshot, publish its reports, write its manifest last, and only then advance the simulation clock.

Validation covers schema, types, required values, exact two-decimal amounts, ISO timestamps, label visibility, and duplicate or conflicting identities. Missing optional values, new customers, and negative amounts are valid. A changed record for an existing row is rejected as a conflict, because the loader has no notion of corrections.

If publication fails after acceptance, the next run reuses the accepted delivery and finishes the same month. A killed process can leave an attempt marked `running` and unreferenced S3 objects; nothing cleans these up automatically. The platform database's `ingestion_attempts`, `ingestion_batches`, and `monthly_snapshot_progress` tables hold every attempt, the accepted lineage, and the clock.

`SNAPSHOT_ARGS='--fault <name>'` corrupts the first delivered records, in memory, without touching the source: `missing-amount`, `invalid-amount`, `duplicate`, or `conflict`. A clean rerun then processes the same month. A snapshot can be valid data but unsuitable for training, for example when a partition has too few frauds or reaches the holdout. Ingestion still publishes it, and training refuses it with the reason.

## Snapshots and storage

```text
s3://fraud-data/
  batches/<stream>/<attempt>/part-*.jsonl, raw-manifest.json, manifest.json
  quarantine/<attempt>.json
  reports/ingestion/<attempt>/report.json, report.html
  reports/snapshots/<request-hash>/v5/report.json, report.html
  objects/<sha256>.parquet
  snapshots/<snapshot-id>/manifest.json
  snapshot-requests/<request-hash>.json
```

A snapshot's partitions (context, train, validation, evaluation) are Parquet objects named by their SHA-256, so identical chunks are shared across snapshots. Every write is conditional: an existing key must already hold the same bytes. The manifest lists every object with its checksum, size, and row count, and is written last, so a crash leaves no half snapshot. The snapshot ID is the hash of the manifest. Training checks checksums, schemas, counts, windows, and label visibility before fitting; `make verify-snapshot` also replays the context and re-derives every partition row by row. This makes snapshots immutable to the application, but not to an administrator; S3 Object Lock is not used.

Extraction and export work in 50,000-row chunks. DuckDB uses every core and at most 2 GB of memory, spilling to disk beyond that; set `FRAUD_DUCKDB_MEMORY_LIMIT` to change it. LocalStack runs S3 only, in memory: after a restart, `make initial-snapshot` rebuilds from the source database, with new snapshot IDs.

## Data reports

Every delivery attempt, including a rejected or failed one, gets a JSON and an HTML report, and every snapshot gets one too. Pages are self-contained, with charts embedded. Reports carry a format version (currently 5); a format change gets a new number, so existing snapshots keep the reports they were published with. `make snapshot-report SNAPSHOT=<id>` downloads a snapshot's report, and `make refresh-snapshot-report SNAPSHOT=<id>` renders the current format locally from the snapshot's files, without changing it.

| Report | Shows |
| --- | --- |
| Delivery | Extracted, delivered, valid, invalid, and accepted counts, label updates, each validation rule's count with example rows, and charts of the valid new transactions |
| Snapshot | Partition timeline and counts, training eligibility, the latest delivery's accounting, the training sample against all eligible rows, training-only fraud exploration, the latest month against the preceding month of equal length, and daily volume, label coverage, and amounts |

The conventions behind the numbers:

- Statistics are SQL aggregates over whole populations, with exact percentiles.
- Daily charts use real calendar dates, including empty days.
- Fraud rates count only matured labels, in seven-day groups, with Wilson intervals that assume independent transactions and so understate uncertainty for fraud episodes.
- Month-to-month changes use total variation distance (0 identical, 1 disjoint). They describe drift but never reject data or trigger training.
- Fraud exploration uses only the training partition and labels released by the cutoff, so it never reveals validation or evaluation outcomes. Amount distributions are normalised per class, and the amount-versus-hour scatter shows at most 20,000 legitimate and 5,000 fraudulent points, sampled by hash and labelled with population counts. Heatmap cells and categories with fewer than 100 rows hide their rates, and the 12 largest MCCs are shown, with the rest as Other.
