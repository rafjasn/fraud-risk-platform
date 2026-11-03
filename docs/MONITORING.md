# Replay and monitoring

The source database holds the whole IBM history, but the platform only learns it month by month through snapshots. Replay simulates the API's live traffic in between: it scores each new month with the deployed release before that month's outcomes are released, then monitoring measures those predictions as outcomes mature.

## The monthly loop

The monthly coordinator replays traffic, publishes a snapshot, monitors outcomes and trains, then stops at the recommendation. See [Operations](OPERATIONS.md#the-simulation) for commands and overrides.

- **Waits for your promotion decision:** it refuses to start a month while the last cycle's recommended promotion has not been made. Use `make month SKIP_PROMOTION=1` to continue with the deployed release instead.
- **Safe to rerun:** a month already replayed is not replayed again. If training failed after the snapshot advanced, the next `make month` resumes at monitoring and training for that snapshot instead of starting another month, as the next attempt of the same [training job](TRAINING.md#training-jobs).
- **Starts from scratch too:** right after `make initial-snapshot`, it trains the initial cycle.
- **Stops at the end of the data:** it reports `complete` once the simulation reaches the end of the source data.
- **Stops at the holdout:** a snapshot whose evaluation window reaches the reserved final holdout (October 2019 by default) is never trained. The month replays, publishes, and monitors as usual, then reports `holdout_reached`. Later runs stop there again, because that snapshot never gets a cycle. Score the holdout with `make final-evaluation`.
- **Can stop before training:** `month --stop-before-training` returns the snapshot to train instead of training it. The AWS workflow uses it and trains on SageMaker; see [infra/terraform](../infra/terraform/README.md).

## What a replay does

`make replay` reads the simulated clock, which is the cutoff of the latest snapshot in the current configuration's monthly chain, and the deployed release from `deployment.json`. It scores every source transaction from the clock up to the next monthly cutoff (clamped to the source horizon):

- **Same scoring as the API:** the release's model, the shared feature code, and point-in-time card history. Labels count only once released at each transaction's own time. An integration test checks that a replayed score equals the API's score for the same transaction.
- **Never reads the month's outcomes:** the transaction query selects model inputs only. The history query never returns a label released after the window.
- **Before outcomes exist:** it runs only while the clock is at the month's start. A month can be replayed once per simulation; after `make next-snapshot`, the next replay covers the following month, so earlier months cannot be rescored retrospectively.
- **Atomic:** the replay record and all predictions are written in one transaction, so a failure leaves nothing to clean up.
- **Rejections are counted:** a transaction missing a required field is counted by field, as the API would return HTTP 422.

Replayed predictions go into the same `predictions` table as API requests, tagged with `replay_id`. `predicted_at` is when the replay ran; the transaction time is the simulated prediction time.

## What monitoring reports

`make monitor` writes `state/platform/monitoring/<clock>/report.html`, `report.json`, and `monitoring.png`. For each replayed month it shows the release, the review volume, and outcomes so far:

- **Review volume:** flagged transactions and the daily share flagged, against the review budget.
- **Mature and pending outcomes:** only outcomes released by the clock count. With the seven-day label delay, the last week of the latest month is still pending; it counts in the next report.
- **Precision and recall at the release's threshold,** and recall at the review budget ranked within mature transactions. Each sits next to what the same release achieved on its evaluation month.
- **Weekly recall on mature outcomes,** because daily fraud counts are too small to read.

Batch replay records scoring throughput, not API latency; see below.

## Final holdout

The holdout is the calendar month from `final_holdout_start` in `config/data.json`: October 2019, the last IBM month with frauds. No training cycle fits, selects, or evaluates on it. After `holdout_reached` and confirmed champion rollout, the [final evaluation](OPERATIONS.md#the-simulation) scores the confirmed champion on the holdout month, beside every release confirmed serving before it, so the report shows both the final model and whether the monthly updates improved on the first one. Nothing is chosen on this month. The run:

- **Reads a verified snapshot:** the latest one, or `SNAPSHOT=<id>`, whose released labels must cover the whole month. The 2019-11-08 snapshot that `make month` stops at covers October exactly. Features use the training code and card history from that snapshot's context.
- **Reports the gate's measure:** recall among the top 0.5% of scores, with paired bootstrap intervals and the champion's lead over each earlier release, plus AP and precision, recall, and alert rate at each saved threshold.
- **Runs once:** it writes `state/platform/final-evaluation/report.html` and `report.json`, and refuses to run again, so the holdout cannot be rescored after a change.

## API latency and errors

Batch replay measures scoring throughput. The [API load command](OPERATIONS.md#serving-and-observability) measures HTTP requests separately.

- **What it sends:** a deterministic sample of the month after the simulated clock (the transactions with the lowest MD5 of their source row), in transaction-time order, with every source field the API accepts. The API builds card history as of each transaction's time, so no outcomes leak. Its predictions are stored like any API prediction, without a `replay_id`, so they never enter replay monitoring.
- **Fixed schedule:** request *n* is sent at *n*/rate seconds, whether or not earlier responses have returned. A slow API therefore delays later requests rather than hiding its own latency, and the report shows how far sending fell behind.
- **Requires a ready API:** it stops if `/ready` fails. It does not change the simulated clock, replay, or deployment.

`state/platform/load/<UTC time>/report.html` (with `report.json` and `latency.png`) shows two views:

- **Client:** statuses, the server-error rate (5xx or no response), the client-error rate (4xx), and latency percentiles of successful requests, including HTTP and Docker networking.
- **Server:** the API's own metrics, read before and after the run: requests by status, errors by cause, and mean time per step. These include any other requests the API served meanwhile.

The report links to the same period in Grafana.

### Metrics and dashboard

The API exposes Prometheus metrics at `/metrics`, and Prometheus scrapes it every 5 seconds. The API keeps serving whether or not Prometheus runs.

| Metric | Labels | Meaning |
| --- | --- | --- |
| `fraud_api_request_seconds` (histogram) | `endpoint`, `status`, `release` | Server time per request; unknown paths share `endpoint="unmatched"` |
| `fraud_api_step_seconds` (histogram) | `step`, `release` | `history` (card-history query, history releases only), `scoring` (features and model), `persistence` (source check and prediction insert) |
| `fraud_api_errors_total` | `endpoint`, `cause`, `release` | `validation` (422), `conflict` (409), `no_model` and `storage` (503; a database read or write failed), `unhandled` (500), otherwise `http_<status>` |
| `fraud_api_predictions_total` | `release`, `flagged` | Recorded predictions and review decisions |
| `fraud_api_release_info` | `release` | 1 for the release the process serves, or `none` |

The error rate counts 5xx responses only. 4xx responses are request or data problems, counted by cause. The provisioned Fraud API dashboard is linked from [Operations](OPERATIONS.md#serving-and-observability). Its p50/p95/p99 values are estimated from histogram buckets; the load report's client percentiles are exact. Prometheus keeps 30 days of samples in the `prometheus` Docker volume.

The API keeps a pool of up to 8 connections to each database for its lifetime, so the `history` and `persistence` steps time queries, not connection setup. A request waits at most 2 seconds for a connection, then fails with `storage` (503); each checkout first checks its connection, so the API recovers on its own after PostgreSQL restarts.

The recorded load measurements use one Uvicorn worker sharing a laptop with PostgreSQL; comparisons should keep that resource configuration in view. On AWS, the Terraform definition's load balancer reports latency and 5xx counts to CloudWatch, with alarms on both; the per-step histograms would need CloudWatch embedded metrics or Amazon Managed Prometheus.
