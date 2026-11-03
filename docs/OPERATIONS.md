# Operating the platform

Every command runs through `make`, in Docker containers. Data, training and monitoring behavior are explained in [DATA.md](DATA.md), [TRAINING.md](TRAINING.md), and [MONITORING.md](MONITORING.md); this page lists the commands.

## Setup

You need Docker with Compose v2 and Make. No AWS or LocalStack account is needed.

| Command | Does |
| --- | --- |
| `make setup` | Build the ingestion, training, and serving images; rebuild after changing code or configuration |
| `make setup-data` | Build only the small ingestion image, for seeding and snapshots |
| `make up` | Start PostgreSQL, LocalStack S3, MLflow, the API, Prometheus, and Grafana |
| `make down` | Stop everything; the PostgreSQL volume and `state/` stay |
| `make logs` | Show recent API and MLflow logs |

The ingestion image holds only data and reporting dependencies. The training image holds every model family with CPU-only PyTorch (about 2.1 GB). The API image serves every model type with scikit-learn, XGBoost, and ONNX Runtime, without PyTorch.

Seeding loads all 24,386,900 IBM transactions. The archive is 279 MB and stays cached in `data/raw/`; the PostgreSQL volume grows to about 15 GB. Training uses a deterministic sample of at most 1,000,000 rows per snapshot; set `dataset.max_train_rows` in `config/data.json` to another number, or `null` for all eligible rows, then rebuild before the first snapshot.

## The simulation

| Command | Does |
| --- | --- |
| `make seed-source` | Download, verify, and load the IBM transactions. A rerun resumes or reuses the import |
| `make initial-snapshot` | Publish the first snapshot (cutoff 2019-07-08), or return it if it exists |
| `make next-snapshot` | Deliver the next month and publish its snapshot, or report that no data is left |
| `make month` | Replay the coming month, publish the next snapshot, monitor, and train. `TRIALS=5` shortens searches; `SKIP_PROMOTION=1` continues without a recommended promotion |
| `make replay`, `make monitor` | The month's replay and monitoring steps on their own |
| `make final-evaluation` | Score the confirmed champion on the reserved holdout month, once; optionally select `SNAPSHOT=<id>` |
| `make reset-simulation CONFIRM=yes` | Start a new simulation: empty the clock, ledgers, replays, and predictions, rename the MLflow experiment and model, and archive `state/platform`. The source import stays |

A typical run: `make initial-snapshot`, then `make month` to train the first cycle, `make promote RELEASE=<id> INITIAL=--initial`, and `make month` again for each month. Promote when a month recommends it. When a month reports `holdout_reached`, run `make final-evaluation`.

## Snapshots and data reports

| Command | Does |
| --- | --- |
| `make snapshots` | List published snapshots |
| `make verify-snapshot SNAPSHOT=<id>` | Replay a snapshot's context and re-derive every partition |
| `make snapshot-report SNAPSHOT=<id>` | Download its HTML report to `state/platform/reports/` |
| `make refresh-snapshot-report SNAPSHOT=<id>` | Render the current report format for an existing snapshot locally, without changing it |

To see validation reject a delivery, inject a fault into the next month and then retry it clean:

```sh
make next-snapshot SNAPSHOT_ARGS='--fault missing-amount'   # rejected; prints its report URI
make next-snapshot                                          # the corrected retry accepts the same month
```

The other faults are `invalid-amount`, `duplicate`, and `conflict`. They change only the delivered copy, never the source database.

## Training and deployment

| Command | Does |
| --- | --- |
| `make train SNAPSHOT=<id>` | Run a cycle. `RECIPE=<name>`, `TRIALS=<n>`, `REFIT_SNAPSHOT=<uncapped-id>`, and `EXPERIMENT=<name>` adjust it |
| `make bootstrap SNAPSHOT=<id>` | Fit the initial recipe once, unsearched |
| `make training-jobs` | List recent training jobs with every attempt |
| `make promote RELEASE=<id>` | Deploy a release its cycle recommended; add `INITIAL=--initial` for the first one |
| `make rollout` | Restart the API on the selected release and confirm it, or restore the last confirmed one |
| `make rollback` | Return to the previously confirmed release |
| `make status` | Show the selected and deployed releases |
| `make sagemaker-inputs SNAPSHOT=<id>`, `make import-release ARCHIVE=<uri>` | Stage a SageMaker job's inputs, and import its release |

### SageMaker inputs and output

From the repository root:

```sh
docker build --target training-sagemaker -t fraud-risk-training-sagemaker:local .
make sagemaker-inputs SNAPSHOT=<id>  # accepts RECIPE, TRIALS and REFIT_SNAPSHOT
# After the training job completes:
make import-release ARCHIVE=s3://<output-bucket>/<job>/output/model.tar.gz
```

The [training guide](TRAINING.md#sagemaker-training) defines channel contents, environment variables and archive verification. The [Terraform guide](../infra/terraform/README.md) covers AWS configuration and the workflow.

## Serving and observability

| Service | Address |
| --- | --- |
| API and its docs | http://127.0.0.1:8000/docs |
| MLflow | http://127.0.0.1:5001 |
| Prometheus | http://127.0.0.1:9090 |
| Grafana, Fraud API dashboard | http://127.0.0.1:3000 (read-only without login; admin password `local-grafana`) |

With a ready API, send a fixed-rate sample of the coming month:

```sh
make load  # 2,000 requests at 20/s over 4 connections
make load LOAD_REQUESTS=500 LOAD_RATE=50 LOAD_CONCURRENCY=8
make outcomes AS_OF=<date>  # predictions joined to outcomes released by this date
```

Load reports are written to `state/platform/load/`; [Monitoring](MONITORING.md#api-latency-and-errors) explains their metrics. In API requests, `transaction_id` is the source CSV's row number; a known transaction must match its source values.

## Databases

One PostgreSQL server, at `127.0.0.1:5433` from the host (`POSTGRES_PORT` changes it), holds three databases. These are local development credentials:

| Database | User / password | Contents |
| --- | --- | --- |
| `fraud_source` | `source_reader` / `local-source-reader` (read-only), `source_app` / `local-source` | IBM transactions and import status |
| `fraud_platform` | `platform_app` / `local-platform` | Simulation clock, ingestion and training ledgers, predictions, replays |
| `mlflow` | `mlflow_app` / `local-mlflow` | MLflow tracking and registry |

`make db-init` creates the roles, databases, and tables without loading data; `infra/postgres/init/01-databases.sql` defines the roles and grants.

## Recovery

- **LocalStack lost its data:** S3 lives in LocalStack's memory and is lost on restart. Run `make initial-snapshot` to start a fresh stream from the source database, then continue. Rebuilt snapshots get new IDs.
- **A delivery was rejected:** its raw data and report are kept, and progress does not move. Fix the cause and rerun the same command.
- **A snapshot failed to publish:** rerun the command; the accepted delivery is reused.
- **A training cycle failed:** rerun it; it runs as the job's next attempt.

## Kubernetes

The [Helm chart](../charts/fraud-risk-platform/README.md) runs the same images on a local kind cluster, in namespace `fraud-risk`.

| Command | Does |
| --- | --- |
| `make k8s-images` | Load the images `make setup` built into the kind cluster |
| `make k8s-install` | Install or upgrade the chart; waits for database setup |
| `make k8s-task ARGS="<command>"` | Run any platform command as a Job and follow its logs, for example `ARGS=seed-source` |
| `make k8s-month` | Run one simulated month from the chart's CronJob |
| `make k8s-promote RELEASE=<id>`, `make k8s-rollback` | Select a release, then roll the API onto it with a readiness-gated rolling update |
| `make k8s-uninstall` | Remove the release; the database and state volumes stay |

## Checks

| Command | Runs |
| --- | --- |
| `make test` | All tests in the test image, against a disposable PostgreSQL and LocalStack |
| `make unit` | Unit tests on the host with `uv`; on macOS, XGBoost needs `brew install libomp` |
| `make lint` | Ruff lint and format checks |
| `make check-ingestion-image` | That the ingestion image works and contains no model libraries |
| `make infra-check` | Terraform format, validation, mocked-provider tests, tflint, and checkov, without an AWS account |
| `make chart-check` | Helm lint, and Kubernetes schema validation of every manifest the chart renders |
