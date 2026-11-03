# Training

A training cycle reads one verified snapshot, searches every configured recipe on its train and validation partitions, and selects one winner. Only then does it score that frozen winner against the deployed model on the later evaluation partition. A cycle never queries the source database, advances simulated time, or deploys a model.

## Cycle configuration

See [Operations](OPERATIONS.md#training-and-deployment) for training, bootstrap and release commands.

`make month` runs the same cycle each simulated month (see [MONITORING.md](MONITORING.md)). Without `RECIPE`, a cycle runs `initial_recipe` and then `challengers` from `config/training.json`: `logistic_history`, `boosting_history`, `xgboost_history`, `mlp_history`, `spline_history`, and `logistic_transactions`. Each recipe with a search space gets up to 20 sequential Optuna trials on the same data. Keep recipes that share a feature set next to each other, because features are rebuilt whenever the set changes. When a release is deployed, the cycle also refits its exact configuration as `incumbent_retrained`, unless a configured recipe already matches it.

Training containers use `TRAINING_THREADS` threads (6 by default; reduce it when Docker has fewer CPUs). Logistic regression and MLP trials are capped at 2 threads because additional threads slowed these fits. Each run records its thread settings for reproducibility.

Progress goes to stderr as timestamped lines: the integrity check, feature loading, each trial's parameters, duration, and metrics, the winner, evaluation, and the recommendation. Metrics appear as `name=value` fields, for example `validation_average_precision=0.517`.

## Recipes

| Recipe | Model | Inputs | Searched (fixed values in brackets) |
| --- | --- | --- | --- |
| `logistic_history` | Logistic regression, `newton-cholesky` | `card_history`, standardised numerics, one-hot categories | `C` 0.1–100 |
| `spline_history` | Logistic regression, C = 10 | `card_history`, quantile cubic splines | Nothing: one fixed fit |
| `logistic_transactions` | Logistic regression | `transaction_details` | `C` 0.1–100 |
| `boosting_history` | scikit-learn histogram gradient boosting, 200 iterations | `card_history`, ordinal categories | Leaves 15–127, learning rate 0.02–0.15, L2 1–300 (minimum leaf 20) |
| `xgboost_history` | XGBoost `hist` | `card_history`, ordinal categories | Trees 200–2,000, learning rate 0.01–0.15, depth 3–10, child weight 0.1–20, L2 1–100 (row and column sampling 0.8) |
| `mlp_history` | PyTorch MLP, served as ONNX | Same inputs as `logistic_history` | Width 16–256, depth 1–3, dropout, learning rate, epochs 3–20, fraud weight 1–20 (weight decay 1e-5, batch 1,024) |

`logistic_basic`, `boosting_basic`, `boosting_transactions`, and `logistic_behavior` stay configured for single runs with `RECIPE`. Every search evaluates the recipe's configured values as trial 0; they must lie inside the search ranges, so `TRIALS=1` fits exactly the defaults. Before any search starts, the cycle checks every parameter name against its estimator, so a misspelled setting fails immediately instead of being ignored by XGBoost.

- **Logistic regression** uses `newton-cholesky` without class weights. `lbfgs` reported convergence well short of the optimum on these features ([results](RESULTS.md#logistic-convergence-spline-features-and-search-ranges)). `spline_history` sets `"numeric_encoding": "quantile_spline"`: each continuous feature is mapped onto [0, 1] through its training quantiles (from up to 200,000 rows), then through a cubic B-spline basis with 8 knots, so the model fits a smooth curve per feature. 0/1 indicators pass through. A spline fit takes about 40 seconds on one million rows, so it runs once instead of being searched.
- **Histogram boosting** needs large L2 values: at a 0.14% fraud rate each row adds little curvature to a leaf, and unregularised trees overfit.
- **XGBoost** imputes numeric gaps with training medians and splits ordinal-encoded categories as categories; unseen categories count as missing. The number of trees is searched rather than early-stopped, because the validation month also selects the winner. Images install `xgboost-cpu`; on macOS, host tests that fit XGBoost need `brew install libomp`.
- **The MLP** has hidden layers of linear, ReLU, and dropout, and one output logit. It trains with AdamW on a binary cross-entropy loss that weights frauds by `pos_weight`, for a fixed number of epochs. Each epoch's training and validation loss are logged as learning curves; validation loss never stops training. The network is then exported to ONNX, and the trial fails unless ONNX Runtime reproduces PyTorch's outputs on up to 1,000 rows. Validation metrics, releases, and the API all use the ONNX graph, so serving never installs PyTorch. A default 10-epoch trial on one million rows took 68 seconds, with a 1.85 GiB peak.

Every trial saves its model files, metrics, and checksums, then reloads them and must reproduce its own validation scores. A trial that fails to fit fails the cycle; a fit that does not converge is saved but cannot qualify. A failed cycle restarts its searches rather than resuming them.

## Feature sets

`config/training.json` defines named `feature_sets`, and each recipe picks one with `feature_set`. Each trial and release stores its resolved definition, so changing the configuration never reinterprets a saved model, and the incumbent is always scored with its own definition. Recipes with different feature sets can run in one cycle. Change the configuration, then rebuild the images: Compose runs the configuration copied into each image.

```json
"transaction_details": {
  "version": "transaction-v1",
  "numeric": ["amount", "log_abs_amount", "is_refund", "hour_sin", "hour_cos"],
  "categorical": ["use_chip", "mcc", "day_of_week"]
}
```

| Kind | Supported features |
| --- | --- |
| Numeric | `amount`, `log_abs_amount` (log1p of the absolute amount), `is_refund` (negative amount), `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `is_weekend` |
| Categorical | `use_chip`, `mcc`, `merchant_city`, `merchant_state`, `postal_code`, `merchant_id`, `day_of_week` (Monday `"0"` to Sunday `"6"`) |
| Card history | The nine features [below](#card-history-features); version `history-v1` |

Recipes without a feature set use `basic-v1`: amount and cyclic hour. Merchant IDs, MCCs, and postal codes stay strings inside the model, keeping signs and leading zeros. Labels, outcome times, `errors`, and arbitrary columns cannot be selected.

Preprocessing is fitted on training rows only. Logistic regression and the MLP impute and scale numerics and one-hot encode categories, with a reserved `__MISSING__` token; an unseen category gives zeros. The tree models ordinal-encode categories, treating unseen and missing values as missing.

The release's feature set decides what `/predict` requires. For `transaction_details`:

```json
{
  "transaction_id": 123,
  "amount": 42.50,
  "timestamp": "2019-07-08T12:30:00",
  "use_chip": "Chip Transaction",
  "mcc": "5411"
}
```

A missing required field returns HTTP 422. `merchant_state` and `postal_code` may be null; other selected fields may not. `mcc` must be a four-digit string and `merchant_id` a signed integer. The fields used for scoring are stored in `predictions.feature_inputs` and must match the source transaction when it is known.

### Card-history features

Feature sets with version `history-v1` can add features describing the same card (`user_id`, `card_id`) before the scored transaction:

| Feature | Definition (the window ends just before the transaction) |
| --- | --- |
| `card_count_1h`, `card_count_24h`, `card_count_7d` | The card's transactions in the last hour, day, and week |
| `card_amount_sum_24h` | Sum of the card's amounts in the last day |
| `amount_vs_card_mean_30d` | log1p of the absolute amount, minus the same for the card's 30-day mean; missing without history |
| `hours_since_card_last` | Hours since the card's previous transaction within 30 days; missing otherwise |
| `new_merchant_for_card_30d`, `new_mcc_for_card_30d` | 1 when the card has not used this merchant or MCC in 30 days |
| `card_confirmed_frauds_30d` | The card's frauds in the last 30 days whose labels were released by this transaction's time |

Training and serving share one implementation ([features/history.py](../src/fraud_risk_platform/features/history.py)). Only strictly earlier transactions count, every lookup stays within 30 days, and a label counts only once `label_available_at` is at or before the scored transaction. Training reads the snapshot's `context` partition, which must reach at least 30 days before training starts. The API reads the card's 30-day window from the source database, with labels masked at the request time. Tests check that both give the same values.

`card_behavior` holds the first eight features and `card_history` all nine. `card_confirmed_frauds_30d` depends on the simulated seven-day label delay and on IBM's synthetic fraud episodes continuing after a fraud is confirmed; real issuers usually block such a card. Compare against `card_behavior` before attributing its gain to real behaviour.

Requests to a card-history model must include `user_id` and `card_id`. Building these features for one million training rows peaked at about 1.1 GiB before fitting, so give Docker 6–8 GB.

## Selection and the promotion gate

Before fitting, training checks the snapshot's integrity: manifest identity, checksums, Parquet schemas and counts, that every row lies in its partition's window, that labels are present exactly when released by the cutoff, class minimums, and that the evaluation window stays before the final holdout. `make verify-snapshot` additionally replays the context and re-derives every partition. Train and validation features are built once per feature set; evaluation features are built only after the winner, and any refit, are frozen.

The objective is recall within a review budget: `review_budget` (0.5%) is the share of transactions analysts can review. Each trial's threshold is the lowest score that flags at most that share of validation transactions; a tie group straddling the budget stays unflagged. A trial qualifies when it converged and catches more fraud than random review would, that is, when its validation recall at the threshold exceeds the budget. Optuna maximises that recall, scoring unqualified trials 0. The winner is the qualified trial with the highest recall, then precision, then the earlier trial. Recall at 90% precision was tried first and dropped before any evaluation month was scored: at a 0.2% fraud rate no logistic model reached it, and boosting reached it for about 1% of fraud.

The winner is loaded from its saved files, not retrained. It is scored on the evaluation partition beside the frozen incumbent, and the gate compares their recall among the top 0.5% of evaluation scores. The candidate must beat random review and beat the incumbent by `minimum_recall_gain`, 0.02. An evaluation month has about 140–235 frauds, so one fraud is 0.4–0.7 points; the margin filters out one- and two-fraud wins. It guards against churn rather than testing significance: when two models disagree on 20 frauds, the standard error of their difference is already about three points. Precision, recall, and alert rate at each saved threshold are reported but not gated. A cycle never searches the evaluation results for another winner, and `keep_current` and `no_initial_model` are normal outcomes.

## Metrics and reports

Every trial records threshold-free validation metrics, qualified or not:

| MLflow metric | Meaning |
| --- | --- |
| `validation_average_precision` | `sklearn.metrics.average_precision_score`; random scores achieve the fraud rate |
| `validation_recall_at_top_{0.1,0.5,1}pct`, `validation_precision_at_top_…` | Results when reviewing that share of the highest scores, with tied boundary scores split evenly; the review budget is always included |
| `validation_roc_auc` | Secondary: with rare fraud it stays high for models of little use |
| `train_average_precision` | On a fixed sample of up to 200,000 training rows; a large gap to validation indicates overfitting |

Trials with a threshold also log precision, recall, confusion counts, and alert rate at it. F1 and log loss are not logged: the threshold is not chosen by F1, and the MLP's fraud weight deliberately distorts probabilities.

MLflow holds a parent run per cycle, a child run per search, and a child run per trial. Every trial is logged as an MLflow model whose signature takes the named features and returns `fraud_score`. The selected candidate is registered under `fraud-risk` with the `candidate` alias and a qualification tag; a registry version is not a deployment.

Each cycle writes `state/platform/cycles/<run-id>/report.json` and `report.html`, also logged to MLflow. Each chart is titled with its partition, period, and row and fraud counts:

- `validation.png`: the selected model's precision–recall curve, frauds caught by review volume, score distributions, threshold curves, and confusion matrix.
- `comparison.png`: every recipe's representative trial on validation, colored in the configured recipe order.
- `trials.png`: every trial's validation recall at the budget, and train versus validation average precision.
- `learning_curves.png`: the MLP's loss per epoch.
- `evaluation.png`: the candidate and incumbent on the evaluation partition.

The selection table adds 95% intervals for each recipe's recall and for the winner's lead over each other recipe, from 1,000 paired resamples of validation rows. Frauds cluster on cards, so these intervals are too narrow; a lead whose interval includes zero is a tie.

## Training jobs

`make train`, `make bootstrap`, and `make month` run each cycle as a job in the platform database (`training_jobs` and `training_attempts`). A job's ID is the hash of its request: the snapshot and refit snapshot, the deployed incumbent, the training policy, recipe and trial overrides, the package's source checksums, dependency versions, `TRAINING_IMAGE_ID`, and an optional experiment name.

- **Succeeded:** an identical request returns the existing report with `"reused": true` and trains nothing. A change to the configuration, code, incumbent, or snapshot makes a new job; `EXPERIMENT=<name>` repeats an identical one deliberately.
- **Failed:** the next identical request runs as attempt 2, 3, and so on. Each attempt keeps its error, its MLflow run (tagged `training_job_id` and `training_attempt`), and its failure report.
- **Running:** a PostgreSQL advisory lock held for the whole attempt refuses an identical request. A process that dies releases the lock with its connection, and the next request records that attempt as `abandoned`.
- **Missing report:** a recorded success whose report has left the state directory fails the command instead of retraining silently.

The database allows at most one running and one succeeded attempt per job. `make training-jobs` lists recent jobs with all their attempts.

## Promotion and rollout

Commands: [promotion, rollout and rollback](OPERATIONS.md#training-and-deployment).

Promotion refuses a release unless its cycle completed, recommended it, and compared it with the release still deployed. It selects the release in `deployment.json`, restarts the API, and runs `verify-deployment` (also `make rollout`). That waits up to two minutes for `/ready` to report the release, which the API does only after verifying its files and reproducing its stored sample scores. Only then is the release recorded as `deployed`, and the registry's `champion` alias moves to it. If the API does not serve it in time, the last confirmed release is selected and confirmed again, and the command fails. Rollback can only return to a release that was once confirmed serving. A registry outage never blocks a confirmed deployment; rerunning `make rollout` retries the alias.

## Optional uncapped refit

Set `REFIT_SNAPSHOT=<uncapped-id>` when [starting a cycle](OPERATIONS.md#training-and-deployment).

After validation selects a configuration, a separate run refits it on all eligible training rows of a published uncapped snapshot. It fits the preprocessing again and selects a new threshold on the unchanged validation partition. The refit snapshot must match the capped one in source, cutoff, windows, label delay, context, and validation and evaluation rows. A refit that fails or does not qualify never falls back to the sampled model. Uncapped still excludes future rows, unreleased labels, and the holdout. `make month` does not refit.

## Storage and provenance

Runs use the `fraud-risk-training-v2` experiment, whose artifacts go to `s3://fraud-mlflow/training-v2` in LocalStack; metadata and the registry stay in PostgreSQL. To change an experiment's artifact location, choose a new experiment name. S3 holds each trial's authoritative model bundle, while a release keeps a verified local copy for serving. LocalStack is disposable, so after it restarts MLflow metadata can point at lost artifacts.

`scripts/compose.sh` passes the current commit, and whether the working tree has uncommitted changes, as build arguments, which the training image stores as `TRAINING_GIT_COMMIT` and `TRAINING_GIT_DIRTY`. It passes the training image's ID to every job as `TRAINING_IMAGE_ID`. Each run's `environment/provenance.json` records these with dependency versions and thread settings, and `source/checksums.json` records every source file's SHA-256, so code changed after a build is still identifiable. Commit before `make setup` to record a clean build.

## SageMaker training

The same cycle implements SageMaker's File-mode container contract. Image build and transfer commands are in [Operations](OPERATIONS.md#sagemaker-inputs-and-output).

| Path | Contents |
| --- | --- |
| `/opt/ml/input/data/config/job.json` | `policy` (the full training policy), optional `recipe` and `trials`, and `snapshot_id` with optional `refit_snapshot_id` |
| `/opt/ml/input/data/snapshot/`, `…/refit/` | The snapshot's manifest and objects at their original keys |
| `/opt/ml/input/data/incumbent/` | The verified incumbent release, when one exists |
| `/opt/ml/model/` | The selected release with its cycle report under `cycle/`, or an outcome marker when nothing was selected |
| `/opt/ml/output/failure` | The error text, with a nonzero exit code |

Set `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_URI` for a new experiment, `TRAINING_IMAGE_ID`, `TRAINING_GIT_COMMIT`, and `TRAINING_GIT_DIRTY`, and set `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to the instance's CPUs. Logs go to CloudWatch, and the Terraform workflow's metric definitions parse the `name=value` fields. `prepare-training-input` exports a verified snapshot as a local channel, which `train --snapshot` also accepts.

The Step Functions workflow in [infra/terraform](../infra/terraform/README.md) stages inputs, starts the job and imports its output.

`sagemaker-inputs` writes the job's channels under `training-inputs/<job-id>/` in the data bucket and prints them as `InputDataConfig`. Each snapshot channel is a SageMaker manifest file listing exactly that snapshot's keys, because the shared `objects/` prefix holds every snapshot. The job ID is the [training job](#training-jobs) identity, so the workflow can use it as its execution name and start an identical request once.

`import-release` verifies the job's `model.tar.gz` and refuses paths outside the archive, a missing cycle report, or a release that fails verification. It places the release and report where `make promote` and `make month` find them; a job that selected nothing still records its cycle. Release bundles contain joblib files, which execute code when loaded, so import only archives from the project's own training output.

A local integration test runs the whole path: it stages channels in LocalStack, downloads them as File mode does, runs the real entrypoint without network access, imports the release, and promotes it.
