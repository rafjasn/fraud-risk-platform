# Measured results

## Final experiment

The model deployed after three monthly updates caught 62.5% of October 2019's frauds while reviewing 0.5% of transactions, on a holdout no training cycle had seen. The first model, trained three months earlier, caught 62.2%. The updates did not measurably improve on it. Two of them were promoted on evaluation-month gains of 4 and 7 frauds, which the holdout shows were within noise. A third, ahead by one fraud, was refused by the 2-point promotion margin.

### Setup

- **Data:** IBM's synthetic transactions, delivered monthly from the 2019-07-08 cutoff through 2019-11-08, with labels released 7 days after each transaction. Each snapshot holds all accepted history (2.75 million to 3.33 million rows) and draws a deterministic 1,000,000-row training sample from 2.29 million to 2.73 million eligible rows (1,350–1,369 frauds). Validation and evaluation windows are 30 days each and uncapped.
- **Models:** six recipes per cycle, each searched with 20 Optuna trials on validation, except the fixed spline recipe: card-history logistic regression, histogram gradient boosting, XGBoost, a PyTorch MLP, spline logistic regression, and transaction-only logistic regression. From the second cycle, the incumbent's configuration is refitted as well when no recipe matches it.
- **Selection and gate:** thresholds fill a 0.5% review budget on validation. A challenger is promoted only if its recall among the top 0.5% of evaluation-month scores beats the incumbent's by at least 2 percentage points. Promotion and rollout confirmation were explicit operator steps.
- **Holdout:** October 2019 (145,074 transactions, 275 frauds), excluded from every cycle and scored once with `make final-evaluation`.

### Monthly cycles

Recall is at the 0.5% review budget. Validation is the month the winner was chosen on; evaluation is the following month, where it faced the incumbent.

| Cutoff | Validation winner | Validation recall | Evaluation month | Candidate | Incumbent | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-07-08 | XGBoost | 71.2% (260 frauds) | June | 52.6% (113 of 215) | – | Initial promotion |
| 2019-08-08 | XGBoost | 56.9% (209) | July | 66.2% (102 of 154) | 63.6% (98) | Promote |
| 2019-09-08 | Histogram boosting | 68.2% (157) | August | 60.0% (141 of 235) | 57.0% (134) | Promote |
| 2019-10-08 | XGBoost | 63.0% (235) | September | 61.3% (87 of 142) | 60.6% (86) | Keep current |

Tree models won every month. On validation, the MLP trailed the winner by 3–4 points, the spline logistic recipe by 3–8, card-history logistic regression by 19–34, and transaction-only logistic regression by 36–47.

### Final holdout: October 2019

| Release | Trained to | Recall at 0.5% | 95% interval | AP | Alert rate at its threshold |
| --- | --- | --- | --- | --- | --- |
| Champion: histogram boosting `020564a3` | 2019-09-08 | 62.5% (172 frauds) | 57.3–67.8% | 0.461 | 0.57% |
| Initial: XGBoost `75c451f5` | 2019-07-08 | 62.2% (171) | 56.5–67.7% | 0.444 | 0.48% |
| August: XGBoost `1798983a` | 2019-08-08 | 61.8% (170) | 57.1–67.7% | 0.463 | 0.54% |

The champion's lead over the initial release was −2.3 to +2.9 points (95% paired bootstrap), and over the August release −2.8 to +2.9. Row resampling understates the uncertainty, because frauds cluster on cards, so these intervals are if anything too narrow. At its saved threshold the champion flagged 821 transactions, 13% more than the budget's 725, with 21.2% precision and 63.3% recall.

### Replayed traffic

Before each month's outcomes were released, the deployed release scored that month's transactions as the API would have. Recall here is at each release's saved threshold, on outcomes released by the 2019-11-08 clock.

| Traffic | Release | Transactions | Mature frauds | Recall at threshold | Same release on its evaluation month | Flagged |
| --- | --- | --- | --- | --- | --- | --- |
| Jul 8 – Aug 8 | `75c451f5` | 147,074 | 198 | 61.6% | 51.2% | 0.46% |
| Aug 8 – Sep 8 | `1798983a` | 146,787 | 226 | 58.4% | 64.9% | 0.51% |
| Sep 8 – Oct 8 | `020564a3` | 141,505 | 151 | 59.6% | 60.9% | 0.44% |
| Oct 8 – Nov 8 | `020564a3` | 145,196 (112,104 mature) | 214 | 62.6% | 60.9% | 0.57% |

Deployed recall stayed between 58% and 63% in every month, and each saved threshold flagged 0.44–0.57% of live traffic against the 0.5% budget. Precision at the threshold was 15–20%.

### What this shows

- The protocol worked end to end: every delivery was accepted, every cycle succeeded on its first attempt, the gate both promoted and refused, and the holdout stayed untouched until the end.
- Retraining on an expanding window gave no measurable holdout gain over three months. The 2-point margin was too small to separate the two promotions from noise: with 142–235 frauds per evaluation month, a difference of 4–7 frauds is within sampling error. Keeping the initial model would have given the same holdout result, so a stricter gate, such as a larger margin or a paired interval that excludes zero, would have lost nothing here.
- One month is a noisy measure of a model. The initial release caught 52.6% of June's frauds, then 61.6% of its replayed month and 62.2% of October's.

### Evaluation context

- **Data reuse:** June and July were reused evaluation periods; August, September, and the October holdout were first evaluated in this experiment. Configuration selection used May validation data only.
- **Build provenance:** jobs record commit `1d43c37` with uncommitted changes, training image `sha256:c37873db…`, and package source hash `e1b93c9fc86924fd17c2ecbba48a491477dc2c724d185e263c4b3cc740a2549b`. The code in this repository is the code that ran: this command reproduces the recorded source hash from `src/`:

```sh
uv run --all-extras python -c "from fraud_risk_platform.training.tracking import source_checksums; from fraud_risk_platform.persistence.files import identity; print(identity(source_checksums()))"
```

### Resources and artifacts

On an 8 GB Mac running Compose, validation took 173 seconds for the initial 2,751,130 records and 70–82 seconds per monthly delivery (about 175,000–180,000 records). Training cycles took 30–43 minutes; replaying about 145,000 transactions took 11–14 seconds. Storage used 1.4 GB in the data bucket and 468 MB in cycle directories, mostly trial models.

The run's reports are in [reports/](reports/README.md): five snapshot and five delivery reports, four training cycles, four monitoring reports, the holdout evaluation, release manifests, and the deployment history. The snapshot and delivery reports were verified against the checksums in their manifests.

## Development findings

Measurements made while building the platform, oldest first. Integration tests use small synthetic fixtures whose perfect model scores are not results; the numbers below come from the IBM data unless stated otherwise.

### Source and images

All 24,386,900 source rows parsed (29,757 frauds), and a full import completed; [ibm-seed-verification.json](ibm-seed-verification.json) records the checks. Splitting dependencies by image shrank the ingestion image from 3.4 GB to 168 MB, and installing CPU-only PyTorch shrank the training image from 9.8 GB to 2.1 GB. The SageMaker container contract passed with networking disabled: prepared channels produced a model bundle, and an invalid job exited with a failure file.

### Card-history features

Fixed default parameters, no search, 400,000 training rows of the July 2019 snapshot, scored on its validation month (141,222 rows, 260 frauds):

| Recipe | Feature set | Validation AP | Train AP | Recall, top 0.5% / 1% |
| --- | --- | --- | --- | --- |
| `logistic_transactions` | Transaction details | 0.031 | 0.055 | 15% / 22% |
| `logistic_behavior` | + card behaviour | 0.056 | 0.077 | 24% / 32% |
| Boosting | + card behaviour | 0.128 | 0.725 | 32% / 40% |
| `logistic_history` | + confirmed card frauds | 0.155 | 0.154 | 38% / 49% |
| `boosting_history` | + confirmed card frauds | 0.389 | 0.830 | 57% / 68% |

Card behaviour helps when modelled non-linearly, and confirmed card frauds add most of the rest. That gain depends on the simulated label delay and on IBM's fraud episodes continuing after confirmation. The logistic rows later proved unreliable, because `lbfgs` stopped early (see [convergence diagnostics](#logistic-convergence-spline-features-and-search-ranges)).

### The review-budget objective

Selection first maximised recall at 90% precision. No logistic model reached 90% precision at any recall, so every logistic trial scored the same, and boosting reached it for only about 1% of fraud. The objective became recall within a 0.5% review budget before any evaluation month was scored.

### First searched models

Twenty-trial searches on the July 2019 snapshot, scored on June (215 frauds). Card history raised logistic recall at 0.5% from 18.1% to 29.8%, and boosting on the same features reached 51.6%. All three models lost 22–28% of their validation recall on June, more than the sampling error of about 3.4 points. June had a lower fraud rate, and selecting the best of 20 trials adds optimism; one month cannot separate the two. June has been reused as an evaluation month since, so it does not decide between models.

### Validation and reports in DuckDB

Snapshot construction moved from SQLite and per-row Python to DuckDB tables and SQL aggregates. On one million synthetic rows, the measured snapshot steps took about 10 seconds instead of 93, with a 1.1 GB peak. Validation findings matched the old implementation's, and the Parquet files it wrote were byte-identical, and report percentiles became exact instead of sampled.

### Training ledger, metrics, SageMaker, and Terraform

- **Job ledger:** integration tests cover a failed attempt then a successful retry, a reused success, a deliberate repeat, a refused concurrent request, an abandoned attempt, and a recorded success whose report is missing.
- **API metrics:** after 600 requests at 40 per second, 24 of them invalid, Prometheus scraped the API and all ten Grafana panels returned data, with the 422s counted as `validation`. That release was a four-row toy model, so this checked the metrics path, not latency.
- **SageMaker path:** a test stages a job's channels in LocalStack, runs the real entrypoint offline, imports the release twice, and promotes it.
- **Terraform:** `make infra-check` passed: 20 mocked-provider test runs, tflint with no issues, and checkov with 250 checks passed. Deliberately breaking a mount, a public-access block, or the schedule's retry policy each failed a test. Running `provision.sql` as a non-superuser found a real error, since PostgreSQL 16 no longer lets a role's creator `SET` it, and the script was fixed.

### Logistic convergence, spline features, and search ranges

Validation-only diagnostics on the July 2019 snapshot (one million training rows, 260 validation frauds); the evaluation month was not scored.

`lbfgs` stops when every gradient component of the mean log loss falls below 1e-4. At a 0.14% fraud rate the fraud-driven components start near that value, so unweighted fits reported convergence after 12–40 iterations, with the objective still 2.5% above its optimum. `newton-cholesky` converged in 9 iterations. Converging raised `logistic_transactions` from 5.5% to 23–24% recall at 0.5%, and unweighted `logistic_history` then matched the balanced-weight model with higher AP. Logistic recipes therefore use `newton-cholesky` without class weights.

Heavy-tailed inputs limit a linear model: `amount` reaches 70 standard deviations. On `card_history`:

| Numeric preprocessing | Validation recall at 0.5% | Validation AP | Train AP |
| --- | --- | --- | --- |
| Standardised (`logistic_history`) | 41.5% | 0.193 | 0.225 |
| Quantile transform to a normal distribution | 57.7% | 0.354 | 0.372 |
| Quantile cubic splines, 8 knots, C = 10 (`spline_history`) | 64.2% | 0.441 | 0.447 |

The selected XGBoost trial reached 71.2% and 0.517 for comparison. Spline results changed little between 5 and 12 knots or C from 0.3 to 100. A spline fit takes about 40 seconds and 1.7 GB on one million rows, so it runs once instead of being searched. Quantile inputs did not help the MLP.

Search ranges follow each parameter's rank correlation with validation results across three earlier cycles. Histogram boosting's L2 regularisation correlated +0.81 to +0.82 with recall, at the top of its old range; L2 = 50 raised AP from 0.498 to 0.513 and cut training AP from 0.848 to 0.709. More XGBoost trees with lower learning rates and child weights gave higher AP, reaching 69.2% recall and AP 0.520 with the new defaults. A lower MLP fraud weight gave higher AP (correlations −0.41 to −0.60). Refits differed by 3–5 validation frauds between the host and the training image, so differences under about three recall points are noise.

### Shared report layout and pooled connections

Every report now renders through one layout and one palette (`reporting/`). Tests check that no other module defines a colour and that a data report renders identical bytes twice, because report checksums are part of each snapshot's identity. The API now keeps a pool of up to 8 connections per database: a query on a new connection took a median 5.30 ms, against 0.81 ms for a pooled checkout. After a database restart the pool recovered on its next query, which it could not do without checking each connection on checkout.
