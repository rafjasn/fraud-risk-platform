# Model card

## Intended use

An educational demonstration of a fraud-detection model's lifecycle on IBM's synthetic card transactions. It is not validated for decisions about real customers or payments.

## Model

The deployed model is card-history histogram gradient boosting (release `020564a3`), trained on data up to the 2019-09-08 cutoff. It gives each transaction a fraud score. A transaction is flagged for review when its score reaches a threshold chosen on validation data to flag 0.5% of transactions. Each month it competed with logistic regression (standardised or spline inputs), XGBoost, and a PyTorch MLP served as ONNX; see [TRAINING.md](TRAINING.md).

**Inputs:** amount, log amount, a negative-amount flag, cyclic hour, payment channel, merchant category, day of week, and nine card-history features from the card's previous 30 days. One of these counts frauds already confirmed for the card. Requests must include `user_id` and `card_id`.

## Training data

IBM's synthetic transactions, delivered monthly. Training covers 2018-01-01 to 2019-07-03 on an expanding window, as a deterministic sample of 1,000,000 rows (1,369 frauds). Validation and evaluation use the following 30-day windows, uncapped. Labels become visible 7 days after each transaction, a simulated delay. October 2019 was held out from every cycle.

## Performance

On the October 2019 holdout (145,074 transactions, 275 frauds), the model caught 62.5% of frauds while reviewing 0.5% of transactions (95% interval 57.3–67.8%), with average precision 0.461. The first deployed model, XGBoost trained three months earlier, caught 62.2%. At its saved threshold the model flagged 0.57% of October's transactions, with 21% precision. While deployed, replayed live traffic gave 59.6–62.6% recall at the threshold. Details: [RESULTS.md](RESULTS.md#final-experiment).

## Limitations

- **Synthetic data.** Labels, fraud patterns, and the label delay are simulated. The confirmed-fraud feature benefits from IBM's fraud episodes continuing after a fraud is confirmed, which real issuers usually stop by blocking the card.
- **Small samples.** Each month has 140–275 frauds. The intervals come from resampling transactions, while frauds cluster on cards, so they understate the uncertainty.
- **Threshold drift.** Thresholds fit one month's budget; on later months they flagged 0.44–0.57% of transactions instead of 0.5%.
- **Uncalibrated scores.** Scores rank transactions; their calibration as probabilities was not checked.
- **Policies, not guarantees.** The 0.5% review budget and the 2-point promotion margin are educational choices, not business-cost or significance thresholds.
