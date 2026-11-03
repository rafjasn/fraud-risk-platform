# Final run reports

Reports from the [final experiment](../RESULTS.md#final-experiment). Each HTML page is self-contained, with its charts embedded, and opens offline; the JSON beside it holds the same report for machines. Snapshot and delivery reports were copied from the run's S3 bucket and verified against the SHA-256 in their manifests. Trial models are not included.

| Folder | File | Contents |
| --- | --- | --- |
|  | [deployment.json](deployment.json) | Promotion, confirmation, and rollback history |
| data | [delivery-2019-07-08.html](data/delivery-2019-07-08.html) | Delivery ending 2019-07-08: 2,751,130 records received |
| data | [delivery-2019-08-08.html](data/delivery-2019-08-08.html) | Delivery ending 2019-08-08: 180,160 records received |
| data | [delivery-2019-09-08.html](data/delivery-2019-09-08.html) | Delivery ending 2019-09-08: 180,349 records received |
| data | [delivery-2019-10-08.html](data/delivery-2019-10-08.html) | Delivery ending 2019-10-08: 174,714 records received |
| data | [delivery-2019-11-08.html](data/delivery-2019-11-08.html) | Delivery ending 2019-11-08: 178,167 records received |
| data | [snapshot-2019-07-08.html](data/snapshot-2019-07-08.html) | Snapshot fe07503ad87d, cutoff 2019-07-08 |
| data | [snapshot-2019-08-08.html](data/snapshot-2019-08-08.html) | Snapshot 01a7e35067e6, cutoff 2019-08-08 |
| data | [snapshot-2019-09-08.html](data/snapshot-2019-09-08.html) | Snapshot 6f4911deaa32, cutoff 2019-09-08 |
| data | [snapshot-2019-10-08.html](data/snapshot-2019-10-08.html) | Snapshot 0706b81d18e9, cutoff 2019-10-08 |
| data | [snapshot-2019-11-08.html](data/snapshot-2019-11-08.html) | Snapshot 2c389fdc1f30, cutoff 2019-11-08 |
| final-evaluation | [report.html](final-evaluation/report.html) | October 2019 holdout, scored once |
| monitoring | [2019-08-08/report.html](monitoring/2019-08-08/report.html) | Replayed traffic at simulated clock 2019-08-08 |
| monitoring | [2019-09-08/report.html](monitoring/2019-09-08/report.html) | Replayed traffic at simulated clock 2019-09-08 |
| monitoring | [2019-10-08/report.html](monitoring/2019-10-08/report.html) | Replayed traffic at simulated clock 2019-10-08 |
| monitoring | [2019-11-08/report.html](monitoring/2019-11-08/report.html) | Replayed traffic at simulated clock 2019-11-08 |
| releases | [2019-07-08-75c451f50e78.json](releases/2019-07-08-75c451f50e78.json) | Release manifest: xgboost_history, promotion eligible True |
| releases | [2019-08-08-1798983a06d5.json](releases/2019-08-08-1798983a06d5.json) | Release manifest: xgboost_history, promotion eligible True |
| releases | [2019-09-08-020564a33e19.json](releases/2019-09-08-020564a33e19.json) | Release manifest: boosting_history, promotion eligible True |
| releases | [2019-10-08-17f16296d6a6.json](releases/2019-10-08-17f16296d6a6.json) | Release manifest: xgboost_history, promotion eligible False |
| training | [cycle-2019-07-08/report.html](training/cycle-2019-07-08/report.html) | Cycle 809488a8c906: initial_promotion, winner xgboost_history |
| training | [cycle-2019-08-08/report.html](training/cycle-2019-08-08/report.html) | Cycle d30dd9df2bdb: promote, winner xgboost_history |
| training | [cycle-2019-09-08/report.html](training/cycle-2019-09-08/report.html) | Cycle 64352377ee54: promote, winner boosting_history |
| training | [cycle-2019-10-08/report.html](training/cycle-2019-10-08/report.html) | Cycle 82ccd9990b56: keep_current, winner xgboost_history |
