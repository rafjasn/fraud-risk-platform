"""Estimator construction and the model release lifecycle.

Where each model family lives:

- Logistic regression, histogram gradient boosting, and XGBoost: one builder each in
  recipes.py, registered in recipes.BUILDERS, with the estimator's class in
  recipes.estimator_class for parameter checks. adapters.SklearnAdapter fits,
  scores, and saves all three with joblib.
- MLP: neural.py trains it with PyTorch, exports it to ONNX, and its NeuralAdapter
  scores it with ONNX Runtime.

releases.py saves and verifies release bundles; deployment.py records which release
serves. A new estimator also needs its name in contracts/training.py (Recipe), any
extra runtime package in releases.runtime_versions and training/tracking.py
(log_trial), and optionally a thread cap in training/trial.py.
"""
