"""Fit one configuration. This layer receives no evaluation data or deployment state."""

import warnings
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from fraud_risk_platform.contracts.training import Recipe
from fraud_risk_platform.evaluation.metrics import (
    metrics,
    ranking_metrics,
    select_threshold,
)
from fraud_risk_platform.features.transaction import (
    artifact_feature_spec,
    model_feature_spec,
)
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.persistence.files import (
    read_json,
    sha256,
    verify_files,
    write_json,
)

# Training AP is an overfitting check, so a fixed sample keeps it cheap on large
# partitions; scoring all one million rows took 1.5-6 seconds per trial.
TRAIN_AP_ROWS = 200_000
# Paired over the same 20 trials on one million rows, 6 threads fit the tree models
# about 26% faster than 2, but logistic regression about 20% and the MLP about 21%
# slower. Capping covers OpenMP and BLAS, which PyTorch's own setting does not.
CAPPED_FIT_THREADS = {"logistic": 2, "mlp": 2}


def load_trial(directory: Path):
    metadata = read_json(directory / "trial.json")
    verify_files(directory, metadata["files"])
    model = adapter_for(metadata["recipe"]["estimator"]).load(directory)
    spec = artifact_feature_spec(metadata)
    if (
        model_feature_spec(model) != spec
        or Recipe.model_validate(metadata["recipe"]).features != spec
    ):
        raise ValueError("Trial model and metadata use different feature definitions")
    return model


def fit_trial(*, recipe: Recipe, train, validation, review_budget: float, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    adapter = adapter_for(recipe.estimator)
    started = perf_counter()
    limit = CAPPED_FIT_THREADS.get(recipe.estimator)
    with (
        warnings.catch_warnings(record=True) as captured,
        threadpool_limits(limits=limit) if limit else nullcontext(),
    ):
        warnings.simplefilter("always", ConvergenceWarning)
        model = adapter.fit(recipe, *train, validation=validation)
    elapsed = perf_counter() - started
    convergence = [
        str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)
    ]
    started = perf_counter()
    scores = adapter.scores(model, validation[0])
    inference_seconds = perf_counter() - started
    threshold = select_threshold(scores, review_budget=review_budget)
    at_threshold = (
        metrics(validation[1], scores, threshold) if threshold is not None else None
    )
    ranking = ranking_metrics(validation[1], scores, review_budget=review_budget)
    # A large gap to validation indicates overfitting rather than weak features.
    rows = len(train[1])
    sample = (
        np.sort(np.random.default_rng(0).choice(rows, TRAIN_AP_ROWS, replace=False))
        if rows > TRAIN_AP_ROWS
        else slice(None)
    )
    train_average_precision = float(
        average_precision_score(
            train[1].iloc[sample], adapter.scores(model, train[0].iloc[sample])
        )
    )
    adapter.save(model, output)
    metadata = {
        "recipe": recipe.model_dump(),
        "feature_version": recipe.features.version,
        "features": recipe.features.columns,
        "feature_spec": recipe.features.model_dump(),
        "review_budget": review_budget,
        "threshold": threshold,
        "converged": not convergence,
        "warnings": convergence,
        # Qualified models catch more fraud within the budget than random review.
        "qualified": threshold is not None
        and not convergence
        and at_threshold["recall"] > review_budget,
        "validation": at_threshold,
        "validation_ranking": ranking,
        "validation_average_precision": ranking["average_precision"],
        "train_average_precision": train_average_precision,
        "train_average_precision_rows": min(rows, TRAIN_AP_ROWS),
        "training_seconds": elapsed,
        "validation_inference_seconds": inference_seconds,
        "training_rows": len(train[1]),
        "validation_rows": len(validation[1]),
        "files": {name: sha256(output / name) for name in adapter.files},
        # Per-epoch losses for neural models; absent for estimators without epochs.
        "training_curve": getattr(model, "training_curve_", None),
    }
    write_json(output / "trial.json", metadata)
    restored = load_trial(output)
    np.testing.assert_allclose(
        adapter.scores(restored, validation[0].iloc[:20]),
        scores[:20],
        rtol=1e-10,
        atol=1e-12,
    )
    return metadata
