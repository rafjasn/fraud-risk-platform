"""Build unfitted scikit-learn-compatible estimators from configured recipes.

Each estimator has one builder below, registered in BUILDERS. A builder chooses the
estimator's numeric and categorical preprocessing; build_model assembles them with
the feature set into one pipeline, after check_parameters rejects settings the
estimator would not use. The MLP is built and trained in models/neural.py.
"""

from typing import NamedTuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    SplineTransformer,
    StandardScaler,
)

from fraud_risk_platform.contracts.features import INDICATOR_FEATURES
from fraud_risk_platform.contracts.training import Recipe

# Settings each builder derives from the feature set, so recipes cannot set them.
DERIVED_PARAMETERS = {
    "hist_gradient_boosting": {"categorical_features"},
    "xgboost": {"enable_categorical", "feature_types"},
}


def one_hot_categories():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def ordinal_categories(**options):
    """Integer codes that trees split as categories; unseen and missing become NaN."""
    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
        **options,
    )


def imputed_numerics():
    """Median imputation, fitted on training only; trees need no scaling."""
    return [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]


def standardized_numerics():
    return [*imputed_numerics(), ("scaler", StandardScaler())]


def spline_numerics(spec):
    """Each continuous feature's own curve, for a linear model.

    Quantiles of up to 200,000 training rows map a feature onto [0, 1], which bounds
    heavy tails; a cubic B-spline basis over that range then lets the coefficients
    follow any smooth shape. 0/1 indicators need neither and pass through.
    """
    continuous = [name for name in spec.numeric if name not in INDICATOR_FEATURES]
    indicators = [name for name in spec.numeric if name in INDICATOR_FEATURES]
    curves = [
        *imputed_numerics(),
        (
            "quantile",
            QuantileTransformer(n_quantiles=1000, subsample=200_000, random_state=0),
        ),
        (
            "spline",
            SplineTransformer(n_knots=8, include_bias=False, sparse_output=True),
        ),
    ]
    return ColumnTransformer(
        [
            ("curves", Pipeline(curves), continuous),
            ("indicators", Pipeline(imputed_numerics()), indicators),
        ],
        sparse_threshold=1.0,
    )


def column_transformer(spec, numeric, categorical, *, sparse):
    """The feature set's numeric and categorical columns through their transformers."""
    transformations = []
    if spec.numeric:
        transformations.append(("numeric", numeric, spec.numeric))
    if spec.categorical:
        transformations.append(("categorical", categorical, spec.categorical))
    return ColumnTransformer(transformations, sparse_threshold=1.0 if sparse else 0.0)


def one_hot_preprocessor(spec):
    """Logistic regression's standard inputs, shared with the MLP: sparse, scaled, one-hot."""
    return column_transformer(
        spec, Pipeline(standardized_numerics()), one_hot_categories(), sparse=True
    )


class Parts(NamedTuple):
    """One estimator's preprocessing and model, before build_model assembles them."""

    numeric: object  # Transformer for the numeric columns
    categorical: object  # Transformer for the categorical columns
    estimator: object
    sparse: bool = False  # Keep the preprocessed matrix sparse (one-hot inputs)


def estimator_class(estimator):
    if estimator == "xgboost":
        # Imported on use: libxgboost needs an OpenMP runtime (libomp on macOS).
        from xgboost import XGBClassifier

        return XGBClassifier
    return {
        "logistic": LogisticRegression,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
    }[estimator]


def build_logistic(recipe, spec):
    return Parts(
        spline_numerics(spec)
        if recipe.numeric_encoding == "quantile_spline"
        else Pipeline(standardized_numerics()),
        one_hot_categories(),
        LogisticRegression(**recipe.parameters),
        sparse=True,
    )


def build_hist_gradient_boosting(recipe, spec):
    return Parts(
        Pipeline(imputed_numerics()),
        ordinal_categories(max_categories=recipe.parameters.get("max_bins", 255)),
        HistGradientBoostingClassifier(
            **recipe.parameters,
            categorical_features=[False] * len(spec.numeric)
            + [True] * len(spec.categorical),
        ),
    )


def build_xgboost(recipe, spec):
    return Parts(
        Pipeline(imputed_numerics()),
        ordinal_categories(),
        # Ordinal codes are declared categorical, so splits partition categories.
        estimator_class("xgboost")(
            **recipe.parameters,
            enable_categorical=True,
            feature_types=["q"] * len(spec.numeric) + ["c"] * len(spec.categorical),
        ),
    )


BUILDERS = {
    "logistic": build_logistic,
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "xgboost": build_xgboost,
}


def check_parameters(recipe: Recipe):
    """Reject settings an estimator lacks, or that its feature set determines.

    Run before any fitting: scikit-learn rejects an unknown name only when a trial
    builds the estimator, and XGBoost passes it to its native library, which ignores
    it. MLPParameters checks the MLP's settings when the configuration loads.
    """
    if recipe.estimator not in BUILDERS:
        return
    names = recipe.parameters.keys() | recipe.search_space.keys()
    derived = names & DERIVED_PARAMETERS.get(recipe.estimator, set())
    if derived:
        raise ValueError(f"The feature set determines {sorted(derived)}")
    unknown = names - estimator_class(recipe.estimator)().get_params().keys()
    if unknown:
        raise ValueError(f"Unknown {recipe.estimator} parameters: {sorted(unknown)}")


def build_model(recipe: Recipe):
    check_parameters(recipe)
    spec = recipe.features
    if spec.version == "basic-v1":
        return build_basic_model(recipe)
    if recipe.estimator not in BUILDERS:
        raise ValueError(f"Unsupported estimator: {recipe.estimator}")
    parts = BUILDERS[recipe.estimator](recipe, spec)
    model = Pipeline(
        [
            (
                "preprocessor",
                column_transformer(
                    spec, parts.numeric, parts.categorical, sparse=parts.sparse
                ),
            ),
            ("model", parts.estimator),
        ]
    )
    model.fraud_feature_spec_ = spec.model_dump()
    return model


def build_basic_model(recipe):
    """The original basic-v1 models, which predate configurable feature sets."""
    if recipe.numeric_encoding != "standard":
        raise ValueError("quantile_spline encoding requires a configured feature set")
    if recipe.estimator == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**recipe.parameters)),
            ]
        )
    if recipe.estimator == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**recipe.parameters)
    if recipe.estimator == "xgboost":
        raise ValueError("XGBoost recipes require a configured feature set")
    raise ValueError(f"Unsupported estimator: {recipe.estimator}")
