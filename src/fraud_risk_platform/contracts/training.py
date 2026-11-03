"""Training policy: model recipes, bounded searches, and acceptance settings."""

from typing import Literal

from pydantic import Field, model_validator

from fraud_risk_platform.contracts.base import Contract
from fraud_risk_platform.contracts.features import FeatureSet, basic_feature_set


class SearchParameter(Contract):
    kind: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[str | int | float | bool | None] | None = None

    @model_validator(mode="after")
    def valid_distribution(self):
        if self.kind == "categorical":
            if (
                not self.choices
                or self.low is not None
                or self.high is not None
                or self.log
            ):
                raise ValueError("Categorical search requires nonempty choices only")
        elif (
            self.low is None
            or self.high is None
            or self.low > self.high
            or self.choices is not None
            or (self.log and self.low <= 0)
            or (
                self.kind == "int"
                and (not self.low.is_integer() or not self.high.is_integer())
            )
        ):
            raise ValueError("Invalid numeric search bounds")
        return self

    def contains(self, value) -> bool:
        if self.kind == "categorical":
            return value in self.choices
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and self.low <= value <= self.high
            and (self.kind == "float" or float(value).is_integer())
        )


class SearchSettings(Contract):
    trials: int = Field(default=1, ge=1, le=1000)
    seed: int = 42
    timeout_seconds: float | None = Field(default=None, gt=0)


class MLPParameters(Contract):
    """Feed-forward network settings; the loss weights frauds by pos_weight."""

    hidden_units: int = Field(default=64, ge=1, le=1024)
    hidden_layers: int = Field(default=2, ge=1, le=4)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    learning_rate: float = Field(default=1e-3, gt=0, le=1)
    weight_decay: float = Field(default=1e-4, ge=0)
    epochs: int = Field(default=10, ge=1, le=200)
    batch_size: int = Field(default=1024, ge=1)
    pos_weight: float = Field(default=1.0, gt=0)
    random_state: int = 42


class Recipe(Contract):
    estimator: Literal["logistic", "hist_gradient_boosting", "xgboost", "mlp"]
    parameters: dict
    search_space: dict[str, SearchParameter] = Field(default_factory=dict)
    feature_set: str = "basic-v1"
    feature_spec: FeatureSet | None = None
    # How logistic regression sees numeric features: standardized values, or each
    # feature's own curve, a cubic spline basis over its training quantiles.
    numeric_encoding: Literal["standard", "quantile_spline"] = "standard"

    @property
    def features(self):
        if self.feature_spec is not None:
            return self.feature_spec
        if self.feature_set == "basic-v1":
            return basic_feature_set()
        raise ValueError("Resolve the recipe's feature set before training")

    @model_validator(mode="after")
    def known_network_parameters(self):
        # Fail when the configuration loads, not midway through a cycle.
        if self.estimator == "mlp":
            MLPParameters.model_validate(self.parameters)
            unknown = self.search_space.keys() - MLPParameters.model_fields.keys()
            if unknown:
                raise ValueError(f"Unknown MLP search parameters: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def supported_numeric_encoding(self):
        if self.numeric_encoding != "standard" and (
            self.estimator != "logistic" or self.feature_set == "basic-v1"
        ):
            raise ValueError(
                "quantile_spline encoding is for logistic regression on a configured feature set"
            )
        return self

    @model_validator(mode="after")
    def searched_defaults_in_range(self):
        # A search's first trial evaluates these defaults (training/search.py).
        for key, space in self.search_space.items():
            if key in self.parameters and not space.contains(self.parameters[key]):
                raise ValueError(
                    f"Default {key}={self.parameters[key]!r} is outside its search space"
                )
        return self


class TrainingPolicy(Contract):
    schema_version: Literal[2] = 2
    # Share of scored transactions analysts can review; thresholds fill it on validation.
    review_budget: float = Field(gt=0, lt=1)
    minimum_recall_gain: float = Field(ge=0, le=1)
    minimum_rows_per_split: int = Field(gt=0)
    minimum_frauds_per_split: int = Field(gt=0)
    initial_recipe: str
    challengers: list[str] = Field(default_factory=list)
    recipes: dict[str, Recipe]
    search: SearchSettings = Field(default_factory=SearchSettings)
    feature_sets: dict[str, FeatureSet] = Field(default_factory=dict)

    def resolve_recipe(self, name):
        recipe = self.recipes[name]
        spec = (
            basic_feature_set()
            if recipe.feature_set == "basic-v1"
            else self.feature_sets.get(recipe.feature_set)
        )
        if spec is None:
            raise ValueError(f"Unknown feature set: {recipe.feature_set}")
        if recipe.feature_spec is not None and recipe.feature_spec != spec:
            raise ValueError("Recipe feature_spec differs from its named feature set")
        return recipe.model_copy(update={"feature_spec": spec.model_copy(deep=True)})

    @model_validator(mode="after")
    def known_recipes(self):
        if not {self.initial_recipe, *self.challengers} <= self.recipes.keys():
            raise ValueError("Every configured recipe must be defined")
        if (
            "basic-v1" in self.feature_sets
            and self.feature_sets["basic-v1"] != basic_feature_set()
        ):
            raise ValueError("Cannot redefine basic-v1")
        for name in self.recipes:
            self.resolve_recipe(name)
        return self
