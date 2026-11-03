"""Coordinate searches and a later acceptance gate; never change deployment.

A cycle runs in phases, each a function below, in this order: check the inputs and
the incumbent, search every recipe on train/validation, select one validation winner,
optionally refit it on an uncapped snapshot, report validation quality, and only
then score the frozen candidate and incumbent on the later evaluation partition.
The recommendation gates promotion; the selected model is registered and saved as
a verified release, but never deployed.
"""

import logging
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraud_risk_platform.contracts.features import FeatureSet
from fraud_risk_platform.contracts.training import (
    Recipe,
    SearchSettings,
    TrainingPolicy,
)
from fraud_risk_platform.dataops.reader import SnapshotReader
from fraud_risk_platform.evaluation.charts import (
    population,
    write_comparison,
    write_evaluation,
    write_learning_curves,
    write_trials,
    write_validation,
)
from fraud_risk_platform.evaluation.metrics import (
    alert_name,
    bootstrap_recall,
    interval,
    log_fields,
    metrics,
    ranking_metrics,
    recommendation,
)
from fraud_risk_platform.features.matrices import feature_matrices
from fraud_risk_platform.features.transaction import artifact_feature_spec
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.models.recipes import check_parameters
from fraud_risk_platform.models.releases import (
    load_release,
    runtime_versions,
    save_release,
)
from fraud_risk_platform.persistence.files import identity, safe_id, sha256, write_json
from fraud_risk_platform.training.inputs import check_refit_snapshot, verified_data
from fraud_risk_platform.training.report import write_report
from fraud_risk_platform.training.search import ranking, run_search
from fraud_risk_platform.training.tracking import (
    configure_tracking,
    log_provenance,
    log_trial,
    ranking_fields,
    register_candidate,
)
from fraud_risk_platform.training.trial import fit_trial, load_trial

logger = logging.getLogger(__name__)


@dataclass
class _Cycle:
    """What every phase of one cycle shares; `report` accumulates their evidence."""

    root: Path
    output: Path
    run_id: str
    policy: TrainingPolicy
    settings: SearchSettings
    reader: SnapshotReader
    report: dict
    baseline: str | None

    @property
    def manifest(self) -> dict:
        return self.reader.manifest

    @property
    def as_of(self) -> str:
        return self.manifest["as_of"][:10]

    @property
    def budget_key(self) -> str:
        return f"recall_at_{alert_name(self.policy.review_budget)}"


@dataclass
class _Validation:
    """Validation features in one feature set."""

    spec: FeatureSet
    X: pd.DataFrame
    y: pd.Series


def run_cycle(
    *,
    root: Path,
    snapshot: Path | str,
    policy: TrainingPolicy,
    tracking_uri: str | None = None,
    bootstrap: bool = False,
    recipe_name: str | None = None,
    trials: int | None = None,
    refit_snapshot: Path | str | None = None,
    incumbent_release_id: str | None = None,
    on_start: Callable[[str], None] | None = None,
) -> dict:
    """Run one cycle; `on_start` receives the MLflow run ID while that run is active."""
    configure_tracking(tracking_uri)
    baseline = incumbent_release_id or deployment(root)["current"]
    if bootstrap and baseline:
        raise ValueError("Bootstrap is only available before the first deployment")
    if bootstrap and (refit_snapshot or recipe_name or trials):
        raise ValueError(
            "Bootstrap is a fixed baseline; use train for recipe search/refit"
        )
    settings, recipes = _plan(
        policy, bootstrap=bootstrap, recipe_name=recipe_name, trials=trials
    )
    with mlflow.start_run(run_name="training-cycle") as run:
        if on_start:
            on_start(run.info.run_id)
        output = root / "cycles" / run.info.run_id
        output.mkdir(parents=True)
        logger.info(
            "Training cycle %s started: snapshot=%s recipes=%s trials=%d",
            run.info.run_id,
            snapshot,
            ",".join(recipes),
            settings.trials,
        )
        report = _new_report(run.info.run_id, snapshot, baseline, bootstrap=bootstrap)
        try:
            with ExitStack() as stack:
                cycle = _Cycle(
                    root=root,
                    output=output,
                    run_id=run.info.run_id,
                    policy=policy,
                    settings=settings,
                    reader=stack.enter_context(verified_data(snapshot, policy)),
                    report=report,
                    baseline=baseline,
                )
                refit_reader = _open_refit(cycle, stack, refit_snapshot)
                _record_snapshot(cycle)
                incumbent_manifest, incumbent_model = _load_incumbent(cycle)
                if incumbent_manifest and recipe_name is None:
                    _add_incumbent_recipe(recipes, incumbent_manifest)
                resolved = _log_configuration(cycle, recipes)

                validation, labels, scores = _search_recipes(cycle, recipes)
                winner, selected = _select_winner(cycle, labels, scores)
                if selected:
                    validation = _validation_for(cycle, validation, _spec(selected))
                    if refit_reader:
                        selected = _refit(
                            cycle, refit_reader, winner, selected, validation
                        )
                candidate_model = (
                    load_trial(Path(selected["directory"])) if selected else None
                )
                label = _report_validation(
                    cycle, validation, labels, scores, winner, selected, candidate_model
                )

                candidate_eval, incumbent_eval = _evaluate(
                    cycle,
                    incumbent_manifest,
                    incumbent_model,
                    selected,
                    candidate_model,
                    label,
                )
                allowed = _recommend(cycle, candidate_eval, incumbent_eval)
                if selected:
                    _publish_candidate(
                        cycle,
                        winner,
                        selected,
                        candidate_model,
                        validation,
                        resolved,
                        allowed,
                    )
                _finish(cycle)
                return report
        except Exception as error:
            report.update(
                recommendation="failed", reason=f"{type(error).__name__}: {error}"
            )
            write_report(output, report)
            try:
                mlflow.log_artifact(str(output / "report.json"), "reports")
                mlflow.set_tag("recommendation", "failed")
            except Exception:
                logger.exception("Failed to log failure report to MLflow")
            raise


def _plan(policy, *, bootstrap, recipe_name, trials):
    """The search settings and resolved recipes a cycle runs."""
    settings = SearchSettings.model_validate(
        {
            **policy.search.model_dump(),
            **({"trials": trials} if trials is not None else {}),
        }
    )
    names = (
        [recipe_name]
        if recipe_name
        else [policy.initial_recipe, *([] if bootstrap else policy.challengers)]
    )
    if any(name not in policy.recipes for name in names):
        raise ValueError("Unknown recipe")
    recipes = {
        safe_id(name): policy.resolve_recipe(name) for name in dict.fromkeys(names)
    }
    # A misnamed setting fails here, before any search spends time on the others.
    for recipe in recipes.values():
        check_parameters(recipe)
    if bootstrap:
        recipes = {
            name: recipe.model_copy(update={"search_space": {}})
            for name, recipe in recipes.items()
        }
        settings = settings.model_copy(update={"trials": 1})
    return settings, recipes


def _new_report(run_id, snapshot, baseline, *, bootstrap):
    """The report a cycle writes even if it fails before any phase completes."""
    return {
        "schema_version": 3,
        "run_id": run_id,
        "as_of": "unknown",
        "snapshot_id": str(snapshot),
        "incumbent_release_id": baseline,
        "recommendation": "failed",
        "reason": "Cycle did not complete",
        "release_id": None,
        "purpose": "initial_logistic_baseline" if bootstrap else "comparative_cycle",
        "candidates": {},
        "searches": {},
        "comparison": {},
    }


def _open_refit(cycle, stack, refit_snapshot):
    """The verified uncapped snapshot for an optional refit, checked compatible."""
    if not refit_snapshot:
        return None
    refit_reader = stack.enter_context(verified_data(refit_snapshot, cycle.policy))
    check_refit_snapshot(cycle.reader, refit_reader)
    cycle.report["refit_snapshot_id"] = refit_reader.manifest["snapshot_id"]
    mlflow.log_dict(refit_reader.manifest, "snapshot/refit-manifest.json")
    return refit_reader


def _record_snapshot(cycle):
    manifest = cycle.manifest
    cycle.report.update(
        snapshot_id=manifest["snapshot_id"],
        as_of=cycle.as_of,
        windows=manifest["windows"],
        splits=manifest["splits"],
        dataset=manifest["dataset"],
        deployment_supported=True,
    )


def _load_incumbent(cycle):
    """The deployed release to compare against, checked to be comparable."""
    if not cycle.baseline:
        return None, None
    incumbent, model = load_release(cycle.root, cycle.baseline)
    manifest = cycle.manifest
    if (
        incumbent["dataset"] != manifest["dataset"]
        or incumbent["source_import_id"] != manifest["source"]["import_id"]
    ):
        raise ValueError("Incumbent and snapshot use different source datasets/imports")
    if manifest["windows"]["evaluation"][0] < incumbent["information_end_exclusive"]:
        raise ValueError(
            "Evaluation overlaps information used by the incumbent release"
        )
    if cycle.as_of <= incumbent["as_of"]:
        raise ValueError("A subsequent cycle must advance simulated time")
    return incumbent, model


def _add_incumbent_recipe(recipes, incumbent_manifest):
    """Retrain the incumbent's exact configuration unless a recipe already matches."""
    saved = Recipe.model_validate(incumbent_manifest["recipe"])
    if not any(
        r.estimator == saved.estimator
        and r.parameters == saved.parameters
        and r.numeric_encoding == saved.numeric_encoding
        and r.features == saved.features
        for r in recipes.values()
    ):
        recipes["incumbent_retrained"] = saved.model_copy(update={"search_space": {}})


def _log_configuration(cycle, recipes):
    """Record the resolved configuration, runtime versions, and provenance."""
    policy, report = cycle.policy, cycle.report
    resolved = {
        **policy.model_dump(),
        "search": cycle.settings.model_dump(),
        "executed_recipes": {
            name: recipe.model_dump() for name, recipe in recipes.items()
        },
        "refit_snapshot_id": report.get("refit_snapshot_id"),
    }
    report["resolved_config"] = resolved
    report["runtime_versions"] = runtime_versions()
    report["provenance"] = log_provenance()
    mlflow.log_params(
        {
            "snapshot_id": cycle.manifest["snapshot_id"],
            "as_of": cycle.as_of,
            "policy_sha256": identity(resolved),
            "incumbent_release_id": cycle.baseline or "none",
            "review_budget": policy.review_budget,
        }
    )
    mlflow.log_dict(resolved, "config/resolved.json")
    mlflow.log_dict(cycle.manifest, "snapshot/manifest.json")
    return resolved


def _search_recipes(cycle, recipes):
    """Search each recipe on train/validation.

    Returns the last feature set's validation features, the validation labels, and
    each recipe's representative model scored on validation, for comparison.
    """
    report = cycle.report
    spec = X = y = None
    labels, scores = None, {}
    for name, recipe in recipes.items():
        if spec != recipe.features:
            # Release the previous feature set's matrices before building the next.
            X = y = None
            X, y = feature_matrices(
                cycle.reader, ("train", "validation"), feature_spec=recipe.features
            )
            spec = recipe.features
        search = run_search(
            name=name,
            recipe=recipe,
            settings=cycle.settings,
            train=(X["train"], y["train"]),
            validation=(X["validation"], y["validation"]),
            review_budget=cycle.policy.review_budget,
            output=cycle.output / name,
            snapshot_id=cycle.manifest["snapshot_id"],
        )
        report["searches"][name] = search
        representative = search["best"] or max(
            search["trials"],
            key=lambda trial: trial["validation_ranking"][cycle.budget_key],
        )
        report["comparison"][name] = {
            "trial": representative["number"],
            "run_id": representative["run_id"],
            **{
                key: representative[key]
                for key in (
                    "qualified",
                    "threshold",
                    "validation",
                    "validation_ranking",
                    "train_average_precision",
                )
            },
            "training_curve": representative.get("training_curve"),
        }
        scores[name] = adapter_for(recipe.estimator).scores(
            load_trial(Path(representative["directory"])), X["validation"]
        )
        # Labels must align across feature sets for the paired comparison.
        if labels is None:
            labels = y["validation"].to_numpy()
        elif not np.array_equal(labels, y["validation"].to_numpy()):
            raise ValueError("Validation rows differ between feature sets")
        report["candidates"][name] = search["best"] or {
            "recipe": recipe.model_dump(),
            "threshold": None,
            "validation": None,
        }
        write_report(cycle.output, report)
    return _Validation(spec, X["validation"], y["validation"]), labels, scores


def _select_winner(cycle, labels, scores):
    """The validation winner and its selected trial, or (None, None)."""
    report, key = cycle.report, cycle.budget_key
    eligible = {
        name: search["best"]
        for name, search in report["searches"].items()
        if search["best"]
    }
    winner = (
        max(eligible, key=lambda name: ranking(eligible[name])) if eligible else None
    )
    report["validation_winner"] = winner
    logger.info("Validation winner: %s", winner or "none qualified")
    # Paired bootstrap: each recipe's recall at the budget, and the winner's
    # lead over it, on the same validation resamples.
    draws = bootstrap_recall(
        labels,
        scores,
        review_budget=cycle.policy.review_budget,
        seed=cycle.settings.seed,
    )
    for name, entry in report["comparison"].items():
        entry["recall_interval"] = interval(draws[name])
        if winner and name != winner:
            entry["winner_lead"] = {
                "estimate": report["comparison"][winner]["validation_ranking"][key]
                - entry["validation_ranking"][key],
                "interval": interval(draws[winner] - draws[name]),
            }
    return winner, eligible[winner] if winner else None


def _spec(result):
    return Recipe.model_validate(result["recipe"]).features


def _validation_for(cycle, validation, spec):
    """Validation features in `spec`, reusing the current ones when they match."""
    if spec == validation.spec:
        return validation
    X, y = feature_matrices(cycle.reader, ("validation",), feature_spec=spec)
    return _Validation(spec, X["validation"], y["validation"])


def _refit(cycle, refit_reader, winner, selected, validation):
    """Refit the winner's configuration on all eligible training history."""
    logger.info(
        "Refitting %s on uncapped snapshot %s",
        winner,
        refit_reader.manifest["snapshot_id"],
    )
    with mlflow.start_run(run_name="full-history-refit", nested=True) as refit_run:
        full_X, full_y = feature_matrices(
            refit_reader, ("train",), feature_spec=_spec(selected)
        )
        directory = cycle.output / "refit"
        refitted = fit_trial(
            recipe=Recipe.model_validate(selected["recipe"]),
            train=(full_X["train"], full_y["train"]),
            validation=(validation.X, validation.y),
            review_budget=cycle.policy.review_budget,
            output=directory,
        )
        mlflow.set_tags(
            {
                "selected_trial_run_id": selected["run_id"],
                "snapshot_id": refit_reader.manifest["snapshot_id"],
            }
        )
        model_uri = log_trial(directory, refitted, validation.X.iloc[:5])
        refitted.update(
            run_id=refit_run.info.run_id,
            model_uri=model_uri,
            directory=str(directory),
            selected_trial_run_id=selected["run_id"],
        )
    cycle.report["refit"] = refitted
    return refitted


def _report_validation(
    cycle, validation, labels, scores, winner, selected, candidate_model
):
    """Validation quality and charts for every cycle, including unqualified ones.

    Summarizes the selected candidate, or else the highest-AP trial. Returns that
    model's label for later charts.
    """
    report, policy = cycle.report, cycle.policy
    if selected:
        name, summary = winner, selected
    else:
        name, summary = max(
            (
                (recipe, trial)
                for recipe, search in report["searches"].items()
                for trial in search["trials"]
            ),
            key=lambda item: item[1]["validation_average_precision"],
        )
    label = name + (
        f" trial-{summary['number']:04d}" if "number" in summary else " refit"
    )
    report["validation_summary"] = {
        "source": "selected_candidate"
        if selected
        else "highest_average_precision_unqualified",
        "model": label,
        "run_id": summary["run_id"],
        "train_average_precision": summary["train_average_precision"],
        **summary["validation_ranking"],
    }
    mlflow.log_metrics(
        {
            **ranking_fields(summary),
            "train_average_precision": summary["train_average_precision"],
        }
    )
    mlflow.set_tags(
        {
            "validation_summary_source": report["validation_summary"]["source"],
            "validation_summary_run_id": summary["run_id"],
        }
    )
    logger.info(
        "Validation summary (%s): %s",
        report["validation_summary"]["source"],
        log_fields(summary["validation_ranking"], "validation_"),
    )
    validation = _validation_for(cycle, validation, _spec(summary))
    model = candidate_model if selected else load_trial(Path(summary["directory"]))
    validation_scores = adapter_for(summary["recipe"]["estimator"]).scores(
        model, validation.X
    )
    window = cycle.manifest["windows"]["validation"]
    report["plots"] = [
        write_validation(
            cycle.output,
            validation.y,
            validation_scores,
            threshold=summary["threshold"],
            review_budget=policy.review_budget,
            title=f"{population('Validation', window, validation.y)} · {label}"
            + ("" if selected else " (not qualified; highest validation AP)"),
        ),
        write_trials(
            cycle.output,
            report["searches"],
            review_budget=policy.review_budget,
            title=population("Search trials", window, validation.y),
        ),
    ]
    curves = [
        (f"{name} trial-{entry['trial']:04d}", entry["training_curve"])
        for name, entry in report["comparison"].items()
        if entry.get("training_curve")
    ]
    if curves:
        report["plots"].insert(
            1,
            write_learning_curves(
                cycle.output,
                curves,
                title=population(
                    "Learning curves (validation loss is monitored only)",
                    window,
                    labels,
                ),
            ),
        )
    if len(report["comparison"]) > 1:
        models = [
            (
                f"{name} trial-{entry['trial']:04d}"
                + ("" if entry["qualified"] else " (not qualified)"),
                scores[name],
                entry["threshold"],
            )
            for name, entry in report["comparison"].items()
        ]
        report["plots"].insert(
            1,
            write_comparison(
                cycle.output,
                labels,
                models,
                review_budget=policy.review_budget,
                title=population("Recipe comparison", window, labels),
            ),
        )
    return label


def _evaluation_metrics(labels, scores, threshold, review_budget):
    return {
        **metrics(labels, scores, threshold),
        **ranking_metrics(labels, scores, review_budget=review_budget),
    }


def _evaluate(
    cycle, incumbent_manifest, incumbent_model, selected, candidate_model, label
):
    """Score the frozen candidate and the incumbent on the evaluation partition.

    Evaluation matrices enter the process only here, after selection/refit is frozen.
    Each model is scored with its own saved feature definition.
    """
    report, policy = cycle.report, cycle.policy
    candidate_eval = incumbent_eval = None
    spec = None
    if incumbent_model is not None:
        spec = artifact_feature_spec(incumbent_manifest)
        X, y = feature_matrices(cycle.reader, ("evaluation",), feature_spec=spec)
        incumbent_scores = adapter_for(
            incumbent_manifest["recipe"]["estimator"]
        ).scores(incumbent_model, X["evaluation"])
        incumbent_eval = _evaluation_metrics(
            y["evaluation"],
            incumbent_scores,
            incumbent_manifest["threshold"],
            policy.review_budget,
        )
    if selected and selected["qualified"]:
        if spec != _spec(selected):
            X, y = feature_matrices(
                cycle.reader, ("evaluation",), feature_spec=_spec(selected)
            )
        candidate_scores = adapter_for(selected["recipe"]["estimator"]).scores(
            candidate_model, X["evaluation"]
        )
        candidate_eval = _evaluation_metrics(
            y["evaluation"],
            candidate_scores,
            selected["threshold"],
            policy.review_budget,
        )
    if candidate_eval or incumbent_eval:
        report["plots"].append(
            write_evaluation(
                cycle.output,
                y["evaluation"],
                candidate=(
                    f"Candidate: {label}",
                    candidate_scores,
                    selected["threshold"],
                )
                if candidate_eval
                else None,
                incumbent=(
                    "Incumbent: "
                    + incumbent_manifest.get("recipe_name", cycle.baseline[:12]),
                    incumbent_scores,
                    incumbent_manifest["threshold"],
                )
                if incumbent_eval
                else None,
                review_budget=policy.review_budget,
                title=population(
                    "Evaluation",
                    cycle.manifest["windows"]["evaluation"],
                    y["evaluation"],
                ),
            )
        )
    report["evaluation"] = {"candidate": candidate_eval, "incumbent": incumbent_eval}
    for name, values in report["evaluation"].items():
        if values:
            logger.info(
                "Evaluation %s: %s", name, log_fields(values, f"evaluation_{name}_")
            )
            mlflow.log_metrics({f"evaluation_{name}_{k}": v for k, v in values.items()})
    return candidate_eval, incumbent_eval


def _recommend(cycle, candidate_eval, incumbent_eval):
    """Apply the evaluation gate; returns whether promotion is allowed."""
    allowed, reason = recommendation(
        candidate_eval,
        incumbent_eval,
        review_budget=cycle.policy.review_budget,
        minimum_recall_gain=cycle.policy.minimum_recall_gain,
    )
    cycle.report.update(
        recommendation=("promote" if cycle.baseline else "initial_promotion")
        if allowed
        else ("keep_current" if cycle.baseline else "no_initial_model"),
        reason=reason,
    )
    return allowed


def _publish_candidate(
    cycle, winner, selected, candidate_model, validation, resolved, allowed
):
    """Register the selected model and save it as a verified release; never deploys."""
    report, manifest = cycle.report, cycle.manifest
    report["selected_model_uri"] = selected["model_uri"]
    registry = register_candidate(
        selected["model_uri"], cycle_run_id=cycle.run_id, qualified=allowed
    )
    report["registry"] = registry
    if selected["threshold"] is None:
        return
    release = save_release(
        cycle.root,
        candidate_model,
        {
            "source_run_id": cycle.run_id,
            "trial_run_id": selected["run_id"],
            "snapshot_id": report.get("refit_snapshot_id", manifest["snapshot_id"]),
            "search_snapshot_id": manifest["snapshot_id"],
            "dataset": manifest["dataset"],
            "source_import_id": manifest["source"]["import_id"],
            "as_of": cycle.as_of,
            "training_end_exclusive": manifest["windows"]["train"][1],
            "information_end_exclusive": manifest["windows"]["evaluation"][1],
            "recipe_name": winner,
            "recipe": selected["recipe"],
            "threshold": selected["threshold"],
            "review_budget": cycle.policy.review_budget,
            "incumbent_release_id": cycle.baseline,
            "promotion_eligible": allowed,
            "resolved_config_sha256": identity(resolved),
            "evaluation": report["evaluation"],
            "recommendation": report["recommendation"],
            "registry": registry,
            "model_uri": selected["model_uri"],
        },
        validation.X.iloc[:20],
    )
    report["release_id"] = release["release_id"]
    mlflow.log_dict(release, "release/manifest.json")
    mlflow.set_tag("artifact_roundtrip", "passed")
    mlflow.MlflowClient().set_model_version_tag(
        registry["name"], registry["version"], "release_id", release["release_id"]
    )


def _finish(cycle):
    """Write and log the final report, then mark the cycle complete."""
    report, output = cycle.report, cycle.output
    write_report(output, report)
    mlflow.log_artifact(str(output / "report.json"), "reports")
    mlflow.log_artifact(str(output / "report.html"), "reports")
    for filename in report["plots"]:
        mlflow.log_artifact(str(output / filename), "reports")
    mlflow.set_tags(
        {
            "recommendation": report["recommendation"],
            "validation_winner": report["validation_winner"] or "none",
        }
    )
    logger.info(
        "Training cycle %s finished: recommendation=%s release=%s (%s)",
        cycle.run_id,
        report["recommendation"],
        report["release_id"] or "none",
        report["reason"],
    )
    write_json(
        output / "complete.json",
        {"run_id": cycle.run_id, "report_sha256": sha256(output / "report.json")},
    )
