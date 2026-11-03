"""Bounded sequential searches with one durable model artifact per completed trial."""

import logging
from time import perf_counter

import mlflow
import optuna

from fraud_risk_platform.evaluation.metrics import log_fields
from fraud_risk_platform.persistence.files import write_json
from fraud_risk_platform.training.tracking import log_trial, ranking_fields
from fraud_risk_platform.training.trial import fit_trial

logger = logging.getLogger(__name__)


def ranking(result):
    metric = result.get("validation") or {}
    return (
        metric.get("recall", -1),
        metric.get("precision", -1),
        -result.get("number", 0),
    )


def run_search(
    *, name, recipe, settings, train, validation, review_budget, output, snapshot_id
):
    with mlflow.start_run(run_name=f"search-{name}", nested=True) as run:
        output.mkdir(parents=True)
        # Progress lines below replace Optuna's, which show only the objective value.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        total = settings.trials if recipe.search_space else 1
        storage = optuna.storages.RDBStorage(
            f"sqlite:///{(output / 'study.db').resolve()}"
        )
        study = optuna.create_study(
            study_name=f"{snapshot_id}-{name}-{run.info.run_id}",
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=settings.seed),
            pruner=optuna.pruners.NopPruner(),
        )
        # Trial 0 evaluates the configured defaults, which sampling may never reach.
        defaults = {
            key: recipe.parameters[key]
            for key in recipe.search_space
            if key in recipe.parameters
        }
        if defaults:
            study.enqueue_trial(defaults)
        results = []
        mlflow.log_dict(recipe.model_dump(), "recipe.json")
        mlflow.log_params(settings.model_dump(exclude_none=True))

        def objective(trial):
            parameters = dict(recipe.parameters)
            for key, space in recipe.search_space.items():
                if space.kind == "categorical":
                    parameters[key] = trial.suggest_categorical(key, space.choices)
                elif space.kind == "int":
                    parameters[key] = trial.suggest_int(
                        key, int(space.low), int(space.high), log=space.log
                    )
                else:
                    parameters[key] = trial.suggest_float(
                        key, space.low, space.high, log=space.log
                    )
            configured = recipe.model_copy(
                update={"parameters": parameters, "search_space": {}}
            )
            directory = output / f"trial-{trial.number:04d}"
            logger.info(
                "Search %s trial-%04d (%d/%d) started: %s",
                name,
                trial.number,
                trial.number + 1,
                total,
                {key: parameters[key] for key in recipe.search_space}
                or "fixed parameters",
            )
            started = perf_counter()
            with mlflow.start_run(
                run_name=f"trial-{trial.number:04d}", nested=True
            ) as child:
                mlflow.set_tags(
                    {
                        "snapshot_id": snapshot_id,
                        "recipe_name": name,
                        "trial_number": trial.number,
                    }
                )
                trial.set_user_attr("run_id", child.info.run_id)
                try:
                    result = fit_trial(
                        recipe=configured,
                        train=train,
                        validation=validation,
                        review_budget=review_budget,
                        output=directory,
                    )
                    model_uri = log_trial(directory, result, validation[0].iloc[:5])
                except Exception as error:
                    mlflow.set_tag("failure", f"{type(error).__name__}: {error}")
                    raise
                result.update(
                    number=trial.number,
                    run_id=child.info.run_id,
                    model_uri=model_uri,
                    directory=str(directory),
                )
                results.append(result)
                threshold_fields = (
                    " threshold_recall={recall:.6f} threshold_precision={precision:.6f}"
                    " threshold_alert_rate={alert_rate:.6f}".format(
                        **result["validation"]
                    )
                    if result["validation"]
                    else ""
                )
                # key=value metrics can be parsed by SageMaker metric definitions.
                logger.info(
                    "Search %s trial-%04d (%d/%d) finished in %.1fs: qualified=%s "
                    "converged=%s train_average_precision=%.6f %s%s",
                    name,
                    trial.number,
                    trial.number + 1,
                    total,
                    perf_counter() - started,
                    str(result["qualified"]).lower(),
                    str(result["converged"]).lower(),
                    result["train_average_precision"],
                    log_fields(result["validation_ranking"], "validation_"),
                    threshold_fields,
                )
                trial.set_user_attr("qualified", result["qualified"])
                trial.set_user_attr("model_uri", model_uri)
                write_json(output / "trials.json", results)
                # The selection quantity: recall within the review budget. Unqualified
                # trials recall at most the budget, so 0 barely understates them.
                return result["validation"]["recall"] if result["qualified"] else 0.0

        try:
            study.optimize(
                objective,
                n_trials=total,
                timeout=settings.timeout_seconds,
            )
        finally:
            storage.engine.dispose()
            mlflow.log_artifact(str(output / "study.db"))
            if (output / "trials.json").exists():
                mlflow.log_artifact(str(output / "trials.json"))
        eligible = [result for result in results if result["qualified"]]
        best = max(eligible, key=ranking) if eligible else None
        # Summarize the search run even when no trial qualifies.
        strongest = max(results, key=lambda r: r["validation_average_precision"])
        mlflow.log_metrics(
            {
                **ranking_fields(strongest),
                "train_average_precision": strongest["train_average_precision"],
            }
        )
        mlflow.set_tag(
            "highest_average_precision_trial", f"trial-{strongest['number']:04d}"
        )
        logger.info(
            "Search %s finished: %d/%d trials qualified; selected: %s; "
            "highest average precision: trial-%04d (%.6f)",
            name,
            len(eligible),
            len(results),
            f"trial-{best['number']:04d}" if best else "none",
            strongest["number"],
            strongest["validation_average_precision"],
        )
        mlflow.set_tag("selected_trial", best["number"] if best else "none")
        return {"run_id": run.info.run_id, "trials": results, "best": best}
