"""IBM source setup, monthly snapshots, training, and explicit release operations."""

import argparse
import json
import logging
import os
import urllib.request
from pathlib import Path

from fraud_risk_platform.contracts.training import TrainingPolicy
from fraud_risk_platform.persistence.files import read_json, state_root, work_root


def served_release(api_url: str) -> str | None:
    """The release the API reports as ready, or None while it starts or is unready."""
    try:
        with urllib.request.urlopen(f"{api_url}/ready", timeout=5) as response:
            return json.load(response).get("release_id")
    except OSError, ValueError:
        return None


def parser():
    result = argparse.ArgumentParser(description="IBM fraud data and model lifecycle")
    result.add_argument("--state-dir", type=Path, default=state_root())
    result.add_argument("--policy", type=Path, default=Path("config/training.json"))
    result.add_argument("--data-config", type=Path, default=Path("config/data.json"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create application tables")
    seed = commands.add_parser(
        "seed-source", help="Download, verify, and seed the IBM database"
    )
    seed.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    seed.add_argument(
        "--source-lock", type=Path, default=Path("config/ibm-tabformer.lock.json")
    )
    seed.add_argument("--archive", type=Path)
    seed.add_argument(
        "--customer-limit", type=int, help="Optional loader smoke-test subset"
    )
    seed.add_argument("--chunk-size", type=int, default=200000)
    for name, help_text in [
        ("initial-snapshot", "Create or reuse the initial historical snapshot"),
        ("next-snapshot", "Create the next available month, or report no new data"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--fault",
            choices=["missing-amount", "invalid-amount", "duplicate", "conflict"],
        )
    verify = commands.add_parser("verify-snapshot")
    verify.add_argument("--snapshot", required=True)
    for name in ("train", "bootstrap"):
        command = commands.add_parser(name)
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument("--snapshot")
        selection.add_argument("--snapshot-as-of")
        command.add_argument(
            "--experiment",
            help="Name a deliberate repeat; an identical request otherwise reuses its job",
        )
        if name == "train":
            command.add_argument("--recipe", help="Train only this configured recipe")
            command.add_argument(
                "--trials", type=int, help="Override the bounded search budget"
            )
            command.add_argument(
                "--refit-snapshot",
                help="Explicit compatible uncapped snapshot ID or prepared directory",
            )
    training_jobs = commands.add_parser(
        "training-jobs", help="List recent training jobs and every attempt"
    )
    training_jobs.add_argument("--limit", type=int, default=20)
    prepare = commands.add_parser(
        "prepare-training-input",
        help="Export a verified snapshot as a local training channel",
    )
    prepare.add_argument("--snapshot", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    staging = commands.add_parser(
        "sagemaker-inputs",
        help="Write a SageMaker training job's input channels to S3",
    )
    staging.add_argument("--snapshot", required=True)
    staging.add_argument("--refit-snapshot")
    staging.add_argument("--recipe")
    staging.add_argument("--trials", type=int)
    importing = commands.add_parser(
        "import-release", help="Verify and add a training job's model.tar.gz"
    )
    importing.add_argument(
        "--archive", required=True, help="Local path or s3:// URI of model.tar.gz"
    )
    snapshots = commands.add_parser(
        "snapshots", help="List and inspect completed snapshots"
    )
    operations = snapshots.add_subparsers(dest="operation", required=True)
    listing = operations.add_parser("list")
    listing.add_argument("--as-of")
    show = operations.add_parser("show")
    show.add_argument("snapshot")
    report = operations.add_parser("report")
    report.add_argument("snapshot")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute a local report from immutable snapshot data",
    )
    promotion = commands.add_parser("promote")
    promotion.add_argument("--release", required=True)
    promotion.add_argument("--initial", action="store_true")
    promotion.add_argument("--reason", required=True)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--release")
    rollback.add_argument("--reason", required=True)
    commands.add_parser("status")
    verify_deployment = commands.add_parser(
        "verify-deployment",
        help="Confirm the API serves the selected release, else restore the last confirmed one",
    )
    verify_deployment.add_argument(
        "--api-url", default=os.environ.get("API_URL", "http://api:8000")
    )
    verify_deployment.add_argument("--timeout", type=float, default=120.0)
    commands.add_parser(
        "replay", help="Score the next month of traffic with the deployed release"
    )
    commands.add_parser(
        "monitor", help="Report replayed predictions against released outcomes"
    )
    load = commands.add_parser(
        "load", help="Send a sample of the coming month to the API and report latency"
    )
    load.add_argument("--requests", type=int, default=2000)
    load.add_argument("--rate", type=float, default=20.0, help="Requests per second")
    load.add_argument("--concurrency", type=int, default=4)
    load.add_argument("--api-url", default=os.environ.get("API_URL", "http://api:8000"))
    month = commands.add_parser(
        "month",
        help="Replay, advance, monitor, and train one simulated month; promotion stays explicit",
    )
    month.add_argument("--trials", type=int)
    month.add_argument("--skip-promotion", action="store_true")
    month.add_argument(
        "--stop-before-training",
        action="store_true",
        help="Return the snapshot to train instead of training it here",
    )
    final = commands.add_parser(
        "final-evaluation",
        help="Score the confirmed champion on the reserved holdout month, once",
    )
    final.add_argument(
        "--snapshot", help="Snapshot to read the holdout from; default: the latest"
    )
    commands.add_parser(
        "reset-simulation",
        help="Start a new simulation; keeps the source import and S3 objects",
    )
    outcomes = commands.add_parser("outcomes")
    outcomes.add_argument("--as-of", required=True)
    return result


def report_to_step_functions(token: str, *, value=None, error=None):
    """Return a command's result to the Step Functions task that started it.

    A workflow starts a container with `STEP_FUNCTIONS_TASK_TOKEN` when the next
    state needs the command's output, such as the snapshot a month produced.
    """
    import boto3

    client = boto3.client("stepfunctions")
    if error is None:
        client.send_task_success(
            taskToken=token, output=json.dumps(value, allow_nan=False)
        )
    else:
        client.send_task_failure(
            taskToken=token, error=type(error).__name__[:256], cause=str(error)[:32768]
        )


def run(args):
    if args.command == "init-db":
        from fraud_risk_platform.persistence.database import initialize

        initialize()
        value = {"status": "initialized"}
    elif args.command == "seed-source":
        from fraud_risk_platform.dataops.dataset import DataSettings
        from fraud_risk_platform.seeding.ibm import seed_ibm

        settings = DataSettings.model_validate(read_json(args.data_config))
        value = seed_ibm(
            source_lock=args.source_lock,
            raw_dir=args.raw_dir,
            label_delay_days=settings.label_delay_days,
            archive=args.archive,
            customer_limit=args.customer_limit,
            chunk_size=args.chunk_size,
        )
    elif args.command in {"initial-snapshot", "next-snapshot"}:
        from fraud_risk_platform.dataops.dataset import DataSettings
        from fraud_risk_platform.dataops.workflow import monthly_snapshot
        from fraud_risk_platform.persistence.objects import ObjectStore

        value = monthly_snapshot(
            initial=args.command == "initial-snapshot",
            settings=DataSettings.model_validate(read_json(args.data_config)),
            store=ObjectStore(),
            work_dir=work_root(args.state_dir),
            fault=args.fault,
        )
    elif args.command == "verify-snapshot":
        from fraud_risk_platform.dataops.reader import SnapshotReader
        from fraud_risk_platform.persistence.objects import ObjectStore

        with SnapshotReader(ObjectStore(), args.snapshot) as reader:
            value = reader.verify()
    elif args.command == "prepare-training-input":
        from fraud_risk_platform.training.inputs import prepare_snapshot

        value = prepare_snapshot(args.snapshot, args.output)
    elif args.command in {"train", "bootstrap"}:
        from fraud_risk_platform.training.jobs import run_training_job

        snapshot = args.snapshot
        if args.snapshot_as_of:
            from fraud_risk_platform.dataops.reader import resolve_snapshot
            from fraud_risk_platform.persistence.objects import ObjectStore

            snapshot = resolve_snapshot(ObjectStore(), as_of=args.snapshot_as_of)
        value = run_training_job(
            root=args.state_dir,
            snapshot=snapshot,
            policy=TrainingPolicy.model_validate(read_json(args.policy)),
            bootstrap=args.command == "bootstrap",
            recipe_name=getattr(args, "recipe", None),
            trials=getattr(args, "trials", None),
            refit_snapshot=getattr(args, "refit_snapshot", None),
            experiment=args.experiment,
        )
    elif args.command == "sagemaker-inputs":
        from fraud_risk_platform.persistence.objects import ObjectStore
        from fraud_risk_platform.training.sagemaker import stage_inputs

        value = stage_inputs(
            args.state_dir,
            ObjectStore(),
            snapshot_id=args.snapshot,
            policy=TrainingPolicy.model_validate(read_json(args.policy)),
            refit_snapshot_id=args.refit_snapshot,
            recipe_name=args.recipe,
            trials=args.trials,
        )
    elif args.command == "import-release":
        from fraud_risk_platform.training.sagemaker import import_release

        value = import_release(args.state_dir, args.archive)
    elif args.command == "training-jobs":
        from fraud_risk_platform.training.jobs import list_jobs

        value = list_jobs(limit=args.limit)
    elif args.command == "snapshots":
        from fraud_risk_platform.dataops.reader import SnapshotReader, list_snapshots
        from fraud_risk_platform.persistence.objects import ObjectStore

        store = ObjectStore()
        if args.operation == "list":
            value = sorted(
                list_snapshots(store, as_of=args.as_of),
                key=lambda item: (item["as_of"], item["snapshot_id"]),
            )
        else:
            with SnapshotReader(store, args.snapshot) as reader:
                value = reader.manifest
                if args.operation == "report":
                    if args.output.exists():
                        raise ValueError(
                            "Report output already exists; choose a new path"
                        )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    if args.refresh:
                        from fraud_risk_platform.dataops.reports.data import (
                            refresh_snapshot_report,
                            render_report,
                        )

                        refreshed = refresh_snapshot_report(reader)
                        args.output.write_text(render_report(refreshed))
                    else:
                        store.verify(reader.manifest["reports"]["html"], args.output)
                    value = {"report": str(args.output)}
    elif args.command in {"promote", "rollback", "status"}:
        from fraud_risk_platform.models.deployment import deployment, promote, rollback

        if args.command == "promote":
            value = promote(
                args.state_dir, args.release, initial=args.initial, reason=args.reason
            )
        elif args.command == "rollback":
            value = rollback(args.state_dir, reason=args.reason, target=args.release)
        else:
            value = deployment(args.state_dir)
    elif args.command == "verify-deployment":
        from fraud_risk_platform.models.deployment import confirm_rollout
        from fraud_risk_platform.persistence.files import safe_id

        value = confirm_rollout(
            args.state_dir,
            lambda: served_release(args.api_url),
            timeout=args.timeout,
        )
        if value["status"] == "deployed":
            registry = read_json(
                args.state_dir
                / "releases"
                / safe_id(value["release_id"])
                / "manifest.json"
            ).get("registry")
            try:
                from fraud_risk_platform.training.tracking import set_champion

                value["registry"] = (
                    set_champion(registry) if registry else "no registered version"
                )
            except Exception as error:
                # Serving is confirmed either way; rerunning this command retries the alias.
                logging.exception("Champion alias not updated")
                value["registry"] = f"not updated: {type(error).__name__}: {error}"
    elif args.command == "month":
        from fraud_risk_platform.dataops.dataset import DataSettings
        from fraud_risk_platform.persistence.objects import ObjectStore
        from fraud_risk_platform.simulation import run_month

        value = run_month(
            args.state_dir,
            DataSettings.model_validate(read_json(args.data_config)),
            TrainingPolicy.model_validate(read_json(args.policy)),
            store=ObjectStore(),
            trials=args.trials,
            skip_promotion=args.skip_promotion,
            train=not args.stop_before_training,
        )
    elif args.command == "load":
        from fraud_risk_platform.dataops.dataset import DataSettings
        from fraud_risk_platform.monitoring.load import run_load

        value = run_load(
            args.state_dir,
            DataSettings.model_validate(read_json(args.data_config)),
            requests=args.requests,
            rate=args.rate,
            concurrency=args.concurrency,
            api_url=args.api_url,
        )
    elif args.command == "final-evaluation":
        from fraud_risk_platform.dataops.dataset import DataSettings
        from fraud_risk_platform.monitoring.holdout import final_evaluation

        value = final_evaluation(
            args.state_dir,
            DataSettings.model_validate(read_json(args.data_config)),
            review_budget=TrainingPolicy.model_validate(
                read_json(args.policy)
            ).review_budget,
            snapshot=args.snapshot,
        )
    elif args.command == "reset-simulation":
        from fraud_risk_platform.simulation import reset_simulation

        value = reset_simulation(args.state_dir)
    elif args.command in {"replay", "monitor"}:
        from fraud_risk_platform.dataops.dataset import DataSettings

        settings = DataSettings.model_validate(read_json(args.data_config))
        if args.command == "replay":
            from fraud_risk_platform.monitoring.replay import replay

            value = replay(args.state_dir, settings)
        else:
            from fraud_risk_platform.monitoring.report import monitor

            value = monitor(
                args.state_dir,
                settings,
                review_budget=TrainingPolicy.model_validate(
                    read_json(args.policy)
                ).review_budget,
            )
    else:
        from fraud_risk_platform.contracts.base import midnight
        from fraud_risk_platform.monitoring.outcomes import query_outcomes

        value = query_outcomes(midnight(args.as_of))
    return value


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    args = parser().parse_args()
    token = os.environ.get("STEP_FUNCTIONS_TASK_TOKEN")
    try:
        value = run(args)
        if args.command == "verify-deployment" and value["status"] != "deployed":
            print(json.dumps(value, indent=2, allow_nan=False))
            raise SystemExit(1)
    except (Exception, SystemExit) as error:
        if token:
            report_to_step_functions(token, error=error)
        raise
    if token:
        report_to_step_functions(token, value=value)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
