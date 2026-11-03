"""SageMaker File-mode entrypoint; the same function is exercised in Docker locally."""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

from fraud_risk_platform.contracts.training import TrainingPolicy
from fraud_risk_platform.persistence.files import read_json, write_json
from fraud_risk_platform.training.cycle import run_cycle
from fraud_risk_platform.training.inputs import LocalSnapshot


def channel(directory, snapshot_id):
    """A snapshot channel: named by the job configuration, or a prepared directory."""
    return LocalSnapshot(directory, snapshot_id) if snapshot_id else directory


def main():
    # SageMaker forwards stdout/stderr to CloudWatch Logs; progress goes to stderr.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if sys.argv[1:] not in ([], ["train"]):
        raise ValueError("The training container accepts only the train operation")
    base = Path(os.environ.get("TRAINING_CHANNEL_ROOT", "/opt/ml/input/data"))
    output = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/data"))
    model_output = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output.mkdir(parents=True, exist_ok=True)
    try:
        job = read_json(base / "config" / "job.json")
        policy = TrainingPolicy.model_validate(job["policy"])
        baseline = None
        if (base / "incumbent" / "manifest.json").exists():
            baseline = read_json(base / "incumbent" / "manifest.json")["release_id"]
            from fraud_risk_platform.persistence.files import safe_id

            shutil.copytree(base / "incumbent", output / "releases" / safe_id(baseline))
        refit = job.get("refit_snapshot_id")
        report = run_cycle(
            root=output,
            snapshot=channel(base / "snapshot", job.get("snapshot_id")),
            policy=policy,
            recipe_name=job.get("recipe"),
            trials=job.get("trials"),
            refit_snapshot=channel(base / "refit", refit)
            if refit or job.get("refit", False)
            else None,
            incumbent_release_id=baseline,
        )
        model_output.mkdir(parents=True, exist_ok=True)
        # The cycle report travels with every outcome: promotion requires it, and
        # it records that the snapshot was trained even without a release.
        evidence = model_output / "cycle"
        evidence.mkdir()
        for path in (output / "cycles" / report["run_id"]).iterdir():
            if path.is_file():
                shutil.copy2(path, evidence / path.name)
        if report["release_id"]:
            shutil.copytree(
                output / "releases" / report["release_id"],
                model_output,
                dirs_exist_ok=True,
            )
        else:
            write_json(
                model_output / "no-qualified-release.json",
                {
                    "run_id": report["run_id"],
                    "recommendation": report["recommendation"],
                },
            )
        print(json.dumps(report, allow_nan=False))
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        (output.parent / "failure").write_text(f"{type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    main()
