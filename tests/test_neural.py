"""The MLP trains with PyTorch, records learning curves, and scores without it."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraud_risk_platform.contracts.features import FeatureSet
from fraud_risk_platform.contracts.training import Recipe
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.training.trial import fit_trial, load_trial

SPEC = FeatureSet(numeric=["amount"], categorical=["use_chip"])


def recipe(**parameters):
    return Recipe(
        estimator="mlp",
        feature_set="details",
        feature_spec=SPEC,
        parameters={"hidden_units": 8, "epochs": 4, "batch_size": 8, **parameters},
    )


@pytest.fixture
def data():
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0] * 10,
            "use_chip": ["Chip", "Online", "Chip", "Online"] * 10,
        }
    )
    return frame, pd.Series([0, 1, 0, 1] * 10)


def test_trial_records_one_loss_per_epoch_for_both_partitions(tmp_path, data):
    import torch

    threads = torch.get_num_threads()
    result = fit_trial(
        recipe=recipe(),
        train=data,
        validation=data,
        review_budget=0.5,
        output=tmp_path / "trial",
    )
    curve = result["training_curve"]
    assert len(curve["train_loss"]) == len(curve["validation_loss"]) == 4
    assert set(result["files"]) == {"preprocessor.joblib", "model.onnx"}
    # The MLP's own thread limit does not leak into the rest of the process.
    assert torch.get_num_threads() == threads


def test_scoring_needs_onnx_runtime_but_not_pytorch(tmp_path, data):
    fit_trial(
        recipe=recipe(),
        train=data,
        validation=data,
        review_budget=0.5,
        output=tmp_path / "trial",
    )
    expected = adapter_for("mlp").scores(load_trial(tmp_path / "trial"), data[0])
    np.save(tmp_path / "expected.npy", expected)
    data[0].to_json(tmp_path / "frame.json")
    # The import hook makes PyTorch look uninstalled, as in the serving image.
    script = f"""
import importlib.abc, sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(name)

sys.meta_path.insert(0, NoTorch())
import numpy as np, pandas as pd
from pathlib import Path
from fraud_risk_platform.models.adapters import adapter_for
from fraud_risk_platform.training.trial import load_trial
directory = Path({str(tmp_path)!r})
scores = adapter_for("mlp").scores(load_trial(directory / "trial"), pd.read_json(directory / "frame.json"))
np.testing.assert_array_equal(scores, np.load(directory / "expected.npy"))
assert "torch" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_unknown_network_settings_fail_when_configuration_loads():
    with pytest.raises(ValidationError, match="hidden_size"):
        recipe(hidden_size=8)
    with pytest.raises(ValidationError, match="Unknown MLP search parameters"):
        Recipe.model_validate(
            {
                **recipe().model_dump(),
                "search_space": {"momentum": {"kind": "float", "low": 0, "high": 1}},
            }
        )
