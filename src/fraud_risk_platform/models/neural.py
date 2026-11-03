"""Feed-forward network trained with PyTorch and scored with ONNX Runtime.

Training imports PyTorch; scoring needs only ONNX Runtime, so serving images never
install PyTorch. Validation metrics are computed through the exported ONNX graph,
which is the artifact releases serve.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from scipy import sparse
from scipy.special import expit

from fraud_risk_platform.contracts.features import FeatureSet
from fraud_risk_platform.contracts.training import MLPParameters
from fraud_risk_platform.models.adapters import (
    checked_probabilities,
    normalize_features,
)
from fraud_risk_platform.models.recipes import one_hot_preprocessor

logger = logging.getLogger(__name__)
# Bounds dense memory when scoring a full training partition.
SCORING_ROWS = 100_000
# Measured on one million rows: 6 threads made this small network slower than 2.
MLP_THREADS = 2


class NeuralModel:
    """Fitted preprocessing plus a serialized ONNX network."""

    def __init__(self, preprocessor, network: bytes, spec: FeatureSet):
        self.preprocessor = preprocessor
        self.network = network
        self.spec = spec
        self.fraud_feature_spec_ = spec.model_dump()
        options = ort.SessionOptions()
        # One thread keeps scores bit-identical across machines for release checks.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            network, options, providers=["CPUExecutionProvider"]
        )

    def logits(self, frame) -> np.ndarray:
        parts = []
        for start in range(0, len(frame), SCORING_ROWS):
            matrix = _dense(
                self.preprocessor.transform(
                    normalize_features(
                        frame.iloc[start : start + SCORING_ROWS], self.spec
                    )
                )
            )
            parts.append(self.session.run(None, {"features": matrix})[0][:, 0])
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _network(width, parameters):
    from torch import nn

    sizes = [width] + [parameters.hidden_units] * parameters.hidden_layers
    layers = []
    for inputs, outputs in zip(sizes, sizes[1:]):
        layers += [
            nn.Linear(inputs, outputs),
            nn.ReLU(),
            nn.Dropout(parameters.dropout),
        ]
    return nn.Sequential(*layers, nn.Linear(sizes[-1], 1))


def _mean_loss(network, loss, matrix, labels, batch_size):
    import torch

    network.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            rows = slice(start, start + batch_size)
            total += loss(
                network(torch.from_numpy(_dense(matrix[rows]))).squeeze(1),
                torch.from_numpy(labels[rows]),
            ).item() * len(labels[rows])
    return total / len(labels)


def _export(network, width) -> bytes:
    import torch

    # The exporter warns about optional vision operators this network never uses.
    exporter_log = logging.getLogger("torch.onnx")
    level = exporter_log.level
    exporter_log.setLevel(logging.ERROR)
    try:
        program = torch.onnx.export(
            network.eval(),
            (torch.zeros(2, width),),
            dynamo=True,
            input_names=["features"],
            output_names=["fraud_logit"],
            dynamic_shapes=({0: torch.export.Dim("rows")},),
            verbose=False,
        )
    finally:
        exporter_log.setLevel(level)
    return program.model_proto.SerializeToString()


def fit_network(recipe, features, labels, *, validation=None) -> NeuralModel:
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(min(MLP_THREADS, previous))
    try:
        return _fit(recipe, features, labels, validation)
    finally:
        torch.set_num_threads(previous)


def _fit(recipe, features, labels, validation) -> NeuralModel:
    import torch
    from torch import nn

    spec = recipe.features
    if spec.version == "basic-v1":
        raise ValueError("MLP recipes require a configured feature set")
    parameters = MLPParameters.model_validate(recipe.parameters)
    torch.manual_seed(parameters.random_state)
    shuffle = np.random.default_rng(parameters.random_state)
    preprocessor = one_hot_preprocessor(spec)
    matrix = sparse.csr_matrix(
        preprocessor.fit_transform(normalize_features(features, spec)),
        dtype=np.float32,
    )
    targets = np.asarray(labels, dtype=np.float32)
    held_out = None
    if validation is not None:
        held_out = (
            sparse.csr_matrix(
                preprocessor.transform(normalize_features(validation[0], spec)),
                dtype=np.float32,
            ),
            np.asarray(validation[1], dtype=np.float32),
        )
    network = _network(matrix.shape[1], parameters)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(parameters.pos_weight))
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )
    # Validation loss is recorded as a learning curve only; it never stops training.
    curve = {"train_loss": [], "validation_loss": []}
    for epoch in range(parameters.epochs):
        network.train()
        order = shuffle.permutation(len(targets))
        total = 0.0
        for start in range(0, len(order), parameters.batch_size):
            rows = order[start : start + parameters.batch_size]
            optimizer.zero_grad()
            value = loss(
                network(torch.from_numpy(_dense(matrix[rows]))).squeeze(1),
                torch.from_numpy(targets[rows]),
            )
            value.backward()
            optimizer.step()
            total += value.item() * len(rows)
        curve["train_loss"].append(total / len(targets))
        if held_out is not None:
            curve["validation_loss"].append(
                _mean_loss(network, loss, *held_out, parameters.batch_size)
            )
        logger.info(
            "MLP epoch %d/%d train_loss=%.6f%s",
            epoch + 1,
            parameters.epochs,
            curve["train_loss"][-1],
            f" validation_loss={curve['validation_loss'][-1]:.6f}"
            if held_out is not None
            else "",
        )
    model = NeuralModel(preprocessor, _export(network, matrix.shape[1]), spec)
    sample = matrix[: min(1000, matrix.shape[0])]
    with torch.no_grad():
        expected = network.eval()(torch.from_numpy(_dense(sample))).numpy()[:, 0]
    exported = model.session.run(None, {"features": _dense(sample)})[0][:, 0]
    if not np.allclose(exported, expected, rtol=1e-4, atol=1e-5):
        raise ValueError("ONNX export does not reproduce the trained network")
    model.training_curve_ = curve
    return model


class NeuralAdapter:
    files = ("preprocessor.joblib", "model.onnx")

    def fit(self, recipe, features, labels, *, validation=None):
        return fit_network(recipe, features, labels, validation=validation)

    def scores(self, model, features):
        if list(features.columns) != model.spec.columns:
            raise ValueError("Model does not match its declared feature contract")
        # Sigmoid in float64 keeps distinct logits distinct near 0 and 1.
        values = expit(model.logits(features).astype(np.float64))
        return checked_probabilities(values, len(features))

    def save(self, model, directory: Path):
        joblib.dump(
            {
                "preprocessor": model.preprocessor,
                "feature_spec": model.fraud_feature_spec_,
            },
            directory / "preprocessor.joblib",
        )
        (directory / "model.onnx").write_bytes(model.network)

    def load(self, directory: Path):
        saved = joblib.load(directory / "preprocessor.joblib")
        return NeuralModel(
            saved["preprocessor"],
            (directory / "model.onnx").read_bytes(),
            FeatureSet.model_validate(saved["feature_spec"]),
        )
