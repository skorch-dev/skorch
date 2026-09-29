"""End-to-end tests for keyword arguments passed during prediction."""

import numpy as np
import torch

from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier, NeuralNetRegressor


class OffsetModule(torch.nn.Module):
    def __init__(self, output_units):
        super().__init__()
        self.linear = torch.nn.Linear(2, output_units)

    def forward(self, X, offset=0.0):
        output = self.linear(X)
        if output.shape[1] == 2:
            return output + torch.tensor([0.0, offset])
        return output + offset


def test_regressor_predict_params():
    X = np.zeros((20, 2), dtype=np.float32)
    y = np.zeros((20, 1), dtype=np.float32)
    net = NeuralNetRegressor(
        OffsetModule, module__output_units=1, max_epochs=1, verbose=0)
    net.fit(X, y)

    base = net.predict(X)
    shifted = net.predict(X, offset=3.0)
    np.testing.assert_allclose(shifted, base + 3.0, rtol=0, atol=1e-6)


def test_classifier_predict_params():
    X = np.zeros((20, 2), dtype=np.float32)
    y = np.array([0, 1] * 10)
    net = NeuralNetClassifier(
        OffsetModule, module__output_units=2, max_epochs=1, verbose=0)
    net.fit(X, y)

    base = net.predict_proba(X)
    shifted = net.predict_proba(X, offset=100.0)
    assert np.all(shifted[:, 1] > base[:, 1])
    assert np.all(net.predict(X, offset=100.0) == 1)


def test_binary_classifier_predict_params():
    X = np.zeros((20, 2), dtype=np.float32)
    y = np.array([0, 1] * 10, dtype=np.float32)
    net = NeuralNetBinaryClassifier(
        OffsetModule, module__output_units=1, max_epochs=1, verbose=0)
    net.fit(X, y)

    base = net.predict_proba(X)
    shifted = net.predict_proba(X, offset=100.0)
    assert np.all(shifted[:, 1] > base[:, 1])
    assert np.all(net.predict(X, offset=100.0) == 1)
