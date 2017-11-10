"""Contains shared fixtures, hooks, etc."""

from unittest.mock import Mock

import numpy as np
import pytest
from torch import nn


###################
# shared fixtures #
###################

@pytest.fixture
def module_cls():
    """Simple mock module for triggering scoring."""
    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.dense = nn.Linear(1, 1)

        # pylint: disable=arguments-differ
        def forward(self, X):
            X = X + 0.0 * self.dense(X)
            return X
    return MyModule


@pytest.fixture
def score55():
    """Simple scoring function."""
    # pylint: disable=unused-argument
    def func(est, X, y, foo=123):
        return 55
    func.__name__ = 'score55'
    return func


@pytest.fixture
def train_split():
    def func(X, y):
        return X[:2], X[2:], y[:2], y[2:]
    return func


@pytest.fixture
def net_cls():
    from skorch import NeuralNetRegressor
    NeuralNetRegressor.score = Mock(side_effect=[10, 8, 6, 11, 7])
    return NeuralNetRegressor


@pytest.fixture
def data():
    X = np.array([0, 2, 3, 0]).astype(np.float32)
    y = np.array([-1, 0, 5, 4]).astype(np.float32).reshape(-1, 1)
    return X, y


pandas_installed = False
try:
    # pylint: disable=unused-import
    import pandas
    pandas_installed = True
except ImportError:
    pass
