"""Contains shared fixtures, hooks, etc."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

F = nn.functional

INFERENCE_METHODS = ['predict', 'predict_proba', 'forward', 'forward_iter']


###################
# shared fixtures #
###################


@pytest.fixture(autouse=True)
def seeds_fixed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def module_cls():
    """Simple mock module for triggering scoring.

    This module returns the input without modifying it.

    """

    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.dense = nn.Linear(1, 1)

        # pylint: disable=arguments-differ
        def forward(self, X):
            X = X + 0.0 * self.dense(X)
            return X

    return MyModule


@pytest.fixture(scope='module')
def classifier_module():
    """Return a simple classifier module class."""
    from skorch.toy import make_classifier
    return make_classifier(
        input_units=20,
        hidden_units=10,
        num_hidden=2,
        dropout=0.5,
    )


@pytest.fixture(scope='module')
def multiouput_module():
    """Return a simple classifier module class."""

    class MultiOutput(nn.Module):
        """Simple classification module."""

        def __init__(self, input_units=20):
            super(MultiOutput, self).__init__()
            self.output = nn.Linear(input_units, 2)

        # pylint: disable=arguments-differ
        def forward(self, X):
            X = F.softmax(self.output(X), dim=-1)
            return X, X[:, 0], X[::2]

    return MultiOutput


@pytest.fixture(scope='module')
def classifier_data():
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    return X.astype(np.float32), y


@pytest.fixture(scope='module')
def regression_data():
    X, y = make_regression(
        1000, 20, n_informative=10, bias=0, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)
    Xt = StandardScaler().fit_transform(X)
    yt = StandardScaler().fit_transform(y)
    return Xt, yt


@pytest.fixture(scope='module')
def multioutput_regression_data():
    X, y = make_regression(
        1000, 20, n_targets=3, n_informative=10, bias=0, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.float32)
    Xt = StandardScaler().fit_transform(X)
    yt = StandardScaler().fit_transform(y)
    return Xt, yt


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
    # pylint: disable=unused-argument
    def func(dataset, y):
        ds_train = type(dataset)(dataset.X[:2], dataset.y[:2])
        ds_valid = type(dataset)(dataset.X[2:], dataset.y[2:])
        return ds_train, ds_valid

    return func


@pytest.fixture
def net_cls():
    from skorch import NeuralNetRegressor
    return NeuralNetRegressor


@pytest.fixture
def data():
    X = np.array([0, 2, 3, 0]).astype(np.float32).reshape(-1, 1)
    y = np.array([-1, 0, 5, 4]).astype(np.float32).reshape(-1, 1)
    return X, y


neptune_installed = False
try:
    # pylint: disable=unused-import
    import neptune

    neptune_installed = True
except ImportError:
    pass

sacred_installed = False
try:
    # pylint: disable=unused-import
    import sacred

    sacred_installed = True
except ImportError:
    pass

wandb_installed = False
try:
    # pylint: disable=unused-import
    import wandb

    wandb_installed = True
except ImportError:
    pass

pandas_installed = False
try:
    # pylint: disable=unused-import
    import pandas

    pandas_installed = True
except ImportError:
    pass

tensorboard_installed = False
try:
    # pylint: disable=unused-import
    import tensorboard

    tensorboard_installed = True
except ImportError:
    pass

mlflow_installed = False
try:
    # pylint: disable=unused-import
    import mlflow

    mlflow_installed = True
except ImportError:
    pass

