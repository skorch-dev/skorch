"""Tests for regressor.py

Only contains tests that are specific for regressor subclasses.

"""

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F


torch.manual_seed(0)


class MyRegressor(nn.Module):
    """Simple regression module.

    We cannot use the module fixtures from conftest because they are
    not pickleable.

    """
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyRegressor, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    # pylint: disable=arguments-differ
    def forward(self, X):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X


class TestNeuralNetRegressor:
    @pytest.fixture(scope='module')
    def data(self, regression_data):
        return regression_data

    @pytest.fixture(scope='module')
    def module_cls(self):
        return MyRegressor

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetRegressor
        return NeuralNetRegressor

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            max_epochs=20,
            lr=0.1,
        )

    @pytest.fixture(scope='module')
    def net_fit(self, net, data):
        # Careful, don't call additional fits on this, since that would have
        # side effects on other tests.
        X, y = data
        return net.fit(X, y)

    def test_fit(self, net_fit):
        # fitting does not raise anything
        pass

    def test_net_learns(self, net_fit):
        train_losses = net_fit.history[:, 'train_loss']
        assert train_losses[0] > 2 * train_losses[-1]

    def test_history_default_keys(self, net_fit):
        expected_keys = {'train_loss', 'valid_loss', 'epoch', 'dur', 'batches'}
        for row in net_fit.history:
            assert expected_keys.issubset(row)

    def test_target_1d_raises(self, net, data):
        X, y = data
        with pytest.raises(ValueError) as exc:
            net.fit(X, y.flatten())
        assert exc.value.args[0] == (
            "The target data shouldn't be 1-dimensional but instead have "
            "2 dimensions, with the second dimension having the same size "
            "as the number of regression targets (usually 1). Please "
            "reshape your target data to be 2-dimensional "
            "(e.g. y = y.reshape(-1, 1).")

    def test_predict_predict_proba(self, net_fit, data):
        X = data[0]
        y_pred = net_fit.predict(X)

        # predictions should not be all zeros
        assert not np.allclose(y_pred, 0)

        y_proba = net_fit.predict_proba(X)
        # predict and predict_proba should be identical for regression
        assert np.allclose(y_pred, y_proba, atol=1e-6)
