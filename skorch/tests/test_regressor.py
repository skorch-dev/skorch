"""Tests for regressor.py

Only contains tests that are specific for regressor subclasses.

"""

import numpy as np
import pytest
from sklearn.base import clone

from skorch.tests.conftest import INFERENCE_METHODS


class TestNeuralNetRegressor:
    @pytest.fixture(scope='module')
    def data(self, regression_data):
        return regression_data

    @pytest.fixture(scope='module')
    def module_cls(self):
        from skorch.toy import make_regressor
        return make_regressor(dropout=0.5)

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
    def multioutput_module_cls(self):
        from skorch.toy import make_regressor
        return make_regressor(output_units=3, dropout=0.5)

    @pytest.fixture(scope='module')
    def multioutput_net(self, net_cls, multioutput_module_cls):
        return net_cls(
            multioutput_module_cls,
            max_epochs=1,
            lr=0.1,
        )

    @pytest.fixture(scope='module')
    def net_fit(self, net, data):
        # Careful, don't call additional fits on this, since that would have
        # side effects on other tests.
        X, y = data
        return net.fit(X, y)

    def test_clone(self, net_fit):
        clone(net_fit)

    def test_fit(self, net_fit):
        # fitting does not raise anything
        pass

    @pytest.mark.parametrize('method', INFERENCE_METHODS)
    def test_not_fitted_raises(self, net_cls, module_cls, data, method):
        from skorch.exceptions import NotInitializedError
        net = net_cls(module_cls)
        X = data[0]
        with pytest.raises(NotInitializedError) as exc:
            # we call `list` because `forward_iter` is lazy
            list(getattr(net, method)(X))

        msg = ("This NeuralNetRegressor instance is not initialized "
               "yet. Call 'initialize' or 'fit' with appropriate arguments "
               "before using this method.")
        assert exc.value.args[0] == msg

    def test_net_learns(self, net, net_cls, data, module_cls):
        X, y = data
        net = net_cls(
            module_cls,
            max_epochs=10,
            lr=0.1,
        )
        net.fit(X, y)
        train_losses = net.history[:, 'train_loss']
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

    def test_score(self, net_fit, data):
        X, y = data
        r2_score = net_fit.score(X, y)
        assert r2_score <= 1.

    def test_multioutput_score(self, multioutput_net, multioutput_regression_data):
        X, y = multioutput_regression_data
        multioutput_net.fit(X, y)
        r2_score = multioutput_net.score(X, y)
        assert r2_score <= 1.
