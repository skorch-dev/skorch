"""Tests for regressor.py

Only contains tests that are specific for regressor subclasses.

"""

from functools import partial

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
    def module_pred_1d_cls(self):
        from skorch.toy import MLPModule
        # Module that returns 1d predictions
        return partial(MLPModule, output_units=1, squeeze_output=True)

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

    def test_fit(self, net_fit, recwarn):
        # fitting does not raise anything and does not warn
        assert not recwarn.list

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

    def test_dimension_mismatch_warning(self, net_cls, module_cls, data, recwarn):
        # When the target and the prediction have different dimensionality, mse
        # loss will broadcast them, calculating all pairwise errors instead of
        # only sample-wise. Since the errors are averaged at the end, there is
        # still a valid loss, which makes the error hard to spot. Thankfully,
        # torch gives a warning in that case. We test that this warning exists,
        # otherwise, skorch users could run into very hard to debug issues
        # during training.
        net = net_cls(module_cls)
        X, y = data
        X, y = X[:100], y[:100].flatten()  # make y 1d
        net.fit(X, y)

        # The warning comes from PyTorch, so checking the exact wording is prone to
        # error in future PyTorch versions. We thus check a substring of the
        # whole message and cross our fingers that it's not changed.
        msg_substr = (
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size"
        )
        warn_list = [w for w in recwarn.list if msg_substr in str(w.message)]
        # one warning for train, one for valid
        assert len(warn_list) == 2

    def test_fitting_with_1d_target_and_pred(
            self, net_cls, module_cls, data, module_pred_1d_cls, recwarn
    ):
        # This test relates to the previous one. In general, users should fit
        # with target and prediction being 2d, even if the 2nd dimension is just
        # 1. However, in some circumstances (like when using BaggingRegressor,
        # see next test), having the ability to fit with 1d is required. In that
        # case, the module output also needs to be 1d for correctness.
        X, y = data
        X, y = X[:100], y[:100]  # less data to run faster
        y = y.flatten()

        net = net_cls(module_pred_1d_cls)
        net.fit(X, y)
        msg_substr = (
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size"
        )
        assert not any(msg_substr in str(w.message) for w in recwarn.list)

    def test_bagging_regressor(
            self, net_cls, module_cls, data, module_pred_1d_cls, recwarn
    ):
        # https://github.com/skorch-dev/skorch/issues/972
        from sklearn.ensemble import BaggingRegressor

        net = net_cls(module_pred_1d_cls)  # module output should be 1d too
        X, y = data
        X, y = X[:100], y[:100]  # less data to run faster
        y = y.flatten()  # make y 1d or else sklearn will complain
        regr = BaggingRegressor(net, n_estimators=2, random_state=0)
        regr.fit(X, y)  # does not raise
        # ensure there is no broadcast warning from torch
        msg_substr = (
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size"
        )
        assert not any(msg_substr in str(w.message) for w in recwarn.list)
