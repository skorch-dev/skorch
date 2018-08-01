"""Tests for classifier.py

Only contains tests that are specific for classifier subclasses.

"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F


torch.manual_seed(0)


class MyClassifier(nn.Module):
    """Simple classification module.

    We cannot use the module fixtures from conftest because they are
    not pickleable.

    """
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyClassifier, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    # pylint: disable=arguments-differ
    def forward(self, X):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


class TestNeuralNet:
    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope='module')
    def dummy_callback(self):
        from skorch.callbacks import Callback
        cb = Mock(spec=Callback)
        # make dummy behave like an estimator
        cb.get_params.return_value = {}
        cb.set_params = lambda **kwargs: cb
        return cb

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def module_cls(self):
        return MyClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls, dummy_callback):
        return net_cls(
            module_cls,
            callbacks=[('dummy', dummy_callback)],
            max_epochs=10,
            lr=0.1,
        )

    @pytest.fixture(scope='module')
    def net_fit(self, net, data):
        # Careful, don't call additional fits on this, since that would have
        # side effects on other tests.
        X, y = data
        return net.fit(X, y)

    def test_predict_and_predict_proba(self, net_fit, data):
        X = data[0]

        y_proba = net_fit.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1, rtol=1e-5)

        y_pred = net_fit.predict(X)
        assert np.allclose(np.argmax(y_proba, 1), y_pred, rtol=1e-5)

    # classifier-specific test
    def test_takes_log_with_nllloss(self, net_cls, module_cls, data):
        net = net_cls(module_cls, criterion=nn.NLLLoss, max_epochs=1)
        net.initialize()

        mock_loss = Mock(side_effect=nn.NLLLoss())
        net.criterion_.forward = mock_loss
        net.partial_fit(*data)  # call partial_fit to avoid re-initialization

        # check that loss was called with log-probabilities
        for (y_log, _), _ in mock_loss.call_args_list:
            assert (y_log < 0).all()
            y_proba = torch.exp(y_log)
            assert torch.isclose(torch.ones(len(y_proba)), y_proba.sum(1)).all()

    # classifier-specific test
    def test_takes_no_log_without_nllloss(self, net_cls, module_cls, data):
        net = net_cls(module_cls, criterion=nn.BCELoss, max_epochs=1)
        net.initialize()

        mock_loss = Mock(side_effect=nn.NLLLoss())
        net.criterion_.forward = mock_loss
        net.partial_fit(*data)  # call partial_fit to avoid re-initialization

        # check that loss was called with raw probabilities
        for (y_out, _), _ in mock_loss.call_args_list:
            assert not (y_out < 0).all()
            assert torch.isclose(torch.ones(len(y_out)), y_out.sum(1)).all()


class MyBinaryClassifier(nn.Module):
    """Simple binary classification module.

    We cannot use the module fixtures from conftest because they are
    not pickleable.

    """

    def __init__(self, num_units=10, nonlin=F.relu):
        super().__init__()

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
        X = self.output(X).squeeze_(-1)
        return X


class TestNeuralNetBinaryClassifier:
    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        X, y = classifier_data
        return X, y.astype('float32')

    @pytest.fixture(scope='module')
    def module_cls(self):
        return MyBinaryClassifier

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch.classifier import NeuralNetBinaryClassifier
        return NeuralNetBinaryClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            max_epochs=1,
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

    def test_history_default_keys(self, net_fit):
        expected_keys = {
            'train_loss', 'valid_loss', 'epoch', 'dur', 'batches', 'valid_acc'
        }
        for row in net_fit.history:
            assert expected_keys.issubset(row)

    @pytest.mark.parametrize('threshold', [0, 0.25, 0.5, 0.75, 1])
    def test_predict_predict_proba(self, net, data, threshold):
        X, y = data
        net.threshold = threshold
        net.fit(X, y)

        y_pred_proba = net.predict_proba(X)
        assert y_pred_proba.ndim == 1
        assert y_pred_proba.shape[0] == X.shape[0]

        y_pred_exp = (y_pred_proba > threshold).astype('uint8')

        y_pred_actual = net.predict(X)
        assert np.allclose(y_pred_exp, y_pred_actual)

    def test_target_2d_raises(self, net, data):
        X, y = data
        with pytest.raises(ValueError) as exc:
            net.fit(X, y[:, None])

        assert exc.value.args[0] == (
            "The target data should be 1-dimensional.")

    def test_custom_loss_does_not_call_sigmoid(
            self, net_cls, data, module_cls, monkeypatch):
        mock = Mock(side_effect=lambda x: x)
        monkeypatch.setattr(torch, "sigmoid", mock)

        net = net_cls(module_cls, max_epochs=1, lr=0.1, criterion=nn.MSELoss)
        X, y = data
        net.fit(X, y)

        net.predict_proba(X)
        assert mock.call_count == 0

    def test_default_loss_does_call_sigmoid(
            self, net_cls, data, module_cls, monkeypatch):
        mock = Mock(side_effect=lambda x: x)
        monkeypatch.setattr(torch, "sigmoid", mock)

        net = net_cls(module_cls, max_epochs=1, lr=0.1)
        X, y = data
        net.fit(X, y)

        net.predict_proba(X)
        assert mock.call_count > 0
