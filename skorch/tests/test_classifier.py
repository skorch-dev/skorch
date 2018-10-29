"""Tests for classifier.py

Only contains tests that are specific for classifier subclasses.

"""

from unittest.mock import Mock

from flaky import flaky
import numpy as np
import pytest
import torch
from torch import nn


torch.manual_seed(0)


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
    def module_cls(self, classifier_module):
        return classifier_module

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


class TestNeuralNetBinaryClassifier:
    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        X, y = classifier_data
        return X, y.astype('float32')

    @pytest.fixture(scope='module')
    def module_cls(self):
        from skorch.toy import make_binary_classifier
        return make_binary_classifier(
            input_units=20,
            hidden_units=10,
            output_units=1,
            num_hidden=2,
            dropout=0.5,
        )

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch.classifier import NeuralNetBinaryClassifier
        return NeuralNetBinaryClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            max_epochs=1,
            lr=1,
        )

    @pytest.fixture(scope='module')
    def net_fit(self, net, data):
        # Careful, don't call additional fits on this, since that would have
        # side effects on other tests.
        net.set_params(max_epochs=10)
        X, y = data
        net.fit(X, y)
        net.set_params(max_epochs=1)
        return net

    def test_fit(self, net_fit):
        # fitting does not raise anything
        pass

    @flaky(max_runs=3)
    def test_net_learns(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(
            module_cls,
            max_epochs=11,
            lr=1,
        )
        net.fit(X, y)

        train_losses = net.history[:, 'train_loss']
        assert train_losses[0] > 1.3 * train_losses[-1]

        valid_acc = net.history[-1, 'valid_acc']
        assert valid_acc > 0.65

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
