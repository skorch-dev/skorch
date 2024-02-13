"""Tests for classifier.py

Only contains tests that are specific for classifier subclasses.

"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from scipy.special import expit
from sklearn.base import clone
from torch import nn

from skorch.tests.conftest import INFERENCE_METHODS


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

    def test_clone(self, net_fit):
        clone(net_fit)

    def test_predict_and_predict_proba(self, net_fit, data):
        X = data[0]

        y_proba = net_fit.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1, rtol=1e-5)

        y_pred = net_fit.predict(X)
        assert np.allclose(np.argmax(y_proba, 1), y_pred, rtol=1e-5)

    def test_score(self, net_fit, data):
        X, y = data
        accuracy = net_fit.score(X, y)
        assert 0. <= accuracy <= 1.

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
        net = net_cls(module_cls, criterion=nn.CrossEntropyLoss, max_epochs=1)
        net.initialize()

        mock_loss = Mock(side_effect=nn.NLLLoss())
        net.criterion_.forward = mock_loss
        net.partial_fit(*data)  # call partial_fit to avoid re-initialization

        # check that loss was called with raw probabilities
        for (y_out, _), _ in mock_loss.call_args_list:
            assert not (y_out < 0).all()
            assert torch.isclose(torch.ones(len(y_out)), y_out.sum(1)).all()

    # classifier-specific test
    def test_high_learning_rate(self, net_cls, module_cls, data):
        # regression test for nan loss with high learning rates issue #481
        net = net_cls(module_cls, max_epochs=2, lr=2, optimizer=torch.optim.Adam)
        net.fit(*data)
        assert np.any(~np.isnan(net.history[:, 'train_loss']))

    def test_binary_classes_set_by_default(self, net_cls, module_cls, data):
        net = net_cls(module_cls).fit(*data)
        assert (net.classes_ == [0, 1]).all()

    def test_non_binary_classes_set_by_default(self, net_cls, module_cls, data):
        X = data[0]
        y = np.arange(len(X)) % 10
        net = net_cls(module_cls, max_epochs=0).fit(X, y)
        assert (net.classes_ == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all()

    def test_classes_data_torch_tensor(self, net_cls, module_cls, data):
        X = torch.as_tensor(data[0])
        y = torch.as_tensor(np.arange(len(X)) % 10)

        net = net_cls(module_cls, max_epochs=0).fit(X, y)
        assert (net.classes_ == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all()

    def test_classes_with_gaps(self, net_cls, module_cls, data):
        X = data[0]
        y = np.arange(len(X)) % 10
        y[(y == 0) | (y == 5)] = 4  # remove classes 0 and 5
        net = net_cls(module_cls, max_epochs=0).fit(X, y)
        assert (net.classes_ == [1, 2, 3, 4, 6, 7, 8, 9]).all()

    def test_pass_classes_explicitly_overrides(self, net_cls, module_cls, data):
        net = net_cls(module_cls, max_epochs=0, classes=['foo', 'bar']).fit(*data)
        assert (net.classes_ == np.array(['foo', 'bar'])).all()

    def test_classes_are_set_with_tensordataset_explicit_y(
            self, net_cls, module_cls, data
    ):
        # see 990
        X = torch.from_numpy(data[0])
        y = torch.arange(len(X)) % 10
        dataset = torch.utils.data.TensorDataset(X, y)
        net = net_cls(module_cls, max_epochs=0).fit(dataset, y)
        assert (net.classes_ == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all()

    def test_classes_are_set_with_tensordataset_implicit_y(
            self, net_cls, module_cls, data
    ):
        # see 990
        from skorch.dataset import ValidSplit

        X = torch.from_numpy(data[0])
        y = torch.arange(len(X)) % 10
        dataset = torch.utils.data.TensorDataset(X, y)
        net = net_cls(
            module_cls, max_epochs=0, train_split=ValidSplit(3, stratified=False)
        ).fit(dataset, None)
        assert (net.classes_ == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all()

    @pytest.mark.parametrize('classes', [[], np.array([])])
    def test_pass_empty_classes_raises(
            self, net_cls, module_cls, data, classes):
        net = net_cls(
            module_cls, max_epochs=0, classes=classes).fit(*data).fit(*data)
        with pytest.raises(AttributeError) as exc:
            net.classes_  # pylint: disable=pointless-statement

        msg = exc.value.args[0]
        expected = "NeuralNetClassifier has no attribute 'classes_'"
        assert msg == expected

    @pytest.mark.xfail
    def test_with_calibrated_classifier_cv(self, net_fit, data):
        # TODO: This fails with sklearn 1.4.0 because CCCV does not work when
        # y_proba is float32. This will be fixed in
        # https://github.com/scikit-learn/scikit-learn/pull/28247, at which
        # point the test should pass again and the xfail can be removed.
        from sklearn.calibration import CalibratedClassifierCV
        cccv = CalibratedClassifierCV(net_fit, cv=2)
        cccv.fit(*data)

    def test_error_when_classes_could_not_be_inferred(self, net_cls, module_cls, data):
        # Provide a better error message when net.classes_ does not exist,
        # though it is pretty difficult to know exactly the circumstanes that
        # led to this, so we have to make a guess.
        # See https://github.com/skorch-dev/skorch/discussions/1003
        class MyDataset(torch.utils.data.Dataset):
            """Dataset class that makes it impossible to access y"""
            def __len__(self):
                return len(data[0])

            def __getitem__(self, i):
                return data[0][i], data[1][i]

        net = net_cls(module_cls, max_epochs=0, train_split=False)
        ds = MyDataset()
        net.fit(ds, y=None)

        msg = (
            "NeuralNetClassifier could not infer the classes from y; "
            "this error probably occurred because the net was trained without y "
            "and some function tried to access the '.classes_' attribute; "
            "a possible solution is to provide the 'classes' argument when "
            "initializing NeuralNetClassifier"
        )
        with pytest.raises(AttributeError, match=msg):
            net.classes_


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
            num_hidden=1,
            dropout=0,
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

    def test_clone(self, net_fit):
        clone(net_fit)

    @pytest.mark.parametrize('method', INFERENCE_METHODS)
    def test_not_fitted_raises(self, net_cls, module_cls, data, method):
        from skorch.exceptions import NotInitializedError
        net = net_cls(module_cls)
        X = data[0]
        with pytest.raises(NotInitializedError) as exc:
            # we call `list` because `forward_iter` is lazy
            list(getattr(net, method)(X))

        msg = ("This NeuralNetBinaryClassifier instance is not initialized "
               "yet. Call 'initialize' or 'fit' with appropriate arguments "
               "before using this method.")
        assert exc.value.args[0] == msg

    def test_net_learns(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(
            module_cls,
            max_epochs=10,
            lr=1,
            batch_size=64,
        )
        net.fit(X, y)

        train_losses = net.history[:, 'train_loss']
        assert train_losses[0] > 1.3 * train_losses[-1]

        valid_acc = net.history[-1, 'valid_acc']
        assert valid_acc > 0.65

    def test_batch_size_one(self, net_cls, module_cls, data):
        X, y = data
        net = net_cls(
            module_cls,
            max_epochs=1,
            batch_size=1,
        )
        net.fit(X, y)

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
        assert y_pred_proba.shape == (X.shape[0], 2)

        # The tests below check that we don't accidentally apply sigmoid twice,
        # which would result in probabilities constrained to [expit(-1),
        # expit(1)]. The lower bound is not expit(0), as one may think at first,
        # because we create the probabilities as:
        # torch.stack((1 - prob, prob), 1)
        # So the lowest value that could be achieved by applying sigmoid twice
        # is 1 - expit(1), which is equal to expit(-1).
        prob_min, prob_max = expit(-1), expit(1)
        assert (y_pred_proba < prob_min).any()
        assert (y_pred_proba > prob_max).any()

        y_pred_exp = (y_pred_proba[:, 1] > threshold).astype('uint8')

        y_pred_actual = net.predict(X)
        assert np.allclose(y_pred_exp, y_pred_actual)

    def test_score(self, net_fit, data):
        X, y = data
        accuracy = net_fit.score(X, y)
        assert 0. <= accuracy <= 1.

    def test_fit_with_dataset_and_y_none(self, net_cls, module_cls, data):
        from skorch.dataset import Dataset

        # deactivate train split since it requires y
        net = net_cls(module_cls, train_split=False, max_epochs=1)
        X, y = data
        dataset = Dataset(X, y)
        assert net.fit(dataset, y=None)

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

        # add a custom nonlinearity - note that the output must return
        # a 2d array from a 1d vector to conform to the required
        # y_proba
        def nonlin(x):
            return torch.stack((1 - x, x), 1)

        net = net_cls(module_cls, max_epochs=1, lr=0.1, criterion=nn.MSELoss,
                      predict_nonlinearity=nonlin)
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

    @pytest.mark.xfail
    def test_with_calibrated_classifier_cv(self, net_fit, data):
        # TODO: This fails with sklearn 1.4.0 because CCCV does not work when
        # y_proba is float32. This will be fixed in
        # https://github.com/scikit-learn/scikit-learn/pull/28247, at which
        # point the test should pass again and the xfail can be removed.
        from sklearn.calibration import CalibratedClassifierCV
        cccv = CalibratedClassifierCV(net_fit, cv=2)
        cccv.fit(*data)

    def test_grid_search_with_roc_auc(self, net_fit, data):
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(
            net_fit,
            {'max_epochs': [1, 2]},
            refit=False,
            cv=3,
            scoring='roc_auc',
        )
        search.fit(*data)

    def test_module_output_not_1d(self, net_cls, data):
        from skorch.toy import make_classifier
        module = make_classifier(
            input_units=20,
            output_units=1,
        )  # the output will not be squeezed
        net = net_cls(module, max_epochs=1)
        net.fit(*data)  # does not raise

    def test_module_output_2d_raises(self, net_cls, data):
        from skorch.toy import make_classifier
        module = make_classifier(
            input_units=20,
            output_units=2,
        )
        net = net_cls(module, max_epochs=1)
        with pytest.raises(ValueError) as exc:
            net.fit(*data)

        msg = exc.value.args[0]
        expected = ("Expected module output to have shape (n,) or "
                    "(n, 1), got (128, 2) instead")
        assert msg == expected

    @pytest.fixture(scope='module')
    def net_with_bceloss(self, net_cls, module_cls, data):
        # binary classification should also work with BCELoss
        net = net_cls(
            module_cls,
            module__output_nonlin=torch.nn.Sigmoid(),
            criterion=torch.nn.BCELoss,
            lr=1,
        )
        X, y = data
        net.fit(X, y)
        return net

    def test_net_with_bceloss_learns(self, net_with_bceloss):
        train_losses = net_with_bceloss.history[:, 'train_loss']
        assert train_losses[0] > 1.3 * train_losses[-1]

    def test_predict_proba_with_bceloss(self, net_with_bceloss, data):
        X, _ = data
        y_proba = net_with_bceloss.predict_proba(X)

        assert y_proba.shape == (X.shape[0], 2)
        assert (y_proba >= 0).all()
        assert (y_proba <= 1).all()

        # The tests below check that we don't accidentally apply sigmoid twice,
        # which would result in probabilities constrained to [expit(-1),
        # expit(1)]. The lower bound is not expit(0), as one may think at first,
        # because we create the probabilities as:
        # torch.stack((1 - prob, prob), 1)
        # So the lowest value that could be achieved by applying sigmoid twice
        # is 1 - expit(1), which is equal to expit(-1).
        prob_min, prob_max = expit(-1), expit(1)
        assert (y_proba < prob_min).any()
        assert (y_proba > prob_max).any()

    def test_predict_with_bceloss(self, net_with_bceloss, data):
        X, _ = data

        y_pred_proba = net_with_bceloss.predict_proba(X)
        y_pred_exp = (y_pred_proba[:, 1] > net_with_bceloss.threshold).astype('uint8')
        y_pred_actual = net_with_bceloss.predict(X)
        assert np.allclose(y_pred_exp, y_pred_actual)
