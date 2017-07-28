import pickle
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import torch
from torch import nn
import torch.nn.functional as F

from inferno.net import to_numpy


torch.manual_seed(0)
torch.cuda.manual_seed(0)


class MyClassifier(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyClassifier, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


class TestNeuralNet:
    @pytest.fixture(scope='module')
    def data(self):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        return X.astype(np.float32), y

    @pytest.fixture(scope='module')
    def dummy_callback(self):
        from inferno.callbacks import Callback
        return Mock(spec=Callback)

    @pytest.fixture(scope='module')
    def module_cls(self, data):
        return MyClassifier

    @pytest.fixture(scope='module')
    def net_cls(self):
        from inferno.net import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls, dummy_callback):
        return net_cls(
            module_cls,
            callbacks=[('dummy', dummy_callback)],
            max_epochs=10,
            lr=0.1,
        )

    @pytest.fixture(scope='module')
    def pipe(self, net):
        return Pipeline([
            ('scale', StandardScaler()),
            ('net', net),
        ])

    @pytest.fixture(scope='module')
    def net_fit(self, net, data):
        # Careful, don't call additional fits on this, since that would have
        # side effects on other tests.
        X, y = data
        return net.fit(X, y)

    @pytest.fixture
    def net_pickleable(self, net_fit):
        # callback fixture not pickleable, remove it
        callbacks = net_fit.callbacks
        net_fit.callbacks = []
        callbacks_ = net_fit.callbacks_
        net_fit.callbacks_ = net_fit.callbacks_[:-1]
        net_clone = clone(net_fit)
        net_fit.callbacks = callbacks
        net_fit.callbacks_ = callbacks_
        return net_clone

    def test_fit(self, net_fit):
        # fitting does not raise anything
        pass

    def test_net_learns(self, net_fit, data):
        X, y = data
        y_pred = net_fit.predict(X)
        assert accuracy_score(y, y_pred) > 0.7

    def test_predict_proba(self, net_fit, data):
        X, y = data

        y_proba = net_fit.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1, rtol=1e-7)

        y_pred = net_fit.predict(X)
        assert np.allclose(np.argmax(y_proba, 1), y_pred, rtol=1e-7)

    def test_dropout(self, net_fit, data):
        # Note: does not test that dropout is really active during
        # training.
        X, y = data

        # check that dropout not active by default
        y_proba = to_numpy(net_fit.forward(X))
        y_proba2 = to_numpy(net_fit.forward(X))
        assert np.allclose(y_proba, y_proba2, rtol=1e-7)

        # check that dropout can be activated
        y_proba = to_numpy(net_fit.forward(X, training_behavior=True))
        y_proba2 = to_numpy(net_fit.forward(X, training_behavior=True))
        assert not np.allclose(y_proba, y_proba2, rtol=1e-7)

    def test_pickle_save_load(self, net_cls, net_pickleable, data, tmpdir):
        X, y = data
        score_before = accuracy_score(y, net_pickleable.predict(X))

        p = tmpdir.mkdir('inferno').join('testmodel.pkl')
        with open(str(p), 'wb') as f:
            pickle.dump(net_pickleable, f)
        del net_pickleable
        with open(str(p), 'rb') as f:
            net_new = pickle.load(f)

        score_after = accuracy_score(y, net_new.predict(X))
        assert np.isclose(score_after, score_before)

    @pytest.mark.parametrize('method, call_count', [
        ('on_train_begin', 1),
        ('on_train_end', 1),
        ('on_epoch_begin', 10),
        ('on_epoch_end', 10),
        # by default: 80/20 train/valid split
        ('on_batch_begin', (800 // 128 + 1) * 10 + (200 // 128 + 1) * 10),
        ('on_batch_end', (800 // 128 + 1) * 10 + (200 // 128 + 1) * 10),
    ])
    def test_callback_is_called(self, net_fit, method, call_count):
        method = getattr(net_fit.callbacks_[-1][1], method)
        assert method.call_count == call_count
        assert method.call_args_list[0][0][0] is net_fit

    def test_history_correct_shape(self, net_fit):
        assert len(net_fit.history) == net_fit.max_epochs

    def test_history_default_keys(self, net_fit):
        expected_keys = {
            'train_loss', 'valid_loss', 'epoch', 'dur', 'batches', 'valid_acc'}
        for row in net_fit.history:
            assert expected_keys.issubset(row)

    def test_history_is_filled(self, net_fit):
        assert len(net_fit.history) == net_fit.max_epochs

    def test_set_params_works(self, net, data):
        X, y = data
        net.fit(X, y)

        assert net.module_.dense0.out_features == 10
        assert net.module_.dense1.in_features == 10
        assert net.module_.nonlin is F.relu
        assert np.isclose(net.lr, 0.1)

        net.set_params(
            module__num_units=20,
            module__nonlin=F.tanh,
            lr=0.2,
        )
        net.fit(X, y)

        assert net.module_.dense0.out_features == 20
        assert net.module_.dense1.in_features == 20
        assert net.module_.nonlin is F.tanh
        assert np.isclose(net.lr, 0.2)

    def test_module_params_in_init(self, net_cls, module_cls, data):
        X, y = data

        net = net_cls(
            module=module_cls,
            module__num_units=20,
            module__nonlin=F.tanh,
        )
        net.fit(X, y)

        assert net.module_.dense0.out_features == 20
        assert net.module_.dense1.in_features == 20
        assert net.module_.nonlin is F.tanh

    def test_criterion_init_with_params(self, net_cls, module_cls):
        mock = Mock()
        net = net_cls(module_cls, criterion=mock, criterion__spam='eggs')
        net.initialize()
        assert mock.call_count == 1
        assert mock.call_args_list[0][1]['spam'] == 'eggs'

    def test_criterion_set_params(self, net_cls, module_cls):
        mock = Mock()
        net = net_cls(module_cls, criterion=mock)
        net.initialize()
        net.set_params(criterion__spam='eggs')
        assert mock.call_count == 2
        assert mock.call_args_list[1][1]['spam'] == 'eggs'

    def test_callback_with_name_init_with_params(self, net_cls, module_cls):
        mock = Mock()
        net = net_cls(
            module_cls,
            criterion=Mock(),
            callbacks=[('cb0', mock)],
            callbacks__cb0__spam='eggs',
        )
        net.initialize()
        assert mock.initialize.call_count == 1
        assert mock.set_params.call_args_list[0][1]['spam'] == 'eggs'

    def test_callback_set_params(self, net_cls, module_cls):
        mock = Mock()
        net = net_cls(
            module_cls,
            criterion=Mock(),
            callbacks=[('cb0', mock)],
        )
        net.initialize()
        net.set_params(callbacks__cb0__spam='eggs')
        assert mock.initialize.call_count == 2
        assert mock.set_params.call_args_list[1][1]['spam'] == 'eggs'

    def test_callback_name_collides_with_default(self, net_cls, module_cls):
        net = net_cls(module_cls, callbacks=[('average_loss', Mock())])

        with pytest.raises(ValueError) as exc:
            net.initialize()
        expected = "The callback name 'average_loss' appears more than once."
        assert str(exc.value) == expected

    def test_callback_same_name_twice(self, net_cls, module_cls):
        callbacks = [('cb0', Mock()),
                     ('cb1', Mock()),
                     ('cb0', Mock())]
        net = net_cls(module_cls, callbacks=callbacks)

        with pytest.raises(ValueError) as exc:
            net.initialize()
        expected = "The callback name 'cb0' appears more than once."
        assert str(exc.value) == expected

    def test_callback_same_inferred_name_twice(self, net_cls, module_cls):
        cb0 = Mock()
        cb1 = Mock()
        cb0.__class__.__name__ = 'some-name'
        cb1.__class__.__name__ = 'some-name'
        net = net_cls(module_cls, callbacks=[cb0, cb1])

        with pytest.raises(ValueError) as exc:
            net.initialize()
        expected = "The callback name 'some-name' appears more than once."
        assert str(exc.value) == expected

    def test_in_sklearn_pipeline(self, pipe, data):
        X, y = data
        pipe.fit(X, y)
        pipe.predict(X)
        pipe.predict_proba(X)
        pipe.set_params(net__module__num_units=20)

    def test_grid_search_works(self, net_cls, module_cls, data):
        net = net_cls(module_cls)
        X, y = data
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
            'module__num_units': [10, 20],
        }
        gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy')
        gs.fit(X[:100], y[:100])  # for speed
        print(gs.best_score_, gs.best_params_)

    def test_change_get_loss(self, net_cls, module_cls, data):
        class MyNet(net_cls):
            def get_loss(self, y_pred, y_true, train=False):
                loss_a = torch.abs(y_true.float() - y_pred[:, 1]).mean()
                loss_b = ((y_true.float() - y_pred[:, 1]) ** 2).mean()
                if train:
                    self.history.record_batch('loss_a', to_numpy(loss_a)[0])
                    self.history.record_batch('loss_b', to_numpy(loss_b)[0])
                return loss_a + loss_b

        X, y = data
        net = MyNet(module_cls, max_epochs=1)
        net.fit(X, y)

        diffs = []
        all_losses = net.history[
            -1, 'batches', :, ('train_loss', 'loss_a', 'loss_b')]
        diffs = [total - a - b for total, a, b in all_losses]
        assert np.allclose(diffs, 0, atol=1e-7)

    def test_net_no_valid(self, net_cls, module_cls, data):
        net = net_cls(
            module_cls,
            max_epochs=10,
            lr=0.1,
            train_split=None,
        )
        X, y = data
        net.fit(X, y)
        assert net.history[:, 'train_loss']
        with pytest.raises(KeyError):
            net.history[:, 'valid_loss']


class MyRegressor(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyRegressor, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.dense1(X)
        X = self.output(X)
        return X


class TestNeuralNetRegressor:
    @pytest.fixture(scope='module')
    def data(self):
        X, y = make_regression(
            1000, 20, n_informative=10, bias=0, random_state=0)
        X, y = X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)
        Xt = StandardScaler().fit_transform(X)
        yt = StandardScaler().fit_transform(y)
        return Xt, yt

    @pytest.fixture(scope='module')
    def module_cls(self, data):
        return MyRegressor

    @pytest.fixture(scope='module')
    def net_cls(self):
        from inferno.net import NeuralNetRegressor
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
        assert train_losses[0] > 3 * train_losses[-1]

    def test_history_default_keys(self, net_fit):
        expected_keys = {'train_loss', 'valid_loss', 'epoch', 'dur', 'batches'}
        for row in net_fit.history:
            assert expected_keys.issubset(row)

    def test_target_1d_raises(self, net, data):
        X, y = data
        with pytest.raises(ValueError) as exc:
            net.fit(X, y.flatten())
        assert exc.value.args[0] == (
            "The target data shouldn't be 1-dimensional; "
            "please reshape (e.g. y.reshape(-1, 1).")
