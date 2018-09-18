"""Tests for callbacks in training.py"""

from functools import partial
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest


class TestCheckpoint:
    @pytest.fixture
    def checkpoint_cls(self):
        from skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            mock.side_effect = lambda x: x
            yield mock

    @pytest.fixture
    def save_history_mock(self):
        with patch('skorch.NeuralNet.save_history') as mock:
            yield mock

    @pytest.fixture
    def pickle_dump_mock(self):
        with patch('pickle.dump') as mock:
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch import NeuralNetRegressor
        from skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_none_monitor_saves_always(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        sink = Mock()
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, sink=sink),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == len(net.history)
        assert sink.call_count == len(net.history)
        assert all((x is True) for x in net.history[:, 'event_cp'])

    def test_default_without_validation_raises_meaningful_error(
            self, net_cls, checkpoint_cls, data):
        net = net_cls(
            callbacks=[
                checkpoint_cls(),
            ],
            train_split=None
        )
        from skorch.exceptions import SkorchException
        with pytest.raises(SkorchException) as e:
            net.fit(*data)
            expected = (
                "Monitor value '{}' cannot be found in history. "
                "Make sure you have validation data if you use "
                "validation scores for checkpointing.".format(
                    'valid_loss_best')
            )
            assert str(e.value) == expected

    def test_string_monitor_and_formatting(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True)

        sink = Mock()
        net = net_cls(callbacks=[
            ('my_score', scoring),
            checkpoint_cls(
                monitor='epoch_3_scorer',
                f_params='model_{last_epoch[epoch]}_{net.max_epochs}.pt',
                sink=sink),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 1
        save_params_mock.assert_called_with('model_3_10.pt')
        assert sink.call_count == 1
        assert all((x is False) for x in net.history[:2, 'event_cp'])
        assert net.history[2, 'event_cp'] is True
        assert all((x is False) for x in net.history[3:, 'event_cp'])

    def test_save_all_targets(
            self, save_params_mock, save_history_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data):
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, f_params='params.pt',
                f_history='history.json', f_pickle='model.pkl'),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == len(net.history)
        assert save_history_mock.call_count == len(net.history)
        assert pickle_dump_mock.call_count == len(net.history)
        save_params_mock.assert_called_with('params.pt')
        save_history_mock.assert_called_with('history.json')

    def test_save_no_targets(
            self, save_params_mock, save_history_mock, pickle_dump_mock,
            net_cls, checkpoint_cls, data):
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None, f_params=None, f_history=None,
                f_pickle=None),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 0
        assert save_history_mock.call_count == 0
        assert pickle_dump_mock.call_count == 0

    def test_target_argument(self, net_cls, checkpoint_cls):
        # TODO: remove this test when the target argument is removed
        # after its deprecation grace period is over.
        with pytest.warns(DeprecationWarning):
            checkpoint = checkpoint_cls(target='foobar.pt')
        assert checkpoint.f_params == 'foobar.pt'


class TestEarlyStopping:

    @pytest.fixture
    def early_stopping_cls(self):
        from skorch.callbacks import EarlyStopping
        return EarlyStopping

    @pytest.fixture
    def epoch_scoring_cls(self):
        from skorch.callbacks import EpochScoring
        return EpochScoring

    @pytest.fixture
    def net_clf_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def broken_classifier_module(self, classifier_module):
        """Return a classifier that does not improve over time."""
        class BrokenClassifier(classifier_module.func):
            def forward(self, x):
                return super().forward(x) * 0 + 0.5
        return BrokenClassifier

    def test_typical_use_case_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_typical_use_case_stopping(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        side_effect = []

        def sink(x):
            side_effect.append(x)

        early_stopping_cb = early_stopping_cls(patience=patience, sink=sink)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

        # check correct output message
        assert len(side_effect) == 1
        msg = side_effect[0]
        expected_msg = ("Stopping since valid_loss has not improved in "
                        "the last 5 epochs.")
        assert msg == expected_msg

    def test_custom_scoring_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_mock = Mock(side_effect=list(range(2, 10)))
        scoring_cb = epoch_scoring_cls(
            scoring_mock, lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_custom_scoring_stop(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_cb = epoch_scoring_cls(
            'roc_auc', lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) < max_epochs

    def test_stopping_big_absolute_threshold(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience,
                                               threshold_mode='abs',
                                               threshold=0.1)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

    def test_wrong_threshold_mode(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(
            patience=patience, threshold_mode='incorrect')
        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )

        with pytest.raises(ValueError) as exc:
            net.fit(*classifier_data)

        expected_msg = "Invalid threshold mode: 'incorrect'"
        assert exc.value.args[0] == expected_msg



class TestParamMapper:

    @pytest.fixture
    def initializer(self):
        from skorch.callbacks import Initializer
        return Initializer

    @pytest.fixture
    def freezer(self):
        from skorch.callbacks import Freezer
        return Freezer

    @pytest.fixture
    def unfreezer(self):
        from skorch.callbacks import Unfreezer
        return Unfreezer

    @pytest.fixture
    def param_mapper(self):
        from skorch.callbacks import ParamMapper
        return ParamMapper

    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.mark.parametrize('at', [0, -1])
    def test_subzero_at_fails(self, net_cls, classifier_module,
                              param_mapper, at):
        cb = param_mapper(patterns='*', at=at)
        net = net_cls(classifier_module, callbacks=[cb])
        with pytest.raises(ValueError):
            net.initialize()

    @pytest.mark.parametrize('mod_init', [False, True])
    @pytest.mark.parametrize('weight_pattern', [
        'sequential.*.weight',
        lambda name: name.startswith('sequential') and name.endswith('.weight'),
    ])
    def test_initialization_is_effective(self, net_cls, classifier_module,
                                         classifier_data, initializer,
                                         mod_init, weight_pattern):
        from torch.nn.init import constant_
        from skorch.utils import to_numpy

        module = classifier_module() if mod_init else classifier_module

        net = net_cls(
            module,
            lr=0,
            max_epochs=1,
            callbacks=[
                initializer(weight_pattern, partial(constant_, val=5)),
                initializer('sequential.3.bias', partial(constant_, val=10)),
            ])

        net.fit(*classifier_data)

        assert np.allclose(to_numpy(net.module_.sequential[0].weight), 5)
        assert np.allclose(to_numpy(net.module_.sequential[3].weight), 5)
        assert np.allclose(to_numpy(net.module_.sequential[3].bias), 10)

    @pytest.mark.parametrize('mod_init', [False, True])
    @pytest.mark.parametrize('mod_kwargs', [
        {},
        # Supply a module__ parameter so the model is forced
        # to re-initialize. Even then parameters should be
        # frozen correctly.
        {'module__hidden_units': 5},
    ])
    def test_freezing_is_effective(self, net_cls, classifier_module,
                                   classifier_data, freezer, mod_init,
                                   mod_kwargs):
        from skorch.utils import to_numpy

        module = classifier_module() if mod_init else classifier_module

        net = net_cls(
            module,
            max_epochs=2,
            callbacks=[
                freezer('sequential.*.weight'),
                freezer('sequential.3.bias'),
            ],
            **mod_kwargs)

        net.initialize()

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # use partial_fit to not re-initialize the module (weights)
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        assert np.allclose(dense0_weight_pre, dense0_weight_post)
        assert np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert np.allclose(dense1_bias_pre, dense1_bias_post)

    def test_unfreezing_is_effective(self, net_cls, classifier_module,
                                     classifier_data, freezer, unfreezer):
        from skorch.utils import to_numpy

        net = net_cls(
            classifier_module,
            max_epochs=1,
            callbacks=[
                freezer('sequential.*.weight'),
                freezer('sequential.3.bias'),
                unfreezer('sequential.*.weight', at=2),
                unfreezer('sequential.3.bias', at=2),
            ])

        net.initialize()

        # epoch 1, freezing parameters
        net.partial_fit(*classifier_data)

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # epoch 2, unfreezing parameters
        net.partial_fit(*classifier_data)

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        # epoch 3, modifications should have been made
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not np.allclose(dense0_weight_pre, dense0_weight_post)
        assert not np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert not np.allclose(dense1_bias_pre, dense1_bias_post)


    def test_schedule_is_effective(self, net_cls, classifier_module,
                                   classifier_data, param_mapper):
        from skorch.utils import to_numpy, noop
        from skorch.utils import freeze_parameter, unfreeze_parameter

        def schedule(net):
            if len(net.history) == 1:
                return freeze_parameter
            elif len(net.history) == 2:
                return unfreeze_parameter
            return noop

        net = net_cls(
            classifier_module,
            max_epochs=1,
            callbacks=[
                param_mapper(
                    ['sequential.*.weight', 'sequential.3.bias'],
                    schedule=schedule,
                ),
            ])

        net.initialize()

        # epoch 1, freezing parameters
        net.partial_fit(*classifier_data)

        assert not net.module_.sequential[0].weight.requires_grad
        assert not net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert not net.module_.sequential[3].bias.requires_grad

        dense0_weight_pre = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_pre = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_pre = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_pre = to_numpy(net.module_.sequential[3].bias).copy()

        # epoch 2, unfreezing parameters
        net.partial_fit(*classifier_data)

        assert net.module_.sequential[0].weight.requires_grad
        assert net.module_.sequential[3].weight.requires_grad
        assert net.module_.sequential[0].bias.requires_grad
        assert net.module_.sequential[3].bias.requires_grad

        # epoch 3, modifications should have been made
        net.partial_fit(*classifier_data)

        dense0_weight_post = to_numpy(net.module_.sequential[0].weight).copy()
        dense1_weight_post = to_numpy(net.module_.sequential[3].weight).copy()
        dense0_bias_post = to_numpy(net.module_.sequential[0].bias).copy()
        dense1_bias_post = to_numpy(net.module_.sequential[3].bias).copy()

        assert not np.allclose(dense0_weight_pre, dense0_weight_post)
        assert not np.allclose(dense1_weight_pre, dense1_weight_post)
        assert not np.allclose(dense0_bias_pre, dense0_bias_post)
        assert not np.allclose(dense1_bias_pre, dense1_bias_post)
