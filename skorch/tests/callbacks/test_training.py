"""Tests for callbacks in training.py"""

from functools import partial
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest


class TestCheckpoint:
    @pytest.yield_fixture
    def checkpoint_cls(self):
        from skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.yield_fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            mock.side_effect = lambda x: x
            yield mock

    @pytest.yield_fixture
    def save_history_mock(self):
        with patch('skorch.NeuralNet.save_history') as mock:
            yield mock

    @pytest.yield_fixture
    def pickle_dump_mock(self):
        with patch('pickle.dump') as mock:
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch import NeuralNetRegressor
        import torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Linear(1, 1)
            # pylint: disable=arguments-differ

            def forward(self, x):
                return self.p(x)

        return partial(
            NeuralNetRegressor,
            module=Module,
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
        class BrokenClassifier(classifier_module):
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
