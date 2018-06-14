
from functools import partial
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

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch.net import NeuralNetRegressor
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
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == len(net.history)

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

        net = net_cls(callbacks=[
            ('my_score', scoring),
            checkpoint_cls(
                monitor='epoch_3_scorer',
                target='model_{last_epoch[epoch]}_{net.max_epochs}.pt'),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 1
        save_params_mock.assert_called_with('model_3_10.pt')
