from unittest.mock import Mock

import numpy as np
import pytest

from .conftest import get_history


class TestAverageLoss:
    @pytest.fixture
    def avg_loss_cls(self):
        from inferno.callbacks import AverageLoss
        return AverageLoss

    @pytest.fixture
    def avg_loss(self, avg_loss_cls):
        return avg_loss_cls().initialize()

    @pytest.fixture
    def history_avg_loss(self, avg_loss):
        return get_history(avg_loss)

    def test_correct_losses(self, history_avg_loss):
        train_losses = history_avg_loss[:, 'train_loss']
        expected = [0.25, 0.65, -0.15]
        assert np.allclose(train_losses, expected)

        valid_losses = history_avg_loss[:, 'valid_loss']
        expected = [7.5, 3.5, 11.5]
        assert np.allclose(valid_losses, expected)

    def test_missing_batch_size(self, avg_loss, history):
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 20)
        # missing batch size, 20 is ignored

        net = Mock(history=history)
        avg_loss.on_epoch_end(net)

        assert history[0, 'train_loss'] == 10

    def test_average_honors_weights(self, avg_loss, history):
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 40)
        history.record_batch('train_batch_size', 2)

        net = Mock(history=history)
        avg_loss.on_epoch_end(net)

        assert history[0, 'train_loss'] == 30

    def test_init_other_keys(self, avg_loss_cls):
        avg_loss = avg_loss_cls(keys_possible=[
            ('train_loss', 'train_batch_size')]).initialize()
        history = get_history(avg_loss)

        train_losses = history[:, 'train_loss']
        expected = [0.25, 0.65, -0.15]
        assert np.allclose(train_losses, expected)

        with pytest.raises(KeyError):
            history[:, 'valid_loss']


class TestBestLoss:
    @pytest.fixture
    def avg_loss(self):
        from inferno.callbacks import AverageLoss
        return AverageLoss()

    @pytest.fixture
    def best_loss_cls(self):
        from inferno.callbacks import BestLoss
        return BestLoss

    @pytest.fixture
    def best_loss(self, best_loss_cls):
        return best_loss_cls(signs=(-1, 1)).initialize()

    @pytest.fixture
    def history_best_loss(self, avg_loss, best_loss):
        return get_history(avg_loss, best_loss)

    def test_best_loss_correct(self, history_best_loss):
        train_loss_best = history_best_loss[:, 'train_loss_best']
        expected = [True, False, True]
        assert train_loss_best == expected

        valid_loss_best = history_best_loss[:, 'valid_loss_best']
        expected = [True, False, True]
        assert valid_loss_best == expected

    def test_other_signs(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(signs=(1, -1)).initialize()
        history = get_history(avg_loss, best_loss)

        train_loss_best = history[:, 'train_loss_best']
        expected = [True, True, False]
        assert train_loss_best == expected

        valid_loss_best = history[:, 'valid_loss_best']
        expected = [True, True, False]
        assert valid_loss_best == expected

    def test_init_other_keys(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(
            keys_possible=('valid_loss',),
            signs=(1,),
        ).initialize()
        history = get_history(avg_loss, best_loss)

        with pytest.raises(KeyError):
            history[:, 'train_loss_best']

        valid_loss_best = history[:, 'valid_loss_best']
        expected = [True, False, True]
        assert valid_loss_best == expected
