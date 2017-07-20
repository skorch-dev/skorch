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
        return avg_loss_cls()

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

    def test_init_ignore_keys(self, avg_loss_cls):
        avg_loss = avg_loss_cls(keys_possible=[
            ('train_loss', 'train_batch_size')])
        history = get_history(avg_loss)

        train_losses = history[:, 'train_loss']
        expected = [0.25, 0.65, -0.15]
        assert np.allclose(train_losses, expected)

        with pytest.raises(KeyError):
            history[:, 'valid_loss']
