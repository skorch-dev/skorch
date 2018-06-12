from functools import partial

import numpy as np
import pytest


class TestProgressBar:
    @pytest.yield_fixture
    def progressbar_cls(self):
        from skorch.callbacks import ProgressBar
        return ProgressBar

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 2 epochs"""
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
            max_epochs=2,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((20, 1), dtype='float32')
        y = np.zeros((20, 1), dtype='float32')
        return X, y

    @pytest.mark.parametrize('postfix', [
        [],
        ['train_loss'],
        ['train_loss', 'valid_loss'],
        ['doesnotexist'],
        ['train_loss', 'doesnotexist'],
    ])
    def test_invalid_postfix(self, postfix, net_cls, progressbar_cls, data):
        net = net_cls(callbacks=[
            progressbar_cls(postfix_keys=postfix),
        ])
        net.fit(*data)

    @pytest.mark.parametrize('scheme', [
        'count',
        'auto',
        None,
        2,  # correct number of batches_per_epoch (20 // 10)
        3,  # offset by +1, should still work
        1,  # offset by -1, should still work
    ])
    def test_different_count_schemes(
            self, scheme, net_cls, progressbar_cls, data):
        net = net_cls(callbacks=[
            progressbar_cls(batches_per_epoch=scheme),
        ])
        net.fit(*data)
