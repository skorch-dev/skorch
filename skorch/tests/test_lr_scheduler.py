from unittest.mock import Mock
from unittest.mock import patch

from torch.optim import SGD
import torch

from skorch.lr_scheduler import WarmRestartLR

import numpy as np
import pytest

#TODO: Copying from pytorch test/test_optim. Should be working
class SchedulerTestNet(torch.nn.Module):
    def __init__(self):
        super(SchedulerTestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

class TestWarmRestartLR():
    def setup_class(self):
        self.net = SchedulerTestNet()
        self.opt = SGD(
            [{'params': self.net.conv1.parameters()},
                {'params': self.net.conv2.parameters(), 'lr': 0.5}],
            lr = 0.05
        )

    def test_warm_restart_lr(self):
        epochs = 12
        min_lr = 5e-5
        max_lr = 5e-2
        base_period = 4
        period_mult = 2
        single_targets = np.array([base_period] * base_period
            + [base_period * period_mult] * base_period * period_mult)
        single_targets = min_lr + 0.5 * (max_lr-min_lr) * (1 + np.cos(
            np.arange(epochs) * np.pi / single_targets))
        single_targets = single_targets.tolist()
        targets = [single_targets, list(map(
            lambda x: x * epochs, single_targets))]
        scheduler = WarmRestartLR(
            self.opt, min_lr, max_lr, base_period, period_mult
        )

    def _test(self, scheduler, targets, epochs=10):
        for epoch in range(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                 self.assertAlmostEqual(
                    target[epoch], param_group['lr'],
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, target[epoch], param_group['lr']), delta=1e-5
                    )
