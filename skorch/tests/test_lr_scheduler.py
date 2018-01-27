from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD
import torch

from skorch.lr_scheduler import WarmRestartLR, LRScheduler
from sklearn.datasets import make_classification
from skorch.net import NeuralNetClassifier
import numpy as np
import pytest

#TODO: Copying from pytorch test/test_optim. Should be working
class SchedulerTestNet(torch.nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(SchedulerTestNet, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X

class TestLRCallbacks:

    def test_lr_callback_init_policies(self):
        lr_scheduler_types = {
            'lambda': LambdaLR,
            'step': StepLR,
            'multi_step': MultiStepLR,
            'exponential': ExponentialLR,
            'reduce_plateau': ReduceLROnPlateau,
            'warm_restart': WarmRestartLR,
        }
        for policy, instance in lr_scheduler_types.items():
            self._lr_callback_init_policies(policy, instance)

    def _lr_callback_init_policies(self, policy, instance):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        lr_policy = LRScheduler(policy)
        net = NeuralNetClassifier(
            SchedulerTestNet,  max_epochs=1, callbacks=[lr_policy]
        )
        net.fit(X, y)
        assert any(list(map(
            lambda x: isinstance(
                getattr(x[1], '_lr_scheduler', None), instance),
            net.callbacks_
        )))

class TestWarmRestartLR():
    def setup_class(self):
        self.net = SchedulerTestNet()
        self.opt = SGD(self.net.parameters(), lr = 0.05)

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
