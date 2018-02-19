import pytest
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from skorch.net import NeuralNetClassifier
from sklearn.datasets import make_classification
from skorch.lr_scheduler import WarmRestartLR, LRScheduler

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
        lr_scheduler_pack = [
            ('LambdaLR', LambdaLR, {'lr_lambda': (lambda x: 1e-1)}),
            ('StepLR', StepLR, {'step_size': 30}),
            ('MultiStepLR', MultiStepLR, {'milestones': [30, 90]}),
            ('ExponentialLR', ExponentialLR, {'gamma': 0.1}),
            ('ReduceLROnPlateau', ReduceLROnPlateau, {}),
            ('WarmRestartLR', WarmRestartLR, {}),
        ]
        for policy, instance, kwargs in lr_scheduler_pack:
            self._lr_callback_init_policies(policy, instance, **kwargs)

    def _lr_callback_init_policies(self, policy, instance, **kwargs):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        lr_policy = LRScheduler(policy, **kwargs)
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
        epochs = 10
        min_lr = 5e-5
        max_lr = 5e-2
        base_period = 10
        period_mult = 2
        single_targets = min_lr + 0.5 * (max_lr-min_lr) * (
            1 + np.cos(np.arange(epochs) * np.pi / epochs))
        single_targets = single_targets.tolist()
        targets = [single_targets, list(map(
            lambda x: x * epochs, single_targets))]
        scheduler = WarmRestartLR(
            self.opt, min_lr, max_lr, base_period, period_mult
        )
        self._test(scheduler, targets, epochs)

    def _test(self, scheduler, targets, epochs=10):
        for epoch in range(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                assert param_group['lr'] == pytest.approx(target[epoch])

    def test_raise_incompatible_len_on_min_lr_err(self):
        with pytest.raises(ValueError) as excinfo:
            WarmRestartLR(self.opt, min_lr=[1e-1, 1e-2, 1e-3])
        assert 'min_lr' in str(excinfo.value)

    def test_raise_incompatible_len_on_max_lr_err(self):
        with pytest.raises(ValueError) as excinfo:
            WarmRestartLR(self.opt, max_lr=[1e-1, 1e-2, 1e-3])
        assert 'max_lr' in str(excinfo.value)
