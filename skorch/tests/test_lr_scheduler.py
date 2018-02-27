"""Tests for lr_scheduler.py"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from skorch.net import NeuralNetClassifier
from skorch.lr_scheduler import WarmRestartLR, LRScheduler

class TestLRCallbacks:

    def test_lr_callback_init_policies(self, classifier_module):
        lr_scheduler_pack = [
            ('LambdaLR', LambdaLR, {'lr_lambda': (lambda x: 1e-1)}),
            ('StepLR', StepLR, {'step_size': 30}),
            ('MultiStepLR', MultiStepLR, {'milestones': [30, 90]}),
            ('ExponentialLR', ExponentialLR, {'gamma': 0.1}),
            ('ReduceLROnPlateau', ReduceLROnPlateau, {}),
            ('WarmRestartLR', WarmRestartLR, {}),
            (WarmRestartLR, WarmRestartLR, {}),
        ]
        for policy, instance, kwargs in lr_scheduler_pack:
            self._lr_callback_init_policies(
                classifier_module(), policy, instance, **kwargs
            )

    def test_raise_invalid_policy_string(self):
        with pytest.raises(AttributeError):
            LRScheduler("invalid_policy")

    def test_raise_invalid_policy_class(self):
        class DummyClass():
            pass

        with pytest.raises(AssertionError):
            LRScheduler(DummyClass)

    def _lr_callback_init_policies(
            self,
            classifier_module,
            policy, instance,
            **kwargs
        ):
        X, y = make_classification(
            1000, 20, n_informative=10, random_state=0
        )
        X = X.astype(np.float32)
        lr_policy = LRScheduler(policy, **kwargs)
        net = NeuralNetClassifier(
            classifier_module, max_epochs=2, callbacks=[lr_policy]
        )
        net.fit(X, y)
        assert any(list(map(
            lambda x: isinstance(
                getattr(x[1], '_lr_scheduler', None), instance),
            net.callbacks_
        )))

class TestWarmRestartLR():

    @pytest.fixture()
    def init_optimizer(self, classifier_module):
        return SGD(classifier_module().parameters(), lr=0.05)

    def test_raise_incompatible_len_on_min_lr_err(self, init_optimizer):
        with pytest.raises(ValueError) as excinfo:
            WarmRestartLR(init_optimizer, min_lr=[1e-1, 1e-2])
        assert 'min_lr' in str(excinfo.value)

    def test_raise_incompatible_len_on_max_lr_err(self, init_optimizer):
        with pytest.raises(ValueError) as excinfo:
            WarmRestartLR(init_optimizer, max_lr=[1e-1, 1e-2])
        assert 'max_lr' in str(excinfo.value)

    def test_single_period(self, init_optimizer):
        optimizer = init_optimizer
        epochs = 3
        min_lr = 5e-5
        max_lr = 5e-2
        base_period = 3
        period_mult = 1
        targets = _single_period_targets(epochs, min_lr, max_lr, base_period)
        _test(
            optimizer,
            targets,
            epochs,
            min_lr,
            max_lr,
            base_period,
            period_mult
        )

    def test_multi_period_with_restart(self, init_optimizer):
        optimizer = init_optimizer
        epochs = 9
        min_lr = 5e-5
        max_lr = 5e-2
        base_period = 2
        period_mult = 2
        targets = _multi_period_targets(
            epochs, min_lr, max_lr, base_period, period_mult
        )
        _test(
            optimizer,
            targets,
            epochs,
            min_lr,
            max_lr,
            base_period,
            period_mult
        )

    def test_restarts_with_multiple_groups(self, classifier_module):
        classifier = classifier_module()
        optimizer = SGD(
            [
                {'params': classifier.dense0.parameters(), 'lr': 1e-3},
                {'params': classifier.dense1.parameters(), 'lr': 1e-2},
                {'params': classifier.output.parameters(), 'lr': 1e-1},
            ]
        )

        epochs = 9
        min_lr_group = [1e-5, 1e-4, 1e-3]
        max_lr_group = [1e-3, 1e-2, 1e-1]
        base_period = 2
        period_mult = 2
        targets = list()
        for min_lr, max_lr in zip(min_lr_group, max_lr_group):
            targets.append(
                _multi_period_targets(
                    epochs, min_lr, max_lr, base_period, period_mult
                )
            )
        _test(
            optimizer,
            targets,
            epochs,
            min_lr_group,
            max_lr_group,
            base_period,
            period_mult
        )

def _test(optimizer, targets, epochs, min_lr, max_lr, base_period, period_mult):
    targets = [targets] if len(optimizer.param_groups) == 1 else targets
    scheduler = WarmRestartLR(
        optimizer, min_lr, max_lr, base_period, period_mult
    )
    for epoch in range(epochs):
        scheduler.step(epoch)
        for param_group, target in zip(optimizer.param_groups, targets):
            assert param_group['lr'] == pytest.approx(target[epoch])

def _single_period_targets(epochs, min_lr, max_lr, period):
    targets = 1 + np.cos(np.arange(epochs) * np.pi / period)
    targets = min_lr + 0.5 * (max_lr-min_lr) * targets
    return targets.tolist()

def _multi_period_targets(epochs, min_lr, max_lr, base_period, period_mult):
    remaining_epochs = epochs
    current_period = base_period
    targets = list()
    while remaining_epochs > 0:
        period_epochs = min(remaining_epochs, current_period+1)
        remaining_epochs -= period_epochs
        targets += _single_period_targets(
            period_epochs, min_lr, max_lr, current_period
        )
        current_period = current_period * period_mult
    return targets
