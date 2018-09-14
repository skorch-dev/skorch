"""Tests for lr_scheduler.py"""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import clone
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

from skorch import NeuralNetClassifier
from skorch.callbacks.lr_scheduler import WarmRestartLR, LRScheduler, CyclicLR


class TestLRCallbacks:

    @pytest.mark.parametrize('policy', [StepLR, 'StepLR'])
    def test_simulate_lrs_epoch_step(self, policy):
        lr_sch = LRScheduler(policy, step_size=2)
        lrs = lr_sch.simulate(6, 1)
        expected = np.array([1.0, 1.0, 0.1, 0.1, 0.01, 0.01])
        assert np.allclose(expected, lrs)

    @pytest.mark.parametrize('policy', [CyclicLR, 'CyclicLR'])
    def test_simulate_lrs_batch_step(self, policy):
        lr_sch = LRScheduler(
            policy, base_lr=1, max_lr=5, step_size_up=4)
        lrs = lr_sch.simulate(11, 1)
        expected = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3])
        assert np.allclose(expected, lrs)

    @pytest.mark.parametrize('policy, instance, kwargs', [
        ('LambdaLR', LambdaLR, {'lr_lambda': (lambda x: 1e-1)}),
        ('StepLR', StepLR, {'step_size': 30}),
        ('MultiStepLR', MultiStepLR, {'milestones': [30, 90]}),
        ('ExponentialLR', ExponentialLR, {'gamma': 0.1}),
        ('ReduceLROnPlateau', ReduceLROnPlateau, {}),
        ('WarmRestartLR', WarmRestartLR, {}),
        ('CyclicLR', CyclicLR, {}),
        ('CosineAnnealingLR', CosineAnnealingLR, {'T_max': 5, 'eta_min': 1e-3}),
        (WarmRestartLR, WarmRestartLR, {}),
    ])
    def test_lr_callback_init_policies(
            self,
            classifier_module,
            classifier_data,
            policy,
            instance,
            kwargs,
    ):
        X, y = classifier_data
        lr_policy = LRScheduler(policy, **kwargs)
        net = NeuralNetClassifier(
            classifier_module, max_epochs=2, callbacks=[lr_policy]
        )
        net.fit(X, y)
        assert any(list(map(
            lambda x: isinstance(
                getattr(x[1], 'lr_scheduler_', None), instance),
            net.callbacks_
        )))

    @pytest.mark.parametrize('policy, kwargs', [
        ('LambdaLR', {'lr_lambda': (lambda x: 1e-1)}),
        ('StepLR', {'step_size': 30}),
        ('MultiStepLR', {'milestones': [30, 90]}),
        ('ExponentialLR', {'gamma': 0.1}),
        ('ReduceLROnPlateau', {}),
        ('WarmRestartLR', {}),
        ('CosineAnnealingLR', {'T_max': 3}),
    ])
    def test_lr_callback_steps_correctly(
            self,
            classifier_module,
            classifier_data,
            policy,
            kwargs,
    ):
        max_epochs = 2
        X, y = classifier_data
        lr_policy = LRScheduler(policy, **kwargs)
        net = NeuralNetClassifier(
            classifier_module(),
            max_epochs=max_epochs,
            batch_size=16,
            callbacks=[lr_policy],
        )
        net.fit(X, y)
        # pylint: disable=protected-access
        assert lr_policy.lr_scheduler_.last_epoch == max_epochs - 1

    @pytest.mark.parametrize('policy, kwargs', [
        ('CyclicLR', {}),
    ])
    def test_lr_callback_batch_steps_correctly(
            self,
            classifier_module,
            classifier_data,
            policy,
            kwargs,
    ):
        num_examples = 1000
        batch_size = 100
        max_epochs = 2

        X, y = classifier_data
        lr_policy = LRScheduler(policy, **kwargs)
        net = NeuralNetClassifier(classifier_module(), max_epochs=max_epochs,
                                  batch_size=batch_size, callbacks=[lr_policy])
        net.fit(X, y)
        expected = (num_examples // batch_size) * max_epochs - 1
        # pylint: disable=protected-access
        assert lr_policy.lr_scheduler_.last_batch_idx == expected

    def test_lr_scheduler_cloneable(self):
        # reproduces bug #271
        scheduler = LRScheduler(CyclicLR, base_lr=123)
        clone(scheduler)  # does not raise

    def test_lr_scheduler_set_params(self, classifier_module, classifier_data):
        scheduler = LRScheduler(CyclicLR, base_lr=123)
        net = NeuralNetClassifier(
            classifier_module,
            max_epochs=0,
            callbacks=[('scheduler', scheduler)],
        )
        net.set_params(callbacks__scheduler__base_lr=456)
        net.fit(*classifier_data)  # we need to trigger on_train_begin
        assert net.callbacks[0][1].lr_scheduler_.base_lrs[0] == 456


class TestReduceLROnPlateau:

    def get_net_with_mock(
            self, classifier_data, classifier_module, monitor='train_loss'):
        """Returns a net with a mocked lr policy that allows to check what
        it's step method was called with.

        """
        X, y = classifier_data
        net = NeuralNetClassifier(
            classifier_module,
            callbacks=[
                ('scheduler', LRScheduler(ReduceLROnPlateau, monitor=monitor)),
            ],
            max_epochs=1,
        ).fit(X, y)

        # mock the policy
        policy = dict(net.callbacks_)['scheduler'].lr_scheduler_
        mock_step = Mock(side_effect=policy.step)
        policy.step = mock_step

        # make sure that mocked policy is set
        scheduler = dict(net.callbacks_)['scheduler']
        # pylint: disable=protected-access
        scheduler._get_scheduler = lambda *args, **kwargs: policy

        net.partial_fit(X, y)
        return net, mock_step

    @pytest.mark.parametrize('monitor', ['train_loss', 'valid_loss', 'epoch'])
    def test_reduce_lr_monitor_with_string(
            self, monitor, classifier_data, classifier_module):
        # step should be called with the 2nd to last value from that
        # history entry
        net, mock_step = self.get_net_with_mock(
            classifier_data, classifier_module, monitor=monitor)
        score = mock_step.call_args_list[0][0][0]
        np.isclose(score, net.history[-2, monitor])

    def test_reduce_lr_monitor_with_callable(
            self, classifier_data, classifier_module):
        # step should always be called with the return value from the
        # callable, 55
        _, mock_step = self.get_net_with_mock(
            classifier_data, classifier_module, monitor=lambda x: 55)
        score = mock_step.call_args_list[0][0][0]
        assert score == 55


class TestWarmRestartLR():
    def assert_lr_correct(
            self, optimizer, targets, epochs, min_lr, max_lr, base_period,
            period_mult):
        """Test that learning rate was set correctly."""
        targets = [targets] if len(optimizer.param_groups) == 1 else targets
        scheduler = WarmRestartLR(
            optimizer, min_lr, max_lr, base_period, period_mult
        )
        for epoch in range(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(optimizer.param_groups, targets):
                assert param_group['lr'] == pytest.approx(target[epoch])

    def _single_period_targets(self, epochs, min_lr, max_lr, period):
        targets = 1 + np.cos(np.arange(epochs) * np.pi / period)
        targets = min_lr + 0.5 * (max_lr-min_lr) * targets
        return targets.tolist()

    # pylint: disable=missing-docstring
    def _multi_period_targets(
            self, epochs, min_lr, max_lr, base_period, period_mult):
        remaining_epochs = epochs
        current_period = base_period
        targets = list()
        while remaining_epochs > 0:
            period_epochs = min(remaining_epochs, current_period+1)
            remaining_epochs -= period_epochs
            targets += self._single_period_targets(
                period_epochs, min_lr, max_lr, current_period
            )
            current_period = current_period * period_mult
        return targets

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
        targets = self._single_period_targets(
            epochs, min_lr, max_lr, base_period)
        self.assert_lr_correct(
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
        targets = self._multi_period_targets(
            epochs, min_lr, max_lr, base_period, period_mult
        )
        self.assert_lr_correct(
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
                {'params': classifier.sequential[0].parameters(), 'lr': 1e-3},
                {'params': classifier.sequential[1].parameters(), 'lr': 1e-2},
                {'params': classifier.sequential[2].parameters(), 'lr': 1e-1},
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
                self._multi_period_targets(
                    epochs, min_lr, max_lr, base_period, period_mult
                )
            )
        self.assert_lr_correct(
            optimizer,
            targets,
            epochs,
            min_lr_group,
            max_lr_group,
            base_period,
            period_mult
        )


class TestCyclicLR():

    @pytest.fixture(params=[1, 3])
    def init_optimizer(self, classifier_module, request):
        if request.param == 1:
            return SGD(classifier_module().parameters(), lr=0.05)
        classifier = classifier_module()
        return SGD(
            [
                {'params': classifier.sequential[0].parameters(), 'lr': 1e-3},
                {'params': classifier.sequential[1].parameters(), 'lr': 1e-2},
                {'params': classifier.sequential[2].parameters(), 'lr': 1e-1},
            ]
        )

    @pytest.fixture
    def num_groups(self, init_optimizer):
        return len(init_optimizer.param_groups)

    def test_invalid_number_of_base_lr(self, init_optimizer):
        with pytest.raises(ValueError):
            CyclicLR(init_optimizer, base_lr=[1, 2])

    def test_invalid_number_of_max_lr(self, init_optimizer):
        with pytest.raises(ValueError):
            CyclicLR(init_optimizer, max_lr=[1, 2])

    def test_invalid_mode(self, init_optimizer):
        with pytest.raises(ValueError):
            CyclicLR(init_optimizer, mode='badmode')

    def test_invalid_not_a_optimizer(self):
        with pytest.raises(TypeError):
            CyclicLR('this is a string')

    def test_triangular_mode(self, init_optimizer, num_groups):
        target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        targets = [target] * num_groups
        scheduler = CyclicLR(init_optimizer, base_lr=1, max_lr=5, step_size_up=4,
                             mode='triangular')
        self._test_cycle_lr(init_optimizer, scheduler, targets)

    def test_triangular_mode_step_size_up_down(self, init_optimizer, num_groups):
        target = [1, 2, 3, 4, 5, 13/3, 11/3, 9/3, 7/3, 5/3, 1]
        targets = [target] * num_groups
        scheduler = CyclicLR(init_optimizer, base_lr=1, max_lr=5,
                             step_size_up=4,
                             step_size_down=6,
                             mode='triangular')
        self._test_cycle_lr(init_optimizer, scheduler, targets)

    def test_triangular2_mode(self, init_optimizer, num_groups):
        base_target = ([1, 2, 3, 4, 5, 4, 3, 2, 1,
                        1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1,
                        1.25, 1.50, 1.75, 2.00, 1.75])
        deltas = [2*i for i in range(0, num_groups)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        targets = [[x + delta for x in base_target] for delta in deltas]
        scheduler = CyclicLR(init_optimizer, base_lr=base_lrs, max_lr=max_lrs,
                             step_size_up=4, mode='triangular2')
        self._test_cycle_lr(init_optimizer, scheduler, targets)

    def test_triangular2_mode_step_size_up_down(self, init_optimizer, num_groups):
        base_target = ([1, 3, 5, 13/3, 11/3, 9/3, 7/3, 5/3,
                        1, 2, 3, 8/3, 7/3, 6/3, 5/3, 4/3,
                        1, 3/2, 2, 11/6, 10/6, 9/6, 8/6, 7/6])
        deltas = [2*i for i in range(0, num_groups)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        targets = [[x + delta for x in base_target] for delta in deltas]
        scheduler = CyclicLR(init_optimizer, base_lr=base_lrs, max_lr=max_lrs,
                             step_size_up=2, step_size_down=6,
                             mode='triangular2')
        self._test_cycle_lr(init_optimizer, scheduler, targets)

    def test_exp_range_mode(self, init_optimizer, num_groups):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = ([0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25,
               0, 0.25, 0.5, 0.75, 1])
        target = [base_lr + x*diff_lr*gamma**i for i, x in enumerate(xs)]
        targets = [target] * num_groups
        scheduler = CyclicLR(init_optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size_up=4, mode='exp_range', gamma=gamma)
        self._test_cycle_lr(init_optimizer, scheduler, targets)

    def test_exp_range_mode_step_size_up_down(self, init_optimizer, num_groups):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = ([0, 0.5, 1, 5/6, 4/6, 3/6, 2/6, 1/6, 0, 0.5, 1, 5/6, 4/6])
        target = [base_lr + x*diff_lr*gamma**i for i, x in enumerate(xs)]
        targets = [target] * num_groups
        scheduler = CyclicLR(init_optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size_up=2, step_size_down=6,
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(init_optimizer, scheduler, targets)

    def test_batch_idx_with_none(self, init_optimizer):
        scheduler = CyclicLR(init_optimizer)
        scheduler.batch_step()
        assert scheduler.last_batch_idx == 0

    def test_scale_fn(self, init_optimizer):
        def scale_fn(x):
            return x
        scheduler = CyclicLR(init_optimizer, scale_fn=scale_fn)
        assert scheduler.scale_fn == scale_fn

    @staticmethod
    def _test_cycle_lr(optimizer, scheduler, targets):
        batch_idxs = len(targets[0])
        for batch_num in range(batch_idxs):
            scheduler.batch_step(batch_num)
            for param_group, target in zip(optimizer.param_groups, targets):
                assert param_group['lr'] == pytest.approx(target[batch_num])
