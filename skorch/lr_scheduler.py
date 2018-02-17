"""Contains learning rate scheduler callbacks"""

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import Callback

def previous_epoch_train_loss_score(net):
    losses = net.history[-2, 'batches', :, 'train_loss']
    batch_sizes = net.history[-2, 'batches', :, 'train_batch_size']
    return np.average(losses, weights=batch_sizes)

__all__ = ['LRScheduler']

class LRScheduler(Callback):
    """
    Sets the learning rate of each parameter group according to some
    policy.
    """

    def __init__(self, policy='warm_restart', **kwargs):
        self.policy = policy
        self._lr_scheduler = None
        self._init_kwargs = kwargs

    def on_train_begin(self, net, **kwargs):
        self._lr_scheduler = self._get_scheduler(net, **self._init_kwargs)

    def on_epoch_begin(self, net, **kwargs):
        epoch = len(net.history)-1
        if self.policy == 'reduce_plateau':
            metrics = previous_epoch_train_loss_score(net) if epoch else np.inf
            self._lr_scheduler.step(metrics, epoch)
        else:
            self._lr_scheduler.step(epoch)

    def _get_scheduler(self, net, **kwargs):
        optimizer = net.optimizer_
        last_epoch = kwargs.get('last_epoch', -1)
        if self.policy == 'lambda':
            lr_lambda = kwargs.get('lambda', lambda x: 1e-1)
            return LambdaLR(optimizer, lr_lambda, last_epoch)

        if self.policy == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 1e-1)
            return StepLR(optimizer, step_size, gamma, last_epoch)

        if self.policy == 'multi_step':
            milestone = kwargs.get('milestone', [30, 90])
            gamma = kwargs.get('gamma', 1e-1)
            return MultiStepLR(optimizer, milestone, gamma, last_epoch)

        if self.policy == 'exponential':
            gamma = kwargs.get('gamma', 1e-1)
            return ExponentialLR(optimizer, gamma, last_epoch)

        if self.policy == 'reduce_plateau':
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 1e-1)
            patience = kwargs.get('patience', 10)
            verbose = kwargs.get('verbose', False)
            threshold = kwargs.get('threshold', 1e-4)
            threshold_mode = kwargs.get('threshold_mode', 'rel')
            cooldown = kwargs.get('cooldown', 0)
            min_lr = kwargs.get('min_lr', 0)
            eps = kwargs.get('eps', 1e-8)
            return ReduceLROnPlateau(
                optimizer, mode, factor, patience, verbose, threshold,
                threshold_mode, cooldown, min_lr, eps
            )

        if self.policy == 'warm_restart':
            min_lr = kwargs.get('min_lr', 1e-6)
            max_lr = kwargs.get('max_lr', 0.05)
            base_period = kwargs.get('base_period', 10)
            period_mult = kwargs.get('period_mult', 2)
            return WarmRestartLR(
                optimizer, min_lr, max_lr, base_period,
                period_mult, last_epoch
            )

class WarmRestartLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    stochastic gradient descent with warm restarts (SGDR) policy. The
    policy simulates periodic warm restarts of SGD, where in each restart the
    learning rate is initialize to some value and is scheduled to decrease,
    as described in the paper
    `"Stochastic Gradient Descent with Warm Restarts"
    <https://arxiv.org/pdf/1608.03983.pdf>`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        min_lr (float or list): minimum learning rate for each param groups.
        max_lr (float or list): maximum learning rate for each param groups.
        base_period (int): Initial interval between restarts.
        period_mult (int): Multiple factor to increase period between restarts.
        last_epoch (int): The index of the last epoch. Default: -1
    """

    def __init__(self, optimizer, min_lr=1e-6, max_lr=0.05, base_period=10,
        period_mult=2, last_epoch=-1):

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lr, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lr = np.array(min_lr)
        else:
            self.min_lr = min_lr * np.ones(len(optimizer.param_groups))

        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lr = np.array(max_lr)
        else:
            self.max_lr = max_lr * np.ones(len(optimizer.param_groups))
        self.base_period = base_period
        self.period_mult = period_mult
        super(WarmRestartLR, self).__init__(optimizer, last_epoch)

    def _get_current_lr(self, min_lr, max_lr, period, epoch):
        return min_lr + 0.5*(max_lr-min_lr)*(1+ np.cos(epoch * np.pi/period))

    def get_lr(self):
        epoch_idx = float(self.last_epoch)
        current_period = float(self.base_period)
        while epoch_idx / current_period > 1.0:
            epoch_idx -= current_period
            current_period *= self.period_mult
        current_lrs = self._get_current_lr(
            self.min_lr, self.max_lr, current_period, epoch_idx
        )
        return current_lrs.tolist()
