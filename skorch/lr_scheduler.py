import numpy as np
from torch.optim.lr_scheduler import *
from skorch.callbacks import Callback

__all__ = ['LRScheduler']

class LRScheduler(Callback):
    """
    Sets the learning rate of each parameter group according to some
    policy.
    """

    def initialize(self, policy='warm_restart'):
        self.policy = policy
        self._lr_scheduler = None

    def on_train_begin(self, net, **kwargs):
        self._lr_scheduler = self._get_scheduler(net, **kwargs)

    def on_epoch_begin(self, net, **kwargs):
        self._lr_scheduler.step()

    def _get_scheduler(self, net, **kwargs):
        optimizer = net.optimizer_
        if self.policy == 'lambda':
            return LambdaLR(optimizer=optimizer, **kwargs)
        if self.policy == 'step':
            return StepLR(optimizer=optimizer, **kwargs)
        if self.policy == 'multi_step':
            return MultiStepLR(optimizer=optimizer, **kwargs)
        if self.policy == 'exponential':
            return ExponentialLR(optimizer=optimizer, **kwargs)
        if self.policy == 'cosine_annealing':
            return CosineAnnealingLR(optimizer=optimizer, **kwargs)
        if self.policy == 'reduce_plateau':
            return ReduceLROnPlateau(optimizer=optimizer, **kwargs)
        if self.policy == 'warm_restart':
            return WarmRestartLR(optimizer=optimizer, **kwargs)

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

    def __init__(self, optimizer, min_lr=0.0, max_lr=0.05, base_period=10,
        period_mult=2, last_epoch=-1):

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lr, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lr = np.array(min_lr)
        else:
            self.min_lr = min_lr * np.ones(len(optimizer.param_groups))

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
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
        current_lrs = self._get_current_lr(self.min_lr, self.max_lr,
            epoch_idx, current_period)
        return current_lrs.tolist()
