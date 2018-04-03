"""Contains learning rate scheduler callbacks"""

import sys
import numpy as np

# pylint: disable=unused-import
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import Callback


__all__ = ['LRScheduler']


def previous_epoch_train_loss_score(net):
    losses = net.history[-2, 'batches', :, 'train_loss']
    batch_sizes = net.history[-2, 'batches', :, 'train_batch_size']
    return np.average(losses, weights=batch_sizes)


class LRScheduler(Callback):
    """Callback that sets the learning rate of each
    parameter group according to some policy.

    Parameters
    ----------

    policy : str or _LRScheduler class (default='WarmRestartLR')
      Learning rate policy name or scheduler to be used.

    """

    def __init__(self, policy="WarmRestartLR", **kwargs):
        if isinstance(policy, str):
            self.policy = getattr(sys.modules[__name__], policy)
        else:
            assert issubclass(policy, _LRScheduler)
            self.policy = policy
        self.kwargs = kwargs
        self._lr_scheduler = None

    def on_train_begin(self, net, **kwargs):
        self._lr_scheduler = self._get_scheduler(
            net, self.policy, **self.kwargs
        )

    def on_epoch_begin(self, net, **kwargs):
        epoch = len(net.history)-1
        if isinstance(self._lr_scheduler, ReduceLROnPlateau):
            metrics = previous_epoch_train_loss_score(net) if epoch else np.inf
            self._lr_scheduler.step(metrics, epoch)
        else:
            self._lr_scheduler.step(epoch)

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        return policy(net.optimizer_, **scheduler_kwargs)


class WarmRestartLR(_LRScheduler):
    """Stochastic Gradient Descent with Warm Restarts (SGDR) scheduler.

    This scheduler sets the learning rate of each parameter group
    according to stochastic gradient descent with warm restarts (SGDR)
    policy [1]. This policy simulates periodic warm restarts of SGD, where
    in each restart the learning rate is initialize to some value and is
    scheduled to decrease.

    Parameters
    ----------
    optimizer : torch.optimizer.Optimizer instance.
      Optimizer algorithm.

    min_lr : float or list of float (default=1e-6)
      Minimum allowed learning rate during each period for all
      param groups (float) or each group (list).

    max_lr : float or list of float (default=0.05)
      Maximum allowed learning rate during each period for all
      param groups (float) or each group (list).

    base_period : int (default=10)
      Initial restart period to be multiplied at each restart.

    period_mult : int (default=2)
      Multiplicative factor to increase the period between restarts.

    last_epoch : int (default=-1)
      The index of the last valid epoch.

    References
    ----------
    ..[1] Ilya Loshchilov and Frank Hutter, 2017, "Stochastic Gradient
          Descent with Warm Restarts,". "ICLR"
        `<https://arxiv.org/pdf/1608.03983.pdf>`_

    """

    def __init__(
            self, optimizer,
            min_lr=1e-6,
            max_lr=0.05,
            base_period=10,
            period_mult=2,
            last_epoch=-1
        ):
        self.min_lr = self._format_lr('min_lr', optimizer, min_lr)
        self.max_lr = self._format_lr('max_lr', optimizer, max_lr)
        self.base_period = base_period
        self.period_mult = period_mult
        super(WarmRestartLR, self).__init__(optimizer, last_epoch)

    def _format_lr(self, name, optimizer, lr):
        """Return correctly formatted lr for each param group."""
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(lr)))
            return np.array(lr)
        else:
            return lr * np.ones(len(optimizer.param_groups))

    def _get_current_lr(self, min_lr, max_lr, period, epoch):
        return min_lr + 0.5*(max_lr-min_lr)*(1+ np.cos(epoch * np.pi/period))

    def get_lr(self):
        epoch_idx = float(self.last_epoch)
        current_period = float(self.base_period)
        while epoch_idx / current_period > 1.0:
            epoch_idx -= current_period + 1
            current_period *= self.period_mult

        current_lrs = self._get_current_lr(
            self.min_lr,
            self.max_lr,
            current_period,
            epoch_idx
        )
        return current_lrs.tolist()
