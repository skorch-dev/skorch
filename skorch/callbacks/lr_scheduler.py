"""Contains learning rate scheduler callbacks"""

import sys

# pylint: disable=unused-import
import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

try:
    from torch.optim.lr_scheduler import CyclicLR as TorchCyclicLR
except ImportError:
    # Backward compatibility with torch >= 1.0 && < 1.1
    TorchCyclicLR = None
from torch.optim.optimizer import Optimizer
from skorch.callbacks import Callback

__all__ = ['LRScheduler', 'WarmRestartLR']


def _check_lr(name, optimizer, lr):
    """Return one learning rate for each param group."""
    n = len(optimizer.param_groups)
    if not isinstance(lr, (list, tuple)):
        return lr * np.ones(n)

    if len(lr) != n:
        raise ValueError("{} lr values were passed for {} but there are "
                         "{} param groups.".format(n, name, len(lr)))
    return np.array(lr)


class LRScheduler(Callback):
    """Callback that sets the learning rate of each
    parameter group according to some policy.

    Parameters
    ----------

    policy : str or _LRScheduler class (default='WarmRestartLR')
      Learning rate policy name or scheduler to be used.

    monitor : str or callable (default=None)
      Value of the history to monitor or function/callable. In
      the latter case, the callable receives the net instance as
      argument and is expected to return the score (float) used to
      determine the learning rate adjustment.

    event_name: str, (default='event_lr')
      Name of event to be placed in history when the scheduler takes a step.
      Pass ``None`` to disable placing events in history.
      **Note:** This feature works only for pytorch version >=1.4

    step_every: str, (default='epoch')
      Value for when to apply the learning scheduler step. Can be either 'batch'
       or 'epoch'.

    kwargs
      Additional arguments passed to the lr scheduler.

    """

    def __init__(self,
                 policy='WarmRestartLR',
                 monitor='train_loss',
                 event_name="event_lr",
                 step_every='epoch',
                 **kwargs):
        self.policy = policy
        self.monitor = monitor
        self.event_name = event_name
        self.step_every = step_every
        vars(self).update(kwargs)

    def simulate(self, steps, initial_lr):
        """
        Simulates the learning rate scheduler.

        Parameters
        ----------
        steps: int
          Number of steps to simulate

        initial_lr: float
          Initial learning rate

        Returns
        -------
        lrs: numpy ndarray
          Simulated learning rates

        """
        test = torch.ones(1, requires_grad=True)
        opt = torch.optim.SGD([{'params': test, 'lr': initial_lr}])
        policy_cls = self._get_policy_cls()
        sch = policy_cls(opt, **self.kwargs)

        lrs = []
        for _ in range(steps):
            opt.step()  # suppress warning about .step call order
            lrs.append(opt.param_groups[0]['lr'])
            sch.step()

        return np.array(lrs)

    def initialize(self):
        self.policy_ = self._get_policy_cls()
        self.lr_scheduler_ = None
        self.batch_idx_ = 0
        return self

    def _get_policy_cls(self):
        if isinstance(self.policy, str):
            return getattr(sys.modules[__name__], self.policy)
        return self.policy

    @property
    def kwargs(self):
        # These are the parameters that are passed to the
        # scheduler. Parameters that don't belong there must be
        # excluded.
        excluded = ('policy', 'monitor', 'event_name', 'step_every')
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def on_train_begin(self, net, **kwargs):
        if net.history:
            try:
                self.batch_idx_ = sum(net.history[:, 'train_batch_count'])
            except KeyError:
                self.batch_idx_ = sum(len(b) for b in net.history[:, 'batches'])
        self.lr_scheduler_ = self._get_scheduler(
            net, self.policy_, **self.kwargs
        )

    def on_epoch_end(self, net, **kwargs):
        if self.step_every != 'epoch':
            return
        if isinstance(self.lr_scheduler_, ReduceLROnPlateau):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self.lr_scheduler_.step(score)
            # ReduceLROnPlateau does not expose the current lr so it can't be recorded
        else:
            if self.event_name is not None and hasattr(
                    self.lr_scheduler_, "get_last_lr"):
                net.history.record(self.event_name,
                                   self.lr_scheduler_.get_last_lr()[0])
            self.lr_scheduler_.step()

    def on_batch_end(self, net, training, **kwargs):
        if not training or self.step_every != 'batch':
            return
        if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"):
            net.history.record_batch(self.event_name,
                                     self.lr_scheduler_.get_last_lr()[0])
        self.lr_scheduler_.step()
        self.batch_idx_ += 1

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        """Return scheduler, based on indicated policy, with appropriate
        parameters.
        """
        if policy not in [ReduceLROnPlateau] and \
                'last_epoch' not in scheduler_kwargs:
            last_epoch = len(net.history) - 1
            scheduler_kwargs['last_epoch'] = last_epoch

        return policy(net.optimizer_, **scheduler_kwargs)


class WarmRestartLR(_LRScheduler):
    """Stochastic Gradient Descent with Warm Restarts (SGDR) scheduler.

    This scheduler sets the learning rate of each parameter group
    according to stochastic gradient descent with warm restarts (SGDR)
    policy. This policy simulates periodic warm restarts of SGD, where
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
    .. [1] Ilya Loshchilov and Frank Hutter, 2017, "Stochastic Gradient
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
        self.min_lr = _check_lr('min_lr', optimizer, min_lr)
        self.max_lr = _check_lr('max_lr', optimizer, max_lr)
        self.base_period = base_period
        self.period_mult = period_mult
        super(WarmRestartLR, self).__init__(optimizer, last_epoch)

    def _get_current_lr(self, min_lr, max_lr, period, epoch):
        return min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(epoch * np.pi / period))

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
