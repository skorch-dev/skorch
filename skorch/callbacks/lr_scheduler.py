"""Contains learning rate scheduler callbacks"""

import sys

# pylint: disable=unused-import
import warnings

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
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

    def simulate(self, steps, initial_lr, step_args=None):
        """
        Simulates the learning rate scheduler.

        Parameters
        ----------
        steps: int
          Number of steps to simulate

        initial_lr: float
          Initial learning rate

        step_args: None or float or List[float] (default=None)
          Argument to the ``.step()`` function of the policy. If it is an
          indexable object the simulation will try to associate every step of
          the simulation with an entry in ``step_args``. Scalar values are
          passed at every step, unchanged. In the default setting (``None``)
          no additional arguments are passed to ``.step()``.

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
        for step_idx in range(steps):
            opt.step()  # suppress warning about .step call order
            lrs.append(opt.param_groups[0]['lr'])
            if step_args is None:
                sch.step()
            elif hasattr(step_args, '__getitem__'):
                sch.step(step_args[step_idx])
            else:
                sch.step(step_args)

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

    def _step(self, net, lr_scheduler, score=None):
        """Helper method to step the lr scheduler.

        This takes care of two things:

        1. If the lr scheduler is ReduceLROnPlateau, we need to pass the score.
        2. If the net is uses AccelerateMixin, stepping has to be skipped in
           certain conditions.

        For more info on the latter, see:
        https://huggingface.co/docs/accelerate/quicktour#mixed-precision-training

        """
        accelerator_maybe = getattr(net, 'accelerator', None)
        accelerator_step_skipped = (
            accelerator_maybe and accelerator_maybe.optimizer_step_was_skipped
        )
        if accelerator_step_skipped:
            return

        if score is None:
            lr_scheduler.step()
        else:
            lr_scheduler.step(score)

    def _record_last_lr(self, net, kind):
        # helper function to record the last learning rate if possible;
        # only record the first lr returned if more than 1 param group
        if kind not in ('epoch', 'batch'):
            raise ValueError(f"Argument 'kind' should be 'batch' or 'epoch', get {kind}.")

        if (
                (self.event_name is None)
                or not hasattr(self.lr_scheduler_, 'get_last_lr')
        ):
            return

        try:
            last_lrs = self.lr_scheduler_.get_last_lr()
        except AttributeError:
            # get_last_lr fails for ReduceLROnPlateau with PyTorch <= 2.2 on 1st epoch.
            # Take the initial lr instead.
            last_lrs = [group['lr'] for group in net.optimizer_.param_groups]

        if kind == 'epoch':
            net.history.record(self.event_name, last_lrs[0])
        else:
            net.history.record_batch(self.event_name, last_lrs[0])

    def on_epoch_end(self, net, **kwargs):
        if self.step_every != 'epoch':
            return

        self._record_last_lr(net, kind='epoch')

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

            self._step(net, self.lr_scheduler_, score=score)
        else:
            self._step(net, self.lr_scheduler_)

    def on_batch_end(self, net, training, **kwargs):
        if not training or self.step_every != 'batch':
            return

        self._record_last_lr(net, kind='batch')

        if isinstance(self.lr_scheduler_, ReduceLROnPlateau):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, 'batches', -1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self._step(net, self.lr_scheduler_, score=score)
        else:
            self._step(net, self.lr_scheduler_)

        self.batch_idx_ += 1

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        """Return scheduler, based on indicated policy, with appropriate
        parameters.
        """
        if (
                (policy not in [ReduceLROnPlateau])
                and ('last_epoch' not in scheduler_kwargs)
        ):
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
