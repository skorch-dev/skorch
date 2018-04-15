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
from torch.optim.optimizer import Optimizer
from skorch.callbacks import Callback


__all__ = ['LRScheduler', 'WarmRestartLR', 'CyclicLR']


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

    def on_batch_begin(self, net, **kwargs):
        if isinstance(self._lr_scheduler, CyclicLR):
            epoch = len(net.history) - 1
            current_batch_idx = len(net.history[-1, 'batches'])
            batch_cnt = len(net.history[-2, 'batches']) if epoch >= 1 else 0
            batch_iteration = epoch * batch_cnt + current_batch_idx
            self._lr_scheduler.batch_step(batch_iteration)

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
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

class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate
    `CycleLR`.

    This class has three built-in policies, as put forth in the paper:

    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each
        cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations)
        at each cycle iteration.

    This implementation was adapted from the github repo:
    `bckenstler/CLR <https://github.com/bckenstler/CLR>`_

    Parameters
    ----------
    optimizer : torch.optimizer.Optimizer instance.
      Optimizer algorithm.

    base_lr : float or list of float (default=1e-3)
      Initial learning rate which is the lower boundary in the
      cycle for each param groups (float) or each group (list).

    max_lr : float or list of float (default=6e-3)
      Upper boundaries in the cycle for each parameter group (float)
      or each group (list). Functionally, it defines the cycle
      amplitude (max_lr - base_lr). The lr at any cycle is the sum
      of base_lr and some scaling of the amplitude; therefore max_lr
      may not actually be reached depending on scaling function.

    step_size : int (default=2000)
      Number of training iterations per half cycle. Authors suggest
      setting step_size 2-8 x training iterations in epoch.

    mode : str (default='triangular')
      One of {triangular, triangular2, exp_range}. Values correspond
      to policies detailed above. If scale_fn is not None, this
      argument is ignored.

    gamma : float (default=1.0)
      Constant in 'exp_range' scaling function:
      gamma**(cycle iterations)

    scale_fn : function (default=None)
      Custom scaling policy defined by a single argument lambda
      function, where 0 <= scale_fn(x) <= 1 for all x >= 0.
      mode paramater is ignored.

    scale_mode : str (default='cycle')
      One of {'cycle', 'iterations'}. Defines whether scale_fn
      is evaluated on cycle number or cycle iterations (training
      iterations since start of cycle).

    last_batch_iteration : int (default=-1)
      The index of the last batch.

    Examples
    --------

    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = torch.optim.CyclicLR(optimizer)
    >>> data_loader = torch.utils.data.DataLoader(...)
    >>> for epoch in range(10):
    >>>     for batch in data_loader:
    >>>         scheduler.batch_step()
    >>>         train_batch(...)

    References
    ----------
    .. [1] Leslie N. Smith, 2017, "Cyclical Learning Rates for Training Neural Networks,". "ICLR" `<https://arxiv.org/abs/1506.01186>`_

    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle',
                 last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lrs, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lrs, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def step(self, epoch=None):
        pass

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
