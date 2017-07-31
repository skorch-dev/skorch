"""Contains callback base class and callbacks."""

from itertools import cycle
from numbers import Number
import operator
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn import metrics
from tabulate import tabulate

from inferno.utils import Ansi
from inferno.utils import check_history_slice
from inferno.utils import to_numpy
from inferno.utils import to_var


class Callback:
    """Base class for callbacks.

    All custom callbacks should inherit from this class. The subclass
    may override any of the `on_...` methods. It is, however, not
    necessary to override all of them, since it's okay if they don't
    have any effect.

    Classes that inherit from this also gain the `get_params` and
    `set_params` method.

    """
    def initialize(self):
        """(Re-)Set the initial state of the callback. Use this
        e.g. if the callback tracks some state that should be reset
        when the model is re-initialized.

        This method should return self.

        """
        return self

    def on_train_begin(self, net, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, net, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, net, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, net, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, net, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, net, **kwargs):
        """Called at the end of each batch."""
        pass

    def _get_param_names(self):
        return (key for key in self.__dict__ if not key.endswith('_'))

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self


class EpochTimer(Callback):
    """Measures the duration of each epoch and writes it to the
    history with the name `dur`.

    """
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', time.time() - self.epoch_start_time_)


class AverageLoss(Callback):
    """Determines the average loss.

    By default, average train loss and valid loss are determined, if
    present.

    Parameters
    ----------
    key_sizes : dict or None (default=None)
      If not None, this should be a dictionary whose keys are the
      columns in 'history' on which to measure the average,
      e.g. 'train_loss'. The values should be the corresponding batch
      sizes, e.g. 'train_batch_size'. (The latter are required to
      determine the correct average in case the batch sizes are not
      the same across all batches.)

    Attributes
    ----------
    default_key_sizes
      By default, average train loss and average valid loss are
      determined. If any of the losses is not present, it is
      ignored.

    """
    default_key_sizes = {'train_loss': 'train_batch_size',
                         'valid_loss': 'valid_batch_size'}

    def __init__(self, key_sizes=None):
        self.key_sizes = {} if key_sizes is None else key_sizes

    def _yield_key_losses_bs(self, history):
        for key_loss, key_size in self.default_key_sizes.items():
            try:
                row = history[-1, 'batches', :, (key_loss, key_size)]
                yield key_loss, list(zip(*row))
            except KeyError:
                pass
        for key_loss, key_size in self.key_sizes.items():
            row = history[-1, 'batches', :, (key_loss, key_size)]
            yield key_loss, list(zip(*row))

    def on_epoch_end(self, net, **kwargs):
        for key, size in self.key_sizes.items():
            sl = np.s_[-1, 'batches', :, (key, size)]
            check_history_slice(net.history, sl)

        history = net.history
        for key_loss, (losses, bs) in self._yield_key_losses_bs(history):
            loss = np.average(losses, weights=bs)
            history.record(key_loss, loss)


class BestLoss(Callback):
    """For each epoch, determine whether the given loss is the best
    loss achieved yet.

    By default, train loss and valid loss are analyzed.

    Parameters
    ----------
    keys_possible : None or list of str (default=None)
      If list of str, the strings should be the name of the column in
      the history that contains the values that should be analyzed.
      If keys are not found for a specific epoch, they are ignored.

    signs : list or tuple of int (default=(-1, -1))
      The signs should be either -1 or 1. They determine whether the
      value should be minimized (-1) or maximized (1). E.g.,
      cross-entropy losses should be minimized and hence get -1,
      accuracy should be maximized and hence get 1.

    """
    _op_dict = {-1: operator.lt, 1: operator.gt}

    def __init__(self, keys_possible=None, signs=(-1, -1)):
        if keys_possible is None:
            self.keys_possible = ['train_loss', 'valid_loss']
        else:
            self.keys_possible = keys_possible
        self.signs = signs

        self.best_losses_ = None

    def initialize(self):
        if len(self.keys_possible) != len(self.signs):
            raise ValueError("The number of keys and signs should be equal.")

        self.best_losses_ = {key: -1 * sign * np.inf for key, sign
                             in zip(self.keys_possible, self.signs)}
        return self

    def _yield_key_sign_loss(self, history):
        for key, sign in zip(self.keys_possible, self.signs):
            try:
                loss = history[-1, key]
                yield key, sign, loss
            except KeyError:
                pass

    def on_epoch_end(self, net, **kwargs):
        history = net.history
        for key, sign, loss in self._yield_key_sign_loss(history):
            is_best = False
            op = self._op_dict[sign]
            if op(loss, self.best_losses_[key]):
                is_best = True
                self.best_losses_[key] = loss
            history.record('{}_best'.format(key), is_best)


class Scoring(Callback):
    """Callback that performs generic scoring on predictions.

    Parameters
    ----------
    name : str (default='myscore')
      The name of the score. Determines the column name in the
      history.

    scoring : None, str, or callable (default=None)
      If None, use the `score` method of the model. If str, it should
      be a valid sklearn metric (e.g. "f1_score", "accuracy_score"). If
      a callable, it should have the signature (model, X, y), and it
      should return a scalar. This works analogously to the `scoring`
      parameter in sklearn's `GridSearchCV` et al.

    on_train : bool (default=False)
      Whether this should be called during train or validation.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    pred_extractor : callable (default=to_numpy)
      This is called on y_pred before it is passed to scoring.

    """
    def __init__(
            self,
            name='myscore',
            scoring=None,
            on_train=False,
            target_extractor=to_numpy,
            pred_extractor=to_numpy,
    ):
        self.name = name
        self.scoring = scoring
        self.target_extractor = target_extractor
        self.pred_extractor = pred_extractor
        self.on_train = on_train

    def on_batch_end(self, net, X, y, train):
        if train != self.on_train:
            return

        y = self.target_extractor(y)
        if self.scoring is None:
            score = net.score(X, y)
        elif isinstance(self.scoring, str):  # TODO: make py2.7 compatible
            # scoring is a string
            try:
                scorer = getattr(metrics, self.scoring)
            except AttributeError:
                raise NameError("Metric with name '{}' does not exist, "
                                "use a valid sklearn metric name."
                                "".format(self.scoring))
            y_pred = self.pred_extractor(net.infer(to_var(X)))
            score = scorer(y, y_pred)
        else:
            # scoring is a function
            score = self.scoring(net, X, y)

        net.history.record_batch(self.name, score)


class PrintLog(Callback):
    """Print out useful information from the model's history.

    By default, this will print the epoch, train loss, valid loss, and
    epoch duration. In addition, `PrintLog` will take care of
    highlighting the best loss at each epoch.

    To determine the best loss, `PrintLog` looks for keys that end on
    `'_best'` and associates them with the corresponding loss. E.g.,
    `'train_loss_best'` will be matched with `'train_loss'`. The
    `BestLoss` callback takes care of creating those entries, which is
    why `PrintLog` works best in conjunction with that callback.

    Parameters
    ----------
    keys : list of str
      The columns from history that should be printed. Keys that end
      on `'_best'` are used to determine the best corresponding loss
      (see above).

    sink : callable (default=print)
      The target that the output string is sent to. By default, the
      output is printed to stdout, but the sink could also be a
      logger, etc.

    tablefmt : str (default='simple')
      The format of the table. See the documentation of the `tabulate`
      package for more detail. Can be 'plain', 'grid', 'pipe', 'html',
      'latex', among others.

    floatfmt : str (default='.4f')
      The number formatting. See the documentation of the `tabulate`
      package for more details.

    """
    def __init__(
            self,
            keys=('epoch', 'train_loss', 'valid_loss', 'train_loss_best',
                  'valid_loss_best', 'dur'),
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
    ):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

    def initialize(self):
        self.first_iteration_ = True
        self.idx_ = {key: i for i, key in enumerate(self.keys)}
        return self

    def format_row(self, row):
        row_formatted = []
        colors = cycle(Ansi)

        for key, item in zip(self.keys, row):
            if key.endswith('_best'):
                continue

            if not isinstance(item, Number):
                row_formatted.append(item)
                continue

            color = next(colors)
            # if numeric, there could be a 'best' key
            idx_best = self.idx_.get(key + '_best')

            is_integer = float(item).is_integer()
            template = '{}' if is_integer else '{:' + self.floatfmt + '}'

            if (idx_best is not None) and row[idx_best]:
                template = color.value + template + Ansi.ENDC.value
            row_formatted.append(template.format(item))

        return row_formatted

    def table(self, data):
        formatted = [self.format_row(row) for row in data]
        headers = [key for key in self.keys if not key.endswith('_best')]
        return tabulate(
            formatted,
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
        )

    def on_epoch_end(self, net, *args, **kwargs):
        sl = slice(-1, None), self.keys
        check_history_slice(net.history, sl)
        data = net.history[sl]
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self.sink(header)
            self.sink(lines)
            self.first_iteration_ = False

        self.sink(tabulated.rsplit('\n', 1)[-1])
        if self.sink is print:
            sys.stdout.flush()
