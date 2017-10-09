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

from skorch.utils import Ansi
from skorch.utils import to_numpy
from skorch.utils import to_var


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

    lower_is_better : bool (default=True)
      Whether lower (e.g. log loss) or higher (e.g. accuracy) scores
      are better

    on_train : bool (default=False)
      Whether this should be called during train or validation.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    pred_extractor : callable (default=to_numpy)
      This is called on y_pred before it is passed to scoring.

    """
    _op_dict = {True: operator.lt, False: operator.gt}

    def __init__(
            self,
            name='myscore',
            scoring=None,
            lower_is_better=True,
            on_train=False,
            target_extractor=to_numpy,
            pred_extractor=to_numpy,
    ):
        self.name = name
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.target_extractor = target_extractor
        self.pred_extractor = pred_extractor
        self.on_train = on_train

    def initialize(self):
        self.best_loss_ = np.inf if self.lower_is_better else -np.inf
        return self

    def _scoring(self, net, X, y):
        """Resolve scoring and apply it to data."""
        y = self.target_extractor(y)
        if self.scoring is None:
            score = net.score(X, y)
        elif isinstance(self.scoring, str):  # TODO: make py2.7 compatible
            # scoring is a string
            try:
                scorer = getattr(metrics, self.scoring)
            except AttributeError:
                raise NameError("A metric called '{}' does not exist, "
                                "use a valid sklearn metric name."
                                "".format(self.scoring))
            y_pred = self.pred_extractor(net.infer(to_var(X)))
            score = scorer(y, y_pred)
        else:
            # scoring is a function
            score = self.scoring(net, X, y)

        return score

    # pylint: disable=unused-argument,arguments-differ
    def on_batch_end(self, net, X, y, train, **kwargs):
        if train != self.on_train:
            return

        try:
            score = self._scoring(net, X, y)
            net.history.record_batch(self.name, score)
        except KeyError:
            pass

    def get_avg_loss(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, losses = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name]]))
        loss_avg = np.average(losses, weights=weights)
        return loss_avg

    def is_best_loss(self, loss):
        if self.lower_is_better is None:
            return None
        op = self._op_dict[self.lower_is_better]
        return op(loss, self.best_loss_)

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:
            history[-1, 'batches', :, self.name]
        except KeyError:
            return

        loss_avg = self.get_avg_loss(history)
        is_best = self.is_best_loss(loss_avg)
        if is_best:
            self.best_loss_ = loss_avg

        history.record(self.name, loss_avg)
        if is_best is not None:
            history.record(self.name + '_best', is_best)


class PrintLog(Callback):
    """Print out useful information from the model's history.

    By default, `PrintLog` prints everything from the history except
    for `'batches'`.

    To determine the best loss, `PrintLog` looks for keys that end on
    `'_best'` and associates them with the corresponding loss. E.g.,
    `'train_loss_best'` will be matched with `'train_loss'`. The
    `Scoring` callback takes care of creating those entries, which is
    why `PrintLog` works best in conjunction with that callback.

    *Note*: `PrintLog` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.

    Parameters
    ----------
    keys_ignored : str or list of str (default='batches')
      Key or list of keys that should not be part of the printed
      table. Note that keys ending on '_best' are also ignored.

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
            keys_ignored='batches',
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
    ):
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

    def initialize(self):
        self.first_iteration_ = True
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable.

        """
        value = row[key]
        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _sorted_keys(self, keys):
        """Sort keys alphabetically, but put 'epoch' first and 'dur'
        last.

        Ignore keys that are in `self.ignored_keys` or that end on
        '_best'.

        """
        sorted_keys = []
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored):
            sorted_keys.append('epoch')

        for key in sorted(keys):
            if not (
                    (key in ('epoch', 'dur')) or
                    (key in self.keys_ignored) or
                    key.endswith('_best')
            ):
                sorted_keys.append(key)

        if ('dur' in keys) and ('dur' not in self.keys_ignored):
            sorted_keys.append('dur')
        return sorted_keys

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        color = next(colors)
        for key in self._sorted_keys(row.keys()):
            formatted = self.format_row(row, key, color=color)
            yield key, formatted
            color = next(colors)

    def table(self, row):
        headers = []
        formatted = []
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)
            formatted.append(formatted_row)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
        )

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        data = net.history[-1]
        verbose = net.verbose
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink is print:
            sys.stdout.flush()
