"""Contains callback base class and callbacks."""

from itertools import chain
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
from inferno.utils import duplicate_items
from inferno.utils import flatten
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

    keys_optional: list of str or None (default=None)
      If not None, this should be a list of keys whose presence is
      optional. By default, keys indicated in `key_sizes` are
      mandatory, but sometimes we want optional keys. E.g., if keys
      refer to validation data, but validation data is not always
      present, those keys should be optional. When a key that is not
      optional is missing, an exception will be raised.

    Attributes
    ----------
    default_key_sizes
      By default, average train loss and average valid loss are
      determined. If any of the losses is not present, it is
      ignored.

    """
    default_key_sizes = {'train_loss': 'train_batch_size',
                         'valid_loss': 'valid_batch_size'}

    def __init__(self, key_sizes=None, keys_optional=None):
        self.key_sizes = {} if key_sizes is None else key_sizes

        if keys_optional is None:
            self.keys_optional = []
        elif isinstance(keys_optional, str):
            self.keys_optional = [keys_optional]
        else:
            self.keys_optional = keys_optional

        self._check_keys_duplicated()

    def _is_optional(self, key):
        return (key in self.keys_optional) or (key in self.default_key_sizes)

    def _yield_key_losses_bs(self, history):
        key_sizes = chain(
            self.default_key_sizes.items(), self.key_sizes.items())

        for key_loss, key_size in key_sizes:
            try:
                row = row = history[-1, 'batches', :, (key_loss, key_size)]
                yield key_loss, list(zip(*row))
            except KeyError:
                if self._is_optional(key_loss) or self._is_optional(key_size):
                    continue
                raise

    def _check_keys_duplicated(self):
        duplicates = duplicate_items(self.default_key_sizes, self.key_sizes)
        if duplicates:
            raise ValueError("AverageLoss found duplicate keys: {}"
                             "".format(', '.join(sorted(duplicates))))

    def _check_keys_missing(self, history):
        check_keys = []
        # check loss keys and size keys alike
        for key in flatten(self.key_sizes.items()):
            if not self._is_optional(key):
                check_keys.append(key)
        sl = np.s_[-1, 'batches', :, check_keys]
        check_history_slice(history, sl)

    def on_epoch_end(self, net, **kwargs):
        self._check_keys_missing(net.history)

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
    key_signs : dict or None
      If not None, this should be a dictionary whose keys are the
      columns in 'history' that should be checked for whether they
      reached the best value yet, e.g. 'train_loss'. The values should
      be -1 if lower values are better and 1 if higher values are
      better. E.g., log loss should get -1, whereas accuracy should
      get 1.

    keys_optional: list of str or None (default=None)
      If not None, this should be a list of keys whose presence is
      optional. By default, keys indicated in `key_signs` are
      mandatory, but sometimes we want optional keys. E.g., if keys
      refer to validation data, but validation data is not always
      present, those keys should be optional. When a key that is not
      optional is missing, an exception will be raised.

    Attributes
    ----------
    default_key_signs
      By default, the best epochs for average train loss and average
      valid loss are determined. If any of the losses is not present,
      it is ignored.

    """
    default_key_signs = {'train_loss': -1, 'valid_loss': -1}
    _op_dict = {-1: operator.lt, 1: operator.gt}

    def __init__(self, key_signs=None, keys_optional=None):
        self.key_signs = {} if key_signs is None else key_signs
        if keys_optional is None:
            self.keys_optional = []
        elif isinstance(keys_optional, str):
            self.keys_optional = [keys_optional]
        else:
            self.keys_optional = keys_optional

        self.best_losses_ = None

        self._check_keys_duplicated()
        self._check_signs()

    def _is_optional(self, key):
        return (key in self.keys_optional) or (key in self.default_key_signs)

    def _check_keys_duplicated(self):
        duplicates = duplicate_items(self.default_key_signs, self.key_signs)
        if duplicates:
            raise ValueError("BestLoss found duplicate keys: {}"
                             "".format(', '.join(sorted(duplicates))))

    def _check_signs(self):
        signs = chain(self.default_key_signs.values(), self.key_signs.values())
        signs_allowed = sorted(self._op_dict.keys())
        for sign in signs:
            if sign not in signs_allowed:
                raise ValueError(
                    "Wrong sign {}, expected one of {}."
                    "".format(sign, ", ".join(map(str, signs_allowed))))

    def initialize(self):
        items = chain(self.default_key_signs.items(), self.key_signs.items())
        self.best_losses_ = {key: -1 * sign * np.inf for key, sign in items}
        return self

    def _check_keys_missing(self, history):
        check_keys = [k for k in self.key_signs if not self._is_optional(k)]
        sl = np.s_[-1, check_keys]
        check_history_slice(history, sl)

    def _yield_key_sign_loss(self, history):
        key_signs = chain(
            self.default_key_signs.items(), self.key_signs.items())

        for key_loss, sign in key_signs:
            try:
                loss = history[-1, key_loss]
                yield key_loss, sign, loss
            except KeyError:
                if not self._is_optional(key_loss):
                    raise

    def on_epoch_end(self, net, **kwargs):
        self._check_keys_missing(net.history)

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

    *Note*: `PrintLog` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.

    Parameters
    ----------
    keys : list of str
      The columns from history that should be printed. Keys that end
      on `'_best'` are used to determine the best corresponding loss
      (see above).

    keys_optional: list of str or None (default=None)
      If not None, this should be a list of keys whose presence is
      optional. By default, keys indicated in `key` must be in the
      history, but sometimes we want optional keys. E.g., if keys
      refer to validation data, but validation data is not always
      present, those keys should be optional. When a key that is not
      optional is missing, an exception will be raised.

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

    Attributes
    ----------
    default_keys
      By default, `PrintLog` prints the epoch, the train loss, the
      valid loss, and the time it took to process the epoch
      ("dur"). It will also highlight the best train and valid
      loss. If any of the mentioned keys is not found, it is ignored.

    """

    default_keys = ['epoch', 'train_loss', 'valid_loss', 'train_loss_best',
                    'valid_loss_best', 'dur']

    def __init__(
            self,
            keys=None,
            keys_optional=None,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
    ):
        if keys is None:
            self.keys = []
        else:
            self.keys = [keys] if isinstance(keys, str) else keys

        if keys_optional is None:
            self.keys_optional = []
        elif isinstance(keys_optional, str):
            self.keys_optional = [keys_optional]
        else:
            self.keys_optional = keys_optional

        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

        self._check_keys_duplicated()

    def _is_optional(self, key):
        return (key in self.keys_optional) or (key in self.default_keys)

    def _check_keys_duplicated(self):
        duplicates = duplicate_items(self.default_keys, self.keys)
        if duplicates:
            raise ValueError("PrintLog found duplicate keys: {}"
                             "".format(', '.join(sorted(duplicates))))

    def initialize(self):
        self.first_iteration_ = True
        return self

    def _check_keys_missing(self, history):
        check_keys = [key for key in self.keys if not self._is_optional(key)]
        sl = np.s_[-1, check_keys]
        check_history_slice(history, sl)

    def format_row(self, row, key, color):
        value = row[key]
        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color.value + template + Ansi.ENDC.value
        return template.format(value)

    def _yield_keys_formatted(self, row):
        colors = cycle(Ansi)
        color = next(colors)
        for key in chain(self.default_keys, self.keys):
            if key.endswith('_best'):
                continue
            try:
                formatted = self.format_row(row, key, color=color)
                yield key, formatted
                color = next(colors)
            except KeyError:
                if not self._is_optional(key):
                    raise

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

    def on_epoch_end(self, net, *args, **kwargs):
        self._check_keys_missing(net.history)

        data = net.history[-1]
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self.sink(header)
            self.sink(lines)
            self.first_iteration_ = False

        self.sink(tabulated.rsplit('\n', 1)[-1])
        if self.sink is print:
            sys.stdout.flush()
