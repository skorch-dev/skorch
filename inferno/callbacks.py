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
from inferno.utils import to_numpy
from inferno.utils import to_var
from inferno.utils import check_history_slice


class Callback:
    def initialize(self):
        return self

    def on_train_begin(self, net, **kwargs):
        pass

    def on_train_end(self, net, **kwargs):
        pass

    def on_epoch_begin(self, net, **kwargs):
        pass

    def on_epoch_end(self, net, **kwargs):
        pass

    def on_batch_begin(self, net, **kwargs):
        pass

    def on_batch_end(self, net, **kwargs):
        pass

    def _get_param_names(self):
        return (key for key in self.__dict__ if not key.endswith('_'))

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)


class EpochTimer(Callback):
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', time.time() - self.epoch_start_time_)


class AverageLoss(Callback):
    def __init__(self, keys_possible=None):
        if keys_possible is None:
            self.keys_possible = [('train_loss', 'train_batch_size'),
                                  ('valid_loss', 'valid_batch_size')]
        else:
            self.keys_possible = keys_possible

    def _yield_key_losses_bs(self, history):
        for key_tuple in self.keys_possible:
            try:
                row = history[-1, 'batches', :, key_tuple]
                yield key_tuple[0], list(zip(*row))
            except KeyError:
                pass

    def on_epoch_end(self, net, **kwargs):
        history = net.history
        for key_loss, (losses, bs) in self._yield_key_losses_bs(history):
            loss = np.average(losses, weights=bs)
            history.record(key_loss, loss)


class BestLoss(Callback):
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
    def __init__(
            self,
            name,
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
            y_pred = self.pred_extractor(net.module_(to_var(X)))
            score = scorer(y, y_pred)
        else:
            # scoring is a function
            score = self.scoring(net, X, y)

        net.history.record_batch(self.name, score)


class PrintLog(Callback):
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
