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
            scoring,
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

        if isinstance(self.scoring, str):  # TODO: make py2.7 compatible
            # scoring is a string
            y = self.target_extractor(y)
            scorer = getattr(metrics, self.scoring)
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
            float_template='{:.4f}',
    ):
        self.keys = keys
        self.sink = sink
        self.tablefmt = tablefmt
        self.float_template = float_template

    def initialize(self):
        self.first_iteration_ = True
        self.idx_ = {key: i for i, key in enumerate(self.keys)}
        return self

    def _format_rows(self, rows):
        for row in rows:
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
                template = '{}' if is_integer else self.float_template

                if (idx_best is not None) and row[idx_best]:
                    template = color.value + template + Ansi.ENDC.value
                row_formatted.append(template.format(item))

            yield row_formatted

    def table(self, history):
        data = list(self._format_rows(history[:, self.keys]))
        headers = [key for key in self.keys if not key.endswith('_best')]
        return tabulate(
            data,
            headers=headers,
            tablefmt=self.tablefmt,
        )

    def on_epoch_end(self, net, *args, **kwargs):
        tabulated = self.table(net.history)
        out = ""

        if self.first_iteration_:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"
            self.first_iteration_ = False
        out += tabulated.rsplit('\n', 1)[-1]

        self.sink(out)
        if self.sink is print:
            sys.stdout.flush()
