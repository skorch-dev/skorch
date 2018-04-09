from collections import defaultdict
from pydoc import locate
from functools import partial

import numpy as np
from sklearn import metrics

from skorch.utils import to_numpy
from skorch.callbacks import Callback


__all__ = ['BatchMetrics', 'EpochMetrics']


class MetricsBase(Callback):
    class _Metric:
        def __init__(
            self,
            func,
            lower_is_better=True,
            on_train=False,
            name=None,
        ):
            func_ = func
            if isinstance(func, str):
                func_ = getattr(metrics, func, None)
                if func_ is None:
                    func_ = locate(func)
            assert func_ is not None

            if name is None:
                if isinstance(func_, partial):
                    name = func_.func.__name__
                else:
                    name = func_.__name__

            self.func = func_
            self.lower_is_better = lower_is_better
            self.on_train = on_train
            self.name = name

            self.best_ = np.inf if self.lower_is_better else -np.inf

        def __call__(self, y_true, y_pred):
            return self.func(y_true, y_pred)

        def is_best(self, result, best):
            return (
                self.lower_is_better and result < best
                or
                not self.lower_is_better and result > best
                )

    make_metric = _Metric

    def __init__(
        self,
        metrics,
    ):
        metrics = [
            self.make_metric(**m) if isinstance(m, dict) else m
            for m in metrics
            ]

        self.metrics = metrics
        self.best_ = {
            m.name: np.inf if m.lower_is_better else -np.inf
            for m in metrics
            }


class EpochMetrics(MetricsBase):
    def on_epoch_begin(self, net, **kwargs):
        self.y_train_trues_ = []
        self.y_train_preds_ = []
        self.y_valid_trues_ = []
        self.y_valid_preds_ = []

    def on_batch_end(self, net, y, y_pred, training, **kwargs):
        if training:
            y_trues = self.y_train_trues_
            y_preds = self.y_train_preds_
        else:
            y_trues = self.y_valid_trues_
            y_preds = self.y_valid_preds_

        y_trues.append(to_numpy(y))
        y_preds.append(to_numpy(y_pred))

    def on_epoch_end(self, net, **kwargs):
        history = net.history

        y_train_true = np.concatenate(self.y_train_trues_)
        y_train_pred = np.concatenate(self.y_train_preds_)
        y_valid_true = np.concatenate(self.y_valid_trues_)
        y_valid_pred = np.concatenate(self.y_valid_preds_)

        for metric in self.metrics:
            name = metric.name
            if metric.on_train:
                y_true, y_pred = y_train_true, y_train_pred
            else:
                y_true, y_pred = y_valid_true, y_valid_pred
            result = metric(y_true, y_pred)
            history.record(name, result)
            if metric.is_best(result, self.best_[name]):
                history.record(name + '_best', True)
                self.best_[name] = result


class BatchMetrics(MetricsBase):
    def on_epoch_begin(self, net, **kwargs):
        self.metrics_ = defaultdict(list)

    def on_batch_end(self, net, y, y_pred, training, **kwargs):
        metrics = filter(lambda m: m.on_train is training, self.metrics)
        for metric in metrics:
            result = metric(y, y_pred)
            self.metrics_[metric.name].append(result)

    def on_epoch_end(self, net, **kwargs):
        history = net.history
        for metric in self.metrics:
            name = metric.name
            result = np.mean(self.metrics_[name])
            history.record(name, result)
            if metric.is_best(result, self.best_[name]):
                history.record(name + '_best', True)
                self.best_[name] = result
