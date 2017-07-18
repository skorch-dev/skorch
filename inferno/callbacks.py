import time

import numpy as np
from sklearn.base import BaseEstimator


class Callback:
    def initialize(self):
        pass

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
    def compute_average_loss(self, net, loss_key, bs_key):
        losses = net.history[-1, 'batches', :, loss_key]
        batch_sizes = net.history[-1, 'batches', :, bs_key]
        return np.average(losses, weights=batch_sizes)

    def on_epoch_end(self, net, **kwargs):
        todo = []
        keys = [('train_loss', 'train_batch_size'),
                ('valid_loss', 'valid_batch_size')]

        for key in keys:
            try:
                # TODO: remove loop when net.history supports tuples
                # as selector
                for subkey in key:
                    net.history[-1, 'batches', :, subkey]

                todo.append(key)
            except KeyError:
                pass

        for k in todo:
            loss = self.compute_average_loss(net, *k)
            net.history.record(k[0], loss)
