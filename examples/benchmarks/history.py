"""Benchmark to test time and memory performance of History.

Before #312, the timing would be roughly 5 sec and memory usage would
triple. After #312, the timing would be roughly 2 sec and memory usage
roughly constant.

For the reasons, see #306.

"""

from pprint import pprint
import time

import numpy as np
from sklearn.datasets import make_classification
import torch

from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from skorch.toy import make_classifier


side_effects = []


class TriggerKeyError(Callback):
    def on_batch_end(self, net, **kwargs):
        try:
            net.history[-1, 'batches', -1, 'foobar']
        except Exception as e:
            pass


class PrintMemory(Callback):
    def on_batch_end(self, net, **kwargs):
        side_effects.append((
            torch.cuda.memory_allocated() / 1e6,
            torch.cuda.memory_cached() / 1e6
        ))


def train():
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    module = make_classifier(input_units=20)

    net = NeuralNetClassifier(
        module,
        max_epochs=10,
        lr=0.1,
        callbacks=[TriggerKeyError(), PrintMemory()],
        device='cuda',
    )

    return net.fit(X, y)


def safe_slice(history, keys):
    # catch errors
    for key in keys:
        try:
            history[key]
        except (KeyError, IndexError):
            pass


def performance_history(history):
    # SUCCESSFUL
    # level 0
    for i in range(len(history)):
        history[i]

    # level 1
    keys = tuple(history[0].keys())
    history[0, keys]
    history[:, keys]
    for key in keys:
        history[0, key]
        history[:, key]

    # level 2
    for i in range(len(history[0, 'batches'])):
        history[0, 'batches', i]
        history[:, 'batches', i]
    history[:, 'batches', :]

    # level 3
    keys = tuple(history[0, 'batches', 0].keys())
    history[0, 'batches', 0, keys]
    history[:, 'batches', 0, keys]
    history[0, 'batches', :, keys]
    history[:, 'batches', :, keys]
    for key in history[0, 'batches', 0]:
        history[0, 'batches', 0, key]
        history[:, 'batches', 0, key]
        history[0, 'batches', :, key]
        history[:, 'batches', :, key]

    # KEY ERRORS
    # level 0
    safe_slice(history, [100000])

    # level 1
    safe_slice(history, [np.s_[0, 'foo'], np.s_[:, 'foo']])

    # level 2
    safe_slice(history, [
        np.s_[0, 'batches', 0],
        np.s_[:, 'batches', 0],
        np.s_[0, 'batches', :],
        np.s_[:, 'batches', :],
    ])

    # level 3
    safe_slice(history, [
        np.s_[0, 'batches', 0, 'foo'],
        np.s_[:, 'batches', 0, 'foo'],
        np.s_[0, 'batches', :, 'foo'],
        np.s_[:, 'batches', :, 'foo'],
        np.s_[0, 'batches', 0, ('foo', 'bar')],
        np.s_[:, 'batches', 0, ('foo', 'bar')],
        np.s_[0, 'batches', :, ('foo', 'bar')],
        np.s_[:, 'batches', :, ('foo', 'bar')],
    ])

if __name__ == '__main__':
    net = train()
    tic = time.time()
    for _ in range(1000):
        performance_history(net.history)
    toc = time.time()
    print("Time for performing 1000 runs: {:.5f} sec.".format(toc - tic))
    assert toc - tic < 10, "accessing history is too slow"

    print("Allocated / cached memory")
    pprint(side_effects)

    mem_start = side_effects[0][0]
    mem_end = side_effects[-1][0]

    print("Memory epoch 1: {:.4f}, last epoch: {:.4f}".format(
        mem_start, mem_end))
    assert np.isclose(mem_start, mem_end, rtol=1/3), "memory use should be similar"
