"""Benchmark to test runtime and memory performance of
different freezing approaches.

Test A is done by setting `requires_grad=False` manually
while filtering these parameters from the optimizer using
``skorch.helper.filtered_optimizer``.

Test B uses the ``Freezer`` via ``ParamMapper`` without
explicitly removing the parameters from the optimizer.

In theory there should be no difference in memory
consumption and runtime.
"""
from functools import partial
import resource
from multiprocessing import Process, Queue

import torch
import skorch
import skorch.helper
from skorch.toy import make_classifier
import sklearn.datasets
import numpy as np


X, y = sklearn.datasets.make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=0)
X = X.astype('float32')
y = y.astype('int64')

N_LAYERS = 2

make_module_cls = partial(
    make_classifier,
    num_hidden=N_LAYERS,
    input_units=2,
    hidden_units=100,
    output_units=2,
)

linear_idcs = list(range(0, (N_LAYERS+1)*3, 3))

np.random.seed(0)
torch.manual_seed(0)


def test_a():
    # -- first by stripping parameters explicitly
    np.random.seed(0)
    torch.manual_seed(0)

    print('1', end='')
    mod = make_module_cls()()

    # freeze all params but last layer
    for i in linear_idcs[:-1]:
        skorch.utils.freeze_parameter(mod.sequential[i].weight)
        skorch.utils.freeze_parameter(mod.sequential[i].bias)

    opt = skorch.helper.filtered_optimizer(
        torch.optim.SGD,
        skorch.helper.filter_requires_grad)

    net = skorch.NeuralNetClassifier(
        mod,
        verbose=0,
        optimizer=opt,
        warm_start=True)

    for i in linear_idcs[:-1]:
        assert not mod.sequential[i].weight.requires_grad

    net.fit(X, y)

    for i in linear_idcs[:-1]:
        assert not mod.sequential[i].weight.requires_grad
        assert not net.module_.sequential[i].weight.requires_grad

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss, net.history[-1, 'valid_loss'], np.mean(net.history[:, 'dur'])


def test_b():
    # -- second by simply freezing them
    np.random.seed(0)
    torch.manual_seed(0)

    print('2', end='')
    mod = make_module_cls()()

    opt = torch.optim.SGD
    cb = skorch.callbacks.Freezer(
        ['sequential.{}.weight'.format(i) for i in linear_idcs[:-1]] +
        ['sequential.{}.bias'.format(i) for i in linear_idcs[:-1]]
    )

    net = skorch.NeuralNetClassifier(
        mod,
        verbose=0,
        optimizer=opt,
        callbacks=[cb])
    net.fit(X, y)

    for i in linear_idcs[:-1]:
        assert not mod.sequential[i].weight.requires_grad
        assert not net.module_.sequential[i].weight.requires_grad

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss, net.history[-1, 'valid_loss'], np.mean(net.history[:, 'dur'])

def test_runner(q, fn, n_runs):
    q.put(np.mean([fn() for _ in range(n_runs)], axis=0))

def test_forker(test_fn, n_runs):
    q = Queue()
    p = Process(target=test_runner, args=(q, test_fn, n_runs))
    p.start()
    res = q.get()
    p.join()
    return res

if __name__ == '__main__':
    n_runs = 10
    print(f'running tests for {n_runs} runs each.')

    # We fork the tests so that each has its own process and thus its
    # own memory allocation. Therefore tests don't influence each other.
    dur_a = test_forker(test_a, n_runs)
    print()

    dur_b = test_forker(test_b, n_runs)
    print()

    print(f'test_a: µ_rss = {dur_a[0]}, µ_valid_loss = {dur_a[1]}, µ_dur={dur_a[2]}')
    print(f'test_a: µ_rss = {dur_b[0]}, µ_valid_loss = {dur_b[1]}, µ_dur={dur_b[2]}')

    # valid losses must be identical
    assert np.allclose(dur_a[1], dur_b[1])

    # memory usage should be nearly identical (within 4MiB)
    assert np.allclose(dur_a[0], dur_b[0], atol=4*1024**2)

    # duration should be nearly identical
    assert np.allclose(dur_a[2], dur_b[2], atol=0.5)
