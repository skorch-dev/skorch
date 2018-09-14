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
import torch
import skorch
import skorch.helper
import sklearn.datasets
import numpy as np
import resource

class MyModule(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=2, n_layers=2):
        super().__init__()
        d = hidden_dim
        layers = [torch.nn.Linear(input_dim, d)]
        layers += [torch.nn.Linear(d, d) for _ in range(n_layers)]
        layers += [torch.nn.Linear(d, output_dim)]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, X):
        return torch.softmax(self.seq(X), dim=-1)


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



np.random.seed(0)
torch.manual_seed(0)


def test_a():
    # -- first by stripping parameters explicitly
    np.random.seed(0)
    torch.manual_seed(0)

    print('1', end='')
    mod = MyModule(n_layers=N_LAYERS)

    # freeze all params but last layer
    for i in range(len(mod.seq) - 1):
        skorch.utils.freeze_parameter(mod.seq[i].weight)
        skorch.utils.freeze_parameter(mod.seq[i].bias)

    opt = skorch.helper.filtered_optimizer(
            torch.optim.SGD,
            skorch.helper.filter_requires_grad)

    net = skorch.NeuralNetClassifier(
            mod,
            verbose=0,
            optimizer=opt,
            warm_start=True)

    for i in range(len(mod.seq) - 1):
        assert not mod.seq[i].weight.requires_grad

    net.fit(X, y)

    for i in range(len(mod.seq) - 1):
        assert not mod.seq[i].weight.requires_grad
        assert not net.module_.seq[i].weight.requires_grad

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss, net.history[-1, 'valid_loss'], np.mean(net.history[:, 'dur'])


def test_b(verbose=False):
    # -- second by simply freezing them
    np.random.seed(0)
    torch.manual_seed(0)

    print('2', end='')
    mod = MyModule(n_layers=N_LAYERS)

    opt = torch.optim.SGD
    cb = skorch.callbacks.Freezer(
        ['seq.{}.weight'.format(i) for i in range(len(mod.seq) - 1)] +
        ['seq.{}.bias'.format(i) for i in range(len(mod.seq) - 1)]
    )

    net = skorch.NeuralNetClassifier(
            mod,
            verbose=0,
            optimizer=opt,
            callbacks=[cb])
    net.fit(X, y)

    for i in range(len(mod.seq) - 1):
        assert not mod.seq[i].weight.requires_grad
        assert not net.module_.seq[i].weight.requires_grad

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss, net.history[-1, 'valid_loss'], np.mean(net.history[:, 'dur'])


if __name__ == '__main__':
    n_runs = 10
    print(f'running tests for {n_runs} runs each.')

    dur_a = np.mean([test_a() for _ in range(n_runs)], axis=0)
    dur_b = np.mean([test_b() for _ in range(n_runs)], axis=0)

    print()

    print(f'test_a: µ_rss = {dur_a[0]}, µ_valid_loss = {dur_a[1]}, µ_dur={dur_a[2]}')
    print(f'test_a: µ_rss = {dur_b[0]}, µ_valid_loss = {dur_b[1]}, µ_dur={dur_b[2]}')

    # valid losses must be identical
    assert np.allclose(dur_a[1], dur_b[1])

    # memory usage should be nearly identical (within 4MiB)
    assert abs(dur_a[0] - dur_b[0]) < (4*1024**2)

    # duration should be nearly identical
    assert abs(dur_a[2] - dur_b[2]) < 1e-3
