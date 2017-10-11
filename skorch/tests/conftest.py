"""Contains shared fixtures, hooks, etc."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def history_cls():
    from skorch.history import History
    return History


@pytest.fixture
def history():
    return history_cls()


@pytest.fixture
def mock_data():
    return [
        [[3, -2.5], [6, 1.5]],
        [[2, 3], [2, 3]],
        [[1, 0], [0, -1]],
    ]


def get_history(*callbacks, with_valid=True):
    """Create a pseudo-history with variable callbacks. This does not
    necessitate training or mocking a NeuralNet, which is why we can
    call this often without losing too much time. However, this also
    replicates some of the logic related to callbacks.

    """
    h = history_cls()()
    net = Mock()
    net.history = h
    net.infer = lambda x: x
    data = mock_data()
    data = [(range(6, 10), 1, 'hi', data[0][0], data[0][1]),
            (range(2, 6), 2, 'ho', data[1][0], data[1][1]),
            (range(10, 14), 3, 'hu', data[2][0], data[2][1])]

    for range_, epoch, text, X, y in data:
        h.new_epoch()
        for cb in callbacks:
            cb.on_epoch_begin(net)

        for i in range_:
            h.new_batch()
            for cb in callbacks:
                cb.on_batch_begin(net)

            h.record_batch('train_loss', 1 - i / 10)
            h.record_batch('train_batch_size', 10)
            if with_valid:
                h.record_batch('valid_loss', i)
                h.record_batch('valid_batch_size', 1)

            for cb in callbacks:
                for train in (True, False):
                    cb.on_batch_end(net, X=X, y=y, train=train)

            h.record_batch('text', 'ha')

        h.record('epoch', epoch)
        h.record('text', text)
        h.record('dur', 0.123)
        for cb in callbacks:
            cb.on_epoch_end(net)

    return h


pandas_installed = False
try:
    # pylint: disable=unused-import
    import pandas
    pandas_installed = True
except ImportError:
    pass
