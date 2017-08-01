from unittest.mock import Mock

import pytest


@pytest.fixture
def history_cls():
    from inferno.net import History
    return History


@pytest.fixture
def history(history_cls):
    return history_cls()


def get_history(*callbacks, history_cls=history_cls, with_valid=True):
    h = history_cls()()
    net = Mock()
    net.history = h
    data = [(range(6, 10), 1, 'hi'),
            (range(2, 6), 2, 'ho'),
            (range(10, 14), 3, 'hu')]

    for range_, epoch, text in data:
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
                cb.on_batch_end(net)

        h.record('epoch', epoch)
        h.record('text', text)
        for cb in callbacks:
            cb.on_epoch_end(net)

    return h


pandas_installed = False
try:
    import pandas
    pandas_installed = True
except ImportError:
    pass
