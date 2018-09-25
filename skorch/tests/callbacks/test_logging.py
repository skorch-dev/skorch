from functools import partial
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest


class TestPrintLog:
    @pytest.fixture
    def print_log_cls(self):
        from skorch.callbacks import PrintLog
        keys_ignored = ['dur', 'event_odd']
        return partial(PrintLog, sink=Mock(), keys_ignored=keys_ignored)

    @pytest.fixture
    def print_log(self, print_log_cls):
        return print_log_cls().initialize()

    @pytest.fixture
    def scoring_cls(self):
        from skorch.callbacks import EpochScoring
        return EpochScoring

    @pytest.fixture
    def mse_scoring(self, scoring_cls):
        return scoring_cls(
            'neg_mean_squared_error',
            name='nmse',
        ).initialize()

    @pytest.fixture
    def odd_epoch_callback(self):
        from skorch.callbacks import Callback
        class OddEpochCallback(Callback):
            def on_epoch_end(self, net, **kwargs):
                net.history[-1]['event_odd'] = bool(len(net.history) % 2)
        return OddEpochCallback().initialize()

    @pytest.fixture
    def net(self, net_cls, module_cls, train_split, mse_scoring,
            odd_epoch_callback, print_log, data):
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[mse_scoring, odd_epoch_callback], max_epochs=2)
        net.initialize()
        # replace default PrintLog with test PrintLog
        net.callbacks_[-1] = ('print_log', print_log)
        return net.partial_fit(*data)

    @pytest.fixture
    def history(self, net):
        return net.history

    # pylint: disable=unused-argument
    @pytest.fixture
    def sink(self, history, print_log):
        # note: the history fixture is required even if not used because it
        # triggers the calls on print_log
        return print_log.sink

    @pytest.fixture
    def ansi(self):
        from skorch.utils import Ansi
        return Ansi

    def test_call_count(self, sink):
        # header + lines + 2 epochs
        assert sink.call_count == 4

    def test_header(self, sink):
        header = sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['epoch', 'nmse', 'train_loss', 'valid_loss']
        assert columns == expected

    def test_lines(self, sink):
        lines = sink.call_args_list[1][0][0].split()
        # Lines have length 2 + length of column, or 8 if the column
        # name is short and the values are floats.
        expected = [
            '-' * (len('epoch') + 2),
            '-' * 8,
            '-' * (len('train_loss') + 2),
            '-' * (len('valid_loss') + 2),
        ]
        assert lines
        assert lines == expected

    @pytest.mark.parametrize('epoch', [0, 1])
    def test_first_row(self, sink, ansi, epoch, history):
        row = sink.call_args_list[epoch + 2][0][0]
        items = row.split()

        # epoch, nmse, valid, train
        assert len(items) == 4

        # epoch, starts at 1
        assert items[0] == str(epoch + 1)

        # is best
        are_best = [
            history[epoch, 'nmse_best'],
            history[epoch, 'train_loss_best'],
            history[epoch, 'valid_loss_best'],
        ]

        # test that cycled colors are used if best
        for item, color, is_best in zip(items[1:], list(ansi)[1:], are_best):
            if is_best:
                # if best, text colored
                assert item.startswith(color.value)
                assert item.endswith(ansi.ENDC.value)
            else:
                # if not best, text is only float, so converting possible
                float(item)

    def test_args_passed_to_tabulate(self, history):
        with patch('skorch.callbacks.logging.tabulate') as tab:
            from skorch.callbacks import PrintLog
            print_log = PrintLog(
                tablefmt='latex',
                floatfmt='.9f',
            ).initialize()
            print_log.table(history[-1])

            assert tab.call_count == 1
            assert tab.call_args_list[0][1]['tablefmt'] == 'latex'
            assert tab.call_args_list[0][1]['floatfmt'] == '.9f'

    def test_with_additional_key(self, history, print_log_cls):
        keys_ignored = ['event_odd']  # 'dur' no longer ignored
        print_log = print_log_cls(
            sink=Mock(), keys_ignored=keys_ignored).initialize()
        # does not raise
        print_log.on_epoch_end(Mock(history=history))

        header = print_log.sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['epoch', 'nmse', 'train_loss', 'valid_loss', 'dur']
        assert columns == expected

    def test_keys_ignored_as_str(self, print_log_cls):
        print_log = print_log_cls(keys_ignored='a-key')
        assert print_log.keys_ignored == ['a-key']

        print_log.initialize()
        assert print_log.keys_ignored_ == set(['a-key', 'batches'])

    def test_keys_ignored_is_None(self, print_log_cls):
        print_log = print_log_cls(keys_ignored=None)
        assert print_log.keys_ignored is None

        print_log.initialize()
        assert print_log.keys_ignored_ == set(['batches'])

    def test_with_event_key(self, history, print_log_cls):
        print_log = print_log_cls(sink=Mock(), keys_ignored=None).initialize()
        # history has two epochs, write them one by one
        print_log.on_epoch_end(Mock(history=history[:-1]))
        print_log.on_epoch_end(Mock(history=history))

        header = print_log.sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['epoch', 'nmse', 'train_loss', 'valid_loss', 'odd', 'dur']
        assert columns == expected

        odd_row = print_log.sink.call_args_list[2][0][0].split()
        even_row = print_log.sink.call_args_list[3][0][0].split()
        assert len(odd_row) == 6   # odd row has entries in every column
        assert odd_row[4] == '+'   # including '+' sign for the 'event_odd'
        assert len(even_row) == 5  # even row does not have 'event_odd' entry

    def test_witout_valid_data(
            self, net_cls, module_cls, mse_scoring, print_log, data):
        net = net_cls(
            module_cls, batch_size=1, train_split=None,
            callbacks=[mse_scoring], max_epochs=2)
        net.initialize()
        # replace default PrintLog with test PrintLog
        net.callbacks_[-1] = ('print_log', print_log)
        net.partial_fit(*data)

        sink = print_log.sink
        row = sink.call_args_list[2][0][0]
        items = row.split()

        assert len(items) == 2  # no valid, only epoch and train

    def test_print_not_skipped_if_verbose(self, capsys):
        from skorch.callbacks import PrintLog

        print_log = PrintLog().initialize()
        net = Mock(history=[{'loss': 123}], verbose=1)

        print_log.on_epoch_end(net)

        stdout = capsys.readouterr()[0]
        result = [x.strip() for x in stdout.split()]
        expected = ['loss', '------', '123']
        assert result == expected

    def test_print_skipped_if_not_verbose(self, capsys):
        from skorch.callbacks import PrintLog

        print_log = PrintLog().initialize()
        net = Mock(history=[{'loss': 123}], verbose=0)

        print_log.on_epoch_end(net)

        stdout = capsys.readouterr()[0]
        assert not stdout


class TestProgressBar:
    @pytest.fixture
    def progressbar_cls(self):
        from skorch.callbacks import ProgressBar
        return ProgressBar

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 2 epochs"""
        from skorch import NeuralNetRegressor
        from skorch.toy import make_regressor

        module_cls = make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        return partial(
            NeuralNetRegressor,
            module=module_cls,
            train_split=None,
            max_epochs=2,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        X = np.zeros((20, 1), dtype='float32')
        y = np.zeros((20, 1), dtype='float32')
        return X, y

    @pytest.mark.parametrize('postfix', [
        [],
        ['train_loss'],
        ['train_loss', 'valid_loss'],
        ['doesnotexist'],
        ['train_loss', 'doesnotexist'],
    ])
    def test_invalid_postfix(self, postfix, net_cls, progressbar_cls, data):
        net = net_cls(callbacks=[
            progressbar_cls(postfix_keys=postfix),
        ])
        net.fit(*data)

    @patch('tqdm.tqdm')
    @pytest.mark.parametrize('scheme,expected_total', [
        ('auto', [2, 2]),
        ('count', [None, 2]),
        (None, [None, None]),
        (2, [2, 2]),  # correct number of batches_per_epoch (20 // 10)
        (3, [3, 3]),  # offset by +1, should still work
        (1, [1, 1]),  # offset by -1, should still work
    ])
    def test_different_count_schemes(
            self, tqdm_mock, scheme, expected_total, net_cls, progressbar_cls, data):
        net = net_cls(callbacks=[
            progressbar_cls(batches_per_epoch=scheme),
        ])
        net.fit(*data)
        assert tqdm_mock.call_count == 2
        for i, total in enumerate(expected_total):
            assert tqdm_mock.call_args_list[i][1]['total'] == total
