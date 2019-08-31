"""Tests for callbacks/logging.py"""

from functools import partial
import os
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from skorch.tests.conftest import tensorboard_installed


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
        print_log = print_log_cls(keys_ignored='a-key').initialize()
        assert print_log.keys_ignored_ == {'a-key', 'batches'}

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


@pytest.mark.skipif(
    not tensorboard_installed, reason='tensorboard is not installed')
class TestTensorBoard:
    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def data(self, classifier_data):
        X, y = classifier_data
        # accelerate training since we don't care for the loss
        X, y = X[:40], y[:40]
        return X, y

    @pytest.fixture
    def tensorboard_cls(self):
        from skorch.callbacks import TensorBoard
        return TensorBoard

    @pytest.fixture
    def summary_writer_cls(self):
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter

    @pytest.fixture
    def mock_writer(self, summary_writer_cls):
        mock = Mock(spec=summary_writer_cls)
        return mock

    @pytest.fixture
    def net_fitted(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            mock_writer,
    ):
        return net_cls(
            classifier_module,
            callbacks=[tensorboard_cls(mock_writer)],
            max_epochs=3,
        ).fit(*data)

    @pytest.mark.skipif(
        True, reason="Waiting for proper implementation of graph tracing")
    def test_graph_added_once(self, net_fitted, mock_writer):
        # graph should just be added once
        assert mock_writer.add_graph.call_count == 1

    @pytest.mark.skipif(
        True, reason="Waiting for proper implementation of graph tracing")
    def test_include_graph_false(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            mock_writer,
    ):
        net_cls(
            classifier_module,
            callbacks=[tensorboard_cls(mock_writer, include_graph=False)],
            max_epochs=2,
        ).fit(*data)
        assert mock_writer.add_graph.call_count == 0

    def test_writer_closed_automatically(self, net_fitted, mock_writer):
        assert mock_writer.close.call_count == 1

    def test_writer_not_closed(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            mock_writer,
    ):
        net_cls(
            classifier_module,
            callbacks=[tensorboard_cls(mock_writer, close_after_train=False)],
            max_epochs=2,
        ).fit(*data)
        assert mock_writer.close.call_count == 0

    def test_keys_from_history_logged(self, net_fitted, mock_writer):
        add_scalar = mock_writer.add_scalar

        # 3 epochs with 4 keys
        assert add_scalar.call_count == 3 * 4
        keys = {call_args[1]['tag'] for call_args in add_scalar.call_args_list}
        expected = {'dur', 'Loss/train_loss', 'Loss/valid_loss', 'Loss/valid_acc'}
        assert keys == expected

    def test_ignore_keys(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            mock_writer,
    ):
        # ignore 'dur' and 'valid_loss', 'unknown' doesn't exist but
        # this should not cause a problem
        tb = tensorboard_cls(
            mock_writer, keys_ignored=['dur', 'valid_loss', 'unknown'])
        net_cls(
            classifier_module,
            callbacks=[tb],
            max_epochs=3,
        ).fit(*data)
        add_scalar = mock_writer.add_scalar

        keys = {call_args[1]['tag'] for call_args in add_scalar.call_args_list}
        expected = {'Loss/train_loss', 'Loss/valid_acc'}
        assert keys == expected

    def test_keys_ignored_is_string(self, tensorboard_cls, mock_writer):
        tb = tensorboard_cls(mock_writer, keys_ignored='a-key').initialize()
        expected = {'a-key', 'batches'}
        assert tb.keys_ignored_ == expected

    def test_other_key_mapper(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            mock_writer,
    ):
        # just map all keys to uppercase
        tb = tensorboard_cls(mock_writer, key_mapper=lambda s: s.upper())
        net_cls(
            classifier_module,
            callbacks=[tb],
            max_epochs=3,
        ).fit(*data)
        add_scalar = mock_writer.add_scalar

        keys = {call_args[1]['tag'] for call_args in add_scalar.call_args_list}
        expected = {'DUR', 'TRAIN_LOSS', 'VALID_LOSS', 'VALID_ACC'}
        assert keys == expected

    @pytest.fixture
    def add_scalar_maybe(self, tensorboard_cls, mock_writer):
        tb = tensorboard_cls(mock_writer)
        return tb.add_scalar_maybe

    @pytest.fixture
    def history(self):
        return [
            {'loss': 0.1, 'epoch': 1, 'foo': ['invalid', 'type']},
            {'loss': 0.2, 'epoch': 2, 'foo': ['invalid', 'type']},
        ]

    def test_add_scalar_maybe_uses_last_epoch_values(
            self, add_scalar_maybe, mock_writer, history):
        add_scalar_maybe(history, key='loss', tag='myloss', global_step=2)
        call_kwargs = mock_writer.add_scalar.call_args_list[0][1]
        assert call_kwargs['tag'] == 'myloss'
        assert call_kwargs['scalar_value'] == 0.2
        assert call_kwargs['global_step'] == 2

    def test_add_scalar_maybe_infers_epoch(
            self, add_scalar_maybe, mock_writer, history):
        # don't indicate 'global_step' value
        add_scalar_maybe(history, key='loss', tag='myloss')
        call_kwargs = mock_writer.add_scalar.call_args_list[0][1]
        assert call_kwargs['global_step'] == 2

    def test_add_scalar_maybe_unknown_key_does_not_raise(
            self, tensorboard_cls, summary_writer_cls, history):
        tb = tensorboard_cls(summary_writer_cls())
        # does not raise:
        tb.add_scalar_maybe(history, key='unknown', tag='bar')

    def test_add_scalar_maybe_wrong_type_does_not_raise(
            self, tensorboard_cls, summary_writer_cls, history):
        tb = tensorboard_cls(summary_writer_cls())
        # value of 'foo' is a list but that does not raise:
        tb.add_scalar_maybe(history, key='foo', tag='bar')

    def test_fit_with_real_summary_writer(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            summary_writer_cls,
            tmp_path,
    ):
        path = str(tmp_path)

        net = net_cls(
            classifier_module,
            callbacks=[tensorboard_cls(summary_writer_cls(path))],
            max_epochs=5,
        )
        net.fit(*data)

        # is not empty
        assert os.listdir(path)

    def test_fit_with_dict_input(
            self,
            net_cls,
            classifier_module,
            data,
            tensorboard_cls,
            summary_writer_cls,
            tmp_path,
    ):
        from skorch.toy import MLPModule
        path = str(tmp_path)
        X, y = data

        # create a dictionary with unordered keys
        X_dict = {k: X[:, i:i+4] for k, i in zip('cebad', range(0, X.shape[1], 4))}

        class MyModule(MLPModule):
            # use different order for args here
            def forward(self, b, e, c, d, a, **kwargs):
                X = torch.cat((b, e, c, d, a), 1)
                return super().forward(X, **kwargs)

        net = net_cls(
            MyModule(output_nonlin=nn.Softmax(dim=-1)),
            callbacks=[tensorboard_cls(summary_writer_cls(path))],
            max_epochs=5,
        )
        net.fit(X_dict, y)

        # is not empty
        assert os.listdir(path)
