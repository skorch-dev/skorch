from functools import partial
import itertools
from unittest.mock import Mock
from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import pytest

from .conftest import get_history


class TestAllCallbacks:
    @pytest.fixture
    def callbacks(self):
        import inferno.callbacks
        callbacks = []
        for name in dir(inferno.callbacks):
            attr = getattr(inferno.callbacks, name)
            if not type(attr) is type:
                continue
            if issubclass(attr, inferno.callbacks.Callback):
                callbacks.append(attr)
        return callbacks

    @pytest.fixture
    def on_x_methods(self):
        return [
            'on_train_begin',
            'on_train_end',
            'on_epoch_begin',
            'on_epoch_end',
            'on_batch_begin',
            'on_batch_end',
        ]

    def test_on_x_methods_have_kwargs(self, callbacks, on_x_methods):
        import inspect
        for callback, method_name in itertools.product(
                callbacks, on_x_methods):
            method = getattr(callback, method_name)
            argspec = inspect.getargspec(method)
            assert argspec.keywords

    def test_key_missing(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(key_signs={'missing': 1}).initialize()

        with pytest.raises(KeyError) as exc:
            get_history(avg_loss, best_loss)

        expected = ("Key 'missing' could not be found in history; "
                    "maybe there was a typo? To make this key optional, "
                    "add it to the 'keys_optional' parameter.")
        assert exc.value.args[0] == expected

    def test_missing_key_optional(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(
            key_signs={'missing': 1}, keys_optional=['missing']).initialize()

        # does not raise
        get_history(avg_loss, best_loss)

    def test_missing_key_optional_as_str(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(
            key_signs={'missing': 1}, keys_optional='missing').initialize()

        # does not raise
        get_history(avg_loss, best_loss)

    def test_sign_not_allowed(self, best_loss_cls):
        with pytest.raises(ValueError) as exc:
            best_loss_cls(key_signs={'epoch': 2}).initialize()

        expected = "Wrong sign 2, expected one of -1, 1."
        assert exc.value.args[0] == expected

    def test_1_duplicate_key(self, best_loss_cls):
        key_signs = {'train_loss': 1}
        with pytest.raises(ValueError) as exc:
            best_loss_cls(key_signs=key_signs).initialize()

        expected = "BestLoss found duplicate keys: train_loss"
        assert exc.value.args[0] == expected

    def test_2_duplicate_keys(self, best_loss_cls):
        key_signs = {'train_loss': 1, 'valid_loss': 1}
        with pytest.raises(ValueError) as exc:
            best_loss_cls(key_signs=key_signs).initialize()

        expected = "BestLoss found duplicate keys: train_loss, valid_loss"
        assert exc.value.args[0] == expected


class TestScoring:
    @pytest.yield_fixture
    def scoring_cls(self):
        with patch('inferno.callbacks.to_var') as to_var:
            to_var.side_effect = lambda x: x

            from inferno.callbacks import Scoring
            yield partial(
                Scoring,
                target_extractor=Mock(side_effect=lambda x: x),
                pred_extractor=Mock(side_effect=lambda x: x),
            )

    @pytest.fixture
    def mse_scoring(self, scoring_cls):
        return scoring_cls(
            name='mse',
            scoring='mean_squared_error',
        )

    @pytest.fixture
    def net(self):
        from inferno.net import History

        net = Mock(infer=Mock(side_effect=lambda x: x))
        history = History()
        history.new_epoch()
        net.history = history
        return net

    @pytest.fixture
    def data(self):
        return [
            [[3, -2.5], [6, 1.5]],
            [[1, 0], [0, -1]],
        ]

    @pytest.fixture
    def history(self, mse_scoring, net, data):
        for x, y in data:
            net.history.new_batch()
            mse_scoring.on_batch_end(net, x, y, train=False)
        return net.history

    def test_correct_mse(self, history):
        mse = history[:, 'batches', :, 'mse']
        expected = [[12.5, 1.0]]
        assert np.allclose(mse, expected)

    def test_other_score_and_name(self, scoring_cls, net):
        scoring = scoring_cls(
            name='acc',
            scoring='accuracy_score',
        )
        for x, y in zip(np.arange(5), reversed(np.arange(5))):
            net.history.new_batch()
            scoring.on_batch_end(net, [x], [y], train=False)

        acc = net.history[:, 'batches', :, 'acc']
        expected = [0.0, 0.0, 1.0, 0.0, 0.0]
        assert np.allclose(acc, expected)

    def test_custom_scoring_func(self, scoring_cls, net):
        def score_func(estimator, X, y):
            return 555

        scoring = scoring_cls(
            name='acc',
            scoring=score_func,
        )
        for x, y in zip(np.arange(5), reversed(np.arange(5))):
            net.history.new_batch()
            scoring.on_batch_end(net, [x], [y], train=False)

        acc = net.history[:, 'batches', :, 'acc']
        expected = [555] * 5
        assert np.allclose(acc, expected)

    def test_scoring_func_none(self, scoring_cls, net):
        net.score = Mock(return_value=345)
        scoring = scoring_cls(
            name='acc',
            scoring=None,
        )
        for x, y in zip(np.arange(5), reversed(np.arange(5))):
            net.history.new_batch()
            scoring.on_batch_end(net, [x], [y], train=False)

        acc = net.history[:, 'batches', :, 'acc']
        expected = [345] * 5
        assert np.allclose(acc, expected)

    def test_score_func_does_not_exist(self, scoring_cls, net, data):
        scoring = scoring_cls(
            name='myscore',
            scoring='nonexistant-score',
        )
        with pytest.raises(NameError) as exc:
            net.history.new_batch()
            scoring.on_batch_end(net, data[0][0], data[0][1], train=False)

        expected = ("Metric with name 'nonexistant-score' does not exist, "
                    "use a valid sklearn metric name.")
        assert str(exc.value) == expected

    def test_train_is_ignored(self, mse_scoring, net, data):
        for x, y in data:
            net.history.new_batch()
            mse_scoring.on_batch_end(net, x, y, train=True)

        with pytest.raises(KeyError):
            net.history[:, 'batches', :, 'mse']

    def test_valid_is_ignored(self, scoring_cls, net, data):
        mse_scoring = scoring_cls(
            name='mse',
            scoring='mean_squared_error',
            on_train=True,
        )

        for x, y in data:
            net.history.new_batch()
            mse_scoring.on_batch_end(net, x, y, train=False)

        with pytest.raises(KeyError):
            net.history[:, 'batches', :, 'mse']

    def test_target_extractor_is_called(self, mse_scoring, data, history):
        # note: the history fixture is required even if not used because it
        # triggers the calls on mse_scoring
        call_args_list = mse_scoring.target_extractor.call_args_list
        for (_, x), call_args in zip(data, call_args_list):
            assert x == call_args[0][0]

    def test_pred_extractor_is_called(self, mse_scoring, data, history):
        # note: the history fixture is required even if not used because it
        # triggers the calls on mse_scoring
        call_args_list = mse_scoring.pred_extractor.call_args_list
        for (x, _), call_args in zip(data, call_args_list):
            assert x == call_args[0][0]


class TestPrintLog:
    @pytest.fixture
    def print_log_cls(self):
        from inferno.callbacks import PrintLog
        keys_ignored = ['batches', 'dur', 'text']
        return partial(PrintLog, sink=Mock(), keys_ignored=keys_ignored)

    @pytest.fixture
    def print_log(self, print_log_cls):
        return print_log_cls().initialize()

    @pytest.fixture
    def avg_loss(self):
        from inferno.callbacks import AverageLoss
        return AverageLoss().initialize()

    @pytest.fixture
    def best_loss(self):
        from inferno.callbacks import BestLoss
        return BestLoss().initialize()

    @pytest.fixture
    def history(self, avg_loss, best_loss, print_log):
        return get_history(avg_loss, best_loss, print_log)

    @pytest.fixture
    def sink(self, history, print_log):
        # note: the history fixture is required even if not used because it
        # triggers the calls on print_log
        return print_log.sink

    @pytest.fixture
    def ansi(self):
        from inferno.utils import Ansi
        return Ansi

    def test_call_count(self, sink):
        # header + lines + 3 epochs
        assert sink.call_count == 5

    def test_header(self, sink):
        header = sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['epoch', 'train_loss', 'valid_loss']
        assert columns == expected

    def test_lines(self, sink):
        lines = sink.call_args_list[1][0][0].split()
        header = sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['-' * (len(col) + 2) for col in columns]
        assert lines
        assert lines == expected

    def test_first_row(self, sink, ansi):
        row = sink.call_args_list[2][0][0]
        items = row.split()

        assert len(items) == 3
        # epoch
        assert items[0] == '1'
        # color 1 used for item 1
        assert items[1] == list(ansi)[1].value + '0.2500' + ansi.ENDC.value
        # color 2 used for item 1
        assert items[2] == list(ansi)[2].value + '7.5000' + ansi.ENDC.value

    def test_second_row(self, sink, ansi):
        row = sink.call_args_list[3][0][0]
        items = row.split()

        assert len(items) == 3
        assert items[0] == '2'
        # not best, hence no color
        assert items[1] == '0.6500'
        assert items[2] == list(ansi)[2].value + '3.5000' + ansi.ENDC.value

    def test_third_row(self, sink, ansi):
        row = sink.call_args_list[4][0][0]
        items = row.split()

        assert len(items) == 3
        assert items[0] == '3'
        assert items[1] == list(ansi)[1].value + '-0.1500' + ansi.ENDC.value
        assert items[2] == '11.5000'

    def test_args_passed_to_tabulate(self, history):
        with patch('inferno.callbacks.tabulate') as tab:
            from inferno.callbacks import PrintLog
            print_log = PrintLog(
                tablefmt='latex',
                floatfmt='.9f',
            ).initialize()
            print_log.table(history[-1])

            assert tab.call_count == 1
            assert tab.call_args_list[0][1]['tablefmt'] == 'latex'
            assert tab.call_args_list[0][1]['floatfmt'] == '.9f'

    def test_with_additional_key(self, history):
        from inferno.callbacks import PrintLog

        keys_ignored = ['batches']  # 'text' and 'dur' no longer ignored
        print_log = PrintLog(
            sink=Mock(), keys_ignored=keys_ignored).initialize()
        # does not raise
        print_log.on_epoch_end(Mock(history=history))

        header = print_log.sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['epoch', 'text', 'train_loss', 'valid_loss', 'dur']
        assert columns == expected

    def test_keys_ignored_as_str(self, history, print_log_cls):
        from inferno.callbacks import PrintLog
        print_log = PrintLog(keys_ignored='a-key')
        assert print_log.keys_ignored == ['a-key']

    def test_no_valid(self, avg_loss, best_loss, print_log, ansi):
        get_history(avg_loss, best_loss, print_log, with_valid=False)
        sink = print_log.sink
        row = sink.call_args_list[2][0][0]
        items = row.split()

        assert len(items) == 2  # no valid
        # epoch
        assert items[0] == '1'
        # color 1 used for item 1
        assert items[1] == list(ansi)[1].value + '0.2500' + ansi.ENDC.value
