from functools import partial
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from .conftest import get_history


class TestAverageLoss:
    @pytest.fixture
    def avg_loss_cls(self):
        from inferno.callbacks import AverageLoss
        return AverageLoss

    @pytest.fixture
    def avg_loss(self, avg_loss_cls):
        return avg_loss_cls().initialize()

    @pytest.fixture
    def history_avg_loss(self, avg_loss):
        return get_history(avg_loss)

    def test_correct_losses(self, history_avg_loss):
        train_losses = history_avg_loss[:, 'train_loss']
        expected = [0.25, 0.65, -0.15]
        assert np.allclose(train_losses, expected)

        valid_losses = history_avg_loss[:, 'valid_loss']
        expected = [7.5, 3.5, 11.5]
        assert np.allclose(valid_losses, expected)

    def test_missing_batch_size(self, avg_loss, history):
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 20)
        # missing batch size, 20 is ignored

        net = Mock(history=history)
        avg_loss.on_epoch_end(net)

        assert history[0, 'train_loss'] == 10

    def test_average_honors_weights(self, avg_loss, history):
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 40)
        history.record_batch('train_batch_size', 2)

        net = Mock(history=history)
        avg_loss.on_epoch_end(net)

        assert history[0, 'train_loss'] == 30

    def test_init_other_keys(self, avg_loss_cls):
        avg_loss = avg_loss_cls(keys_possible=[
            ('train_loss', 'train_batch_size')]).initialize()
        history = get_history(avg_loss)

        train_losses = history[:, 'train_loss']
        expected = [0.25, 0.65, -0.15]
        assert np.allclose(train_losses, expected)

        with pytest.raises(KeyError):
            history[:, 'valid_loss']


class TestBestLoss:
    @pytest.fixture
    def avg_loss(self):
        from inferno.callbacks import AverageLoss
        return AverageLoss().initialize()

    @pytest.fixture
    def best_loss_cls(self):
        from inferno.callbacks import BestLoss
        return BestLoss

    @pytest.fixture
    def best_loss(self, best_loss_cls):
        return best_loss_cls(signs=(-1, 1)).initialize()

    @pytest.fixture
    def history_best_loss(self, avg_loss, best_loss):
        return get_history(avg_loss, best_loss)

    def test_best_loss_correct(self, history_best_loss):
        train_loss_best = history_best_loss[:, 'train_loss_best']
        expected = [True, False, True]
        assert train_loss_best == expected

        valid_loss_best = history_best_loss[:, 'valid_loss_best']
        expected = [True, False, True]
        assert valid_loss_best == expected

    def test_other_signs(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(signs=(1, -1)).initialize()
        history = get_history(avg_loss, best_loss)

        train_loss_best = history[:, 'train_loss_best']
        expected = [True, True, False]
        assert train_loss_best == expected

        valid_loss_best = history[:, 'valid_loss_best']
        expected = [True, True, False]
        assert valid_loss_best == expected

    def test_init_other_keys(self, best_loss_cls, avg_loss):
        best_loss = best_loss_cls(
            keys_possible=('valid_loss',),
            signs=(1,),
        ).initialize()
        history = get_history(avg_loss, best_loss)

        with pytest.raises(KeyError):
            history[:, 'train_loss_best']

        valid_loss_best = history[:, 'valid_loss_best']
        expected = [True, False, True]
        assert valid_loss_best == expected


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

        net = Mock(module_=Mock(side_effect=lambda x: x))
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
        for x, y in zip(np.arange(5), np.arange(5)[::-1]):
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
        for x, y in zip(np.arange(5), np.arange(5)[::-1]):
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
        for x, y in zip(np.arange(5), np.arange(5)[::-1]):
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
        return partial(PrintLog, sink=Mock())

    @pytest.fixture
    def print_log(self, print_log_cls):
        return print_log_cls(keys=(
            'epoch', 'train_loss', 'train_loss_best', 'valid_loss',
            'valid_loss_best')).initialize()

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
        # epoch
        assert items[0] == '1'
        # color 1 used for item 1
        assert items[1] == list(ansi)[1].value + '0.2500' + ansi.ENDC.value
        # color 2 used for item 1
        assert items[2] == list(ansi)[2].value + '7.5000' + ansi.ENDC.value

    def test_second_row(self, sink, ansi):
        row = sink.call_args_list[3][0][0]
        items = row.split()
        assert items[0] == '2'
        # not best, hence no color
        assert items[1] == '0.6500'
        assert items[2] == list(ansi)[2].value + '3.5000' + ansi.ENDC.value

    def test_third_row(self, sink, ansi):
        row = sink.call_args_list[4][0][0]
        items = row.split()
        assert items[0] == '3'
        assert items[1] == list(ansi)[1].value + '-0.1500' + ansi.ENDC.value
        assert items[2] == '11.5000'

    def test_args_passed_to_tabulate(self, history):
        with patch('inferno.callbacks.tabulate') as tab:
            from inferno.callbacks import PrintLog
            print_log = PrintLog(
                keys=('epoch',),
                tablefmt='some-table',
                floatfmt='some-float',
            ).initialize()
            print_log.table(history[:, ('epoch',)])

            assert tab.call_count == 1
            assert tab.call_args_list[0][1]['tablefmt'] == 'some-table'
            assert tab.call_args_list[0][1]['floatfmt'] == 'some-float'

    def test_with_one_key(self, history, print_log_cls):
        key = 'epoch'
        print_log = print_log_cls(keys=key).initialize()
        # does not raise
        print_log.on_epoch_end(Mock(history=history))

    def test_with_1_missing_key(self, history, print_log_cls):
        keys = 'train_loss', 'missing-key'
        print_log = print_log_cls(keys=keys).initialize()
        with pytest.raises(KeyError) as exc:
            print_log.on_epoch_end(Mock(history=history))

        expected = ("Key 'missing-key' could not be found in history; "
                    "maybe there was a typo?")
        assert exc.value.args[0] == expected

    def test_with_2_missing_keys(self, history, print_log_cls):
        keys = 'missing-key0', 'train_loss', 'missing-key1'
        print_log = print_log_cls(keys=keys).initialize()
        with pytest.raises(KeyError) as exc:
            print_log.on_epoch_end(Mock(history=history))

        expected = ("Key 'missing-key0' could not be found in history; "
                    "maybe there was a typo?")
        assert exc.value.args[0] == expected
