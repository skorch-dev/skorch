"""Tests for callbacks.py"""

from functools import partial
import itertools
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from .conftest import get_history


class TestAllCallbacks:
    @pytest.fixture
    def callbacks(self):
        """Return all callbacks"""
        import skorch.callbacks

        callbacks = []
        for name in dir(skorch.callbacks):
            attr = getattr(skorch.callbacks, name)
            # pylint: disable=unidiomatic-typecheck
            if not type(attr) is type:
                continue
            if issubclass(attr, skorch.callbacks.Callback):
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
            assert "kwargs" in inspect.signature(method).parameters


class TestScoring:
    @pytest.yield_fixture
    def scoring_cls(self):
        with patch('skorch.callbacks.to_var') as to_var:
            to_var.side_effect = lambda x: x

            from skorch.callbacks import Scoring
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
        ).initialize()

    @pytest.fixture
    def net(self):
        from skorch.history import History

        net = Mock(infer=Mock(side_effect=lambda x: x))
        history = History()
        net.history = history
        return net

    @pytest.fixture
    def train_loss(self, scoring_cls):
        from skorch.net import train_loss_score
        return scoring_cls(
            'train_loss',
            train_loss_score,
            on_train=True,
            target_extractor=lambda x: x,
            pred_extractor=lambda x: x,
        ).initialize()

    @pytest.fixture
    def valid_loss(self, scoring_cls):
        from skorch.net import valid_loss_score
        return scoring_cls(
            'valid_loss',
            valid_loss_score,
            target_extractor=lambda x: x,
            pred_extractor=lambda x: x,
        ).initialize()

    @pytest.fixture
    def history(self, train_loss, valid_loss, mse_scoring):
        return get_history(train_loss, valid_loss, mse_scoring)

    def test_correct_train_loss_values(self, history):
        train_losses = history[:, 'train_loss']
        expected = [0.25, 0.65, -0.15]
        assert np.allclose(train_losses, expected)

    def test_correct_valid_loss_values(self, history):
        valid_losses = history[:, 'valid_loss']
        expected = [7.5, 3.5, 11.5]
        assert np.allclose(valid_losses, expected)

    def test_missing_batch_size(self, train_loss, history):
        """We skip one batch size entry in history. This batch should
        simply be ignored.

        """
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 20)
        # missing batch size, loss of 20 is ignored

        net = Mock(history=history)
        train_loss.on_epoch_end(net)

        assert history[-1, 'train_loss'] == 10

    def test_average_honors_weights(self, train_loss, history):
        """The batches may have different batch sizes, which is why it
        necessary to honor the batch sizes. Here we use different
        batch sizes to verify this.

        """
        from skorch.history import History

        history = History()
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 40)
        history.record_batch('train_batch_size', 2)

        net = Mock(history=history)
        train_loss.on_epoch_end(net)

        assert history[0, 'train_loss'] == 30

    def test_best_train_loss_correct(self, history):
        train_loss_best = history[:, 'train_loss_best']
        expected = [True, False, True]
        assert train_loss_best == expected

    def test_best_valid_loss_correct(self, history):
        valid_loss_best = history[:, 'valid_loss_best']
        expected = [True, True, False]
        assert valid_loss_best == expected

    def test_best_loss_with_other_key(self, scoring_cls):
        """Test correct best loss with a loss that simply returns the
        epoch. Since epochs increase, only the first epoch should be
        best.

        """
        # pylint: disable=unused-argument
        def get_bs(net, *args, **kwargs):
            return net.history[-1, 'batches', -1, 'valid_batch_size']

        bs_loss = scoring_cls(
            'bs_loss',
            get_bs,
            target_extractor=lambda x: x,
            pred_extractor=lambda x: x,
        ).initialize()
        history = get_history(bs_loss)

        bs_losses = history[:, 'bs_loss_best']
        expected = [True, False, False]
        assert bs_losses == expected

    def test_correct_mse_values_for_batches(self, history):
        mse = history[:, 'batches', :, 'mse']
        # for the 4 batches per epoch, the loss is constant
        expected = [[12.5] * 4, [0.0] * 4, [1.0] * 4]
        assert np.allclose(mse, expected)

    def test_correct_mse_values_for_epoch(self, history):
        mse = history[:, 'mse']
        expected = [12.5, 0.0, 1.0]
        assert np.allclose(mse, expected)

    def test_correct_mse_is_best(self, history):
        is_best = history[:, 'mse_best']
        assert is_best == [True, True, False]

    def test_other_score_and_name(self, scoring_cls, net):
        """Test that we can change the scoring to accuracy score."""
        scoring = scoring_cls(
            name='acc',
            scoring='accuracy_score',
        )
        for x, y in zip(np.arange(5), reversed(np.arange(5))):
            # The net just returns input values as output; therefore, accuracy
            # is 1 for 2=2 and 0 elsewhere
            net.history.new_epoch()
            net.history.new_batch()
            scoring.on_batch_end(net, [x], [y], train=False)

        acc = net.history[:, 'batches', :, 'acc']
        expected = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]).reshape(-1, 1)
        assert np.allclose(acc, expected)

    def test_custom_scoring_func(self, scoring_cls, net):
        """When passing a custom scoring function, it should be used
        to determine the score.

        """
        # pylint: disable=unused-argument
        def score_func(estimator, X, y):
            return 555

        scoring = scoring_cls(
            name='acc',
            scoring=score_func,
        )
        for x, y in zip(np.arange(5), reversed(np.arange(5))):
            net.history.new_epoch()
            net.history.new_batch()
            scoring.on_batch_end(net, [x], [y], train=False)

        acc = net.history[:, 'batches', :, 'acc']
        expected = [555] * 5
        assert np.allclose(acc, expected)

    def test_scoring_func_none(self, scoring_cls, net):
        """If the scoring function is None, the `score` method of the
        model should be used.

        """
        net.score = Mock(return_value=345)
        scoring = scoring_cls(
            name='acc',
            scoring=None,
        )
        for x, y in zip(np.arange(5), reversed(np.arange(5))):
            net.history.new_epoch()
            net.history.new_batch()
            scoring.on_batch_end(net, [x], [y], train=False)

        acc = net.history[:, 'batches', :, 'acc']
        expected = [345] * 5
        assert np.allclose(acc, expected)

    def test_score_func_does_not_exist(self, scoring_cls, net):
        """When passing a string to `scoring`, it is looked up from
        among the sklearn metrics. If it doesn't exist, we expect a
        useful error message.

        """
        scoring = scoring_cls(
            name='myscore',
            scoring='nonexistant-score',
        )
        with pytest.raises(NameError) as exc:
            net.history.new_epoch()
            net.history.new_batch()
            scoring.on_batch_end(net, X=[0], y=[0], train=False)

        expected = ("A metric called 'nonexistant-score' does not exist, "
                    "use a valid sklearn metric name.")
        assert str(exc.value) == expected

    def test_train_is_ignored(self, mse_scoring, net):
        """By default, the score is determined on the validation
        set. Train is thus ignored.

        """
        for _ in range(3):
            net.history.new_epoch()
            net.history.new_batch()
            mse_scoring.on_batch_end(net, X=[0], y=[0], train=True)

        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            net.history[:, 'batches', :, 'mse']

    def test_train_is_used(self, scoring_cls, net):
        """By default, the score is determined on the validation
        set. When we set `on_train=True`, train data is used.

        """
        mse_scoring = scoring_cls(
            name='mse',
            scoring='mean_squared_error',
            on_train=True,
        )

        for _ in range(3):
            net.history.new_epoch()
            net.history.new_batch()
            mse_scoring.on_batch_end(net, X=[0], y=[0], train=True)

        mse_losses = net.history[:, 'batches', :, 'mse']
        assert np.allclose(mse_losses, [0., 0., 0.])

    def test_valid_is_ignored(self, scoring_cls, net):
        """When setting `on_train=True`, the valid losses should not
        be used.

        """
        mse_scoring = scoring_cls(
            name='mse',
            scoring='mean_squared_error',
            on_train=True,
        )

        for _ in range(3):
            net.history.new_epoch()
            net.history.new_batch()
            mse_scoring.on_batch_end(net, X=[0], y=[0], train=False)

        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            net.history[:, 'batches', :, 'mse']

    # pylint: disable=unused-argument
    def test_target_extractor_is_called(self, mse_scoring, history, mock_data):
        # note: the history fixture is required even if not used because it
        # triggers the calls on mse_scoring
        call_args_list = mse_scoring.target_extractor.call_args_list
        for i in range(3):  # 3 epochs
            data = mock_data[i][1]  # the targets
            for j in range(4):  # 4 batches
                assert call_args_list[4 * i + j][0][0] == data

    # pylint: disable=unused-argument
    def test_pred_extractor_is_called(self, mse_scoring, history, mock_data):
        # note: the history fixture is required even if not used because it
        # triggers the calls on mse_scoring
        call_args_list = mse_scoring.pred_extractor.call_args_list
        for i in range(3):  # 3 epochs
            data = mock_data[i][0]  # the predictions
            for j in range(4):  # 4 batches
                assert call_args_list[4 * i + j][0][0] == data

    def test_is_best_ignored_when_none(self, scoring_cls):
        mse_scoring = scoring_cls(
            name='mse',
            scoring='mean_squared_error',
            lower_is_better=None,
        ).initialize()
        history = get_history(mse_scoring)
        with pytest.raises(KeyError):
            # Since lower_is_better is None, 'is_best' key should not be
            # written
            # pylint: disable=pointless-statement
            history[:, 'is_best']


class TestPrintLog:
    @pytest.fixture
    def print_log_cls(self):
        from skorch.callbacks import PrintLog
        keys_ignored = ['batches', 'dur', 'text']
        return partial(PrintLog, sink=Mock(), keys_ignored=keys_ignored)

    @pytest.fixture
    def print_log(self, print_log_cls):
        return print_log_cls().initialize()

    @pytest.fixture
    def scoring_cls(self):
        from skorch.callbacks import Scoring
        return Scoring

    @pytest.fixture
    def train_loss(self, scoring_cls):
        from skorch.net import train_loss_score
        return scoring_cls(
            'train_loss',
            train_loss_score,
            on_train=True,
            target_extractor=lambda x: x,
            pred_extractor=lambda x: x,
        ).initialize()

    @pytest.fixture
    def valid_loss(self, scoring_cls):
        from skorch.net import valid_loss_score
        return scoring_cls(
            'valid_loss',
            valid_loss_score,
            target_extractor=lambda x: x,
            pred_extractor=lambda x: x,
        ).initialize()

    @pytest.fixture
    def history(self, train_loss, valid_loss, print_log):
        return get_history(train_loss, valid_loss, print_log)

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
        with patch('skorch.callbacks.tabulate') as tab:
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
        keys_ignored = ['batches']  # 'text' and 'dur' no longer ignored
        print_log = print_log_cls(
            sink=Mock(), keys_ignored=keys_ignored).initialize()
        # does not raise
        print_log.on_epoch_end(Mock(history=history))

        header = print_log.sink.call_args_list[0][0][0]
        columns = header.split()
        expected = ['epoch', 'text', 'train_loss', 'valid_loss', 'dur']
        assert columns == expected

    def test_keys_ignored_as_str(self, print_log_cls):
        print_log = print_log_cls(keys_ignored='a-key')
        assert print_log.keys_ignored == ['a-key']

    def test_no_valid(self, train_loss, valid_loss, print_log, ansi):
        get_history(train_loss, valid_loss, print_log, with_valid=False)
        sink = print_log.sink
        row = sink.call_args_list[2][0][0]
        items = row.split()

        assert len(items) == 2  # no valid
        # epoch
        assert items[0] == '1'
        # color 1 used for item 1
        assert items[1] == list(ansi)[1].value + '0.2500' + ansi.ENDC.value

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
