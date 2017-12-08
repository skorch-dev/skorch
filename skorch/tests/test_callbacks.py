"""Tests for callbacks.py"""

from functools import partial
import itertools
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest


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


class TestEpochScoring:
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

    def test_correct_valid_score(
            self, net_cls, module_cls, mse_scoring, train_split, data,
    ):
        net = net_cls(
            module=module_cls,
            callbacks=[mse_scoring],
            train_split=train_split,
            max_epochs=2,
        )
        net.fit(*data)

        expected = -np.mean([(3 - 5) ** 2, (0 - 4) ** 2])
        loss = net.history[:, 'nmse']
        assert np.allclose(loss, expected)

    def test_correct_train_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module=module_cls,
            callbacks=[scoring_cls(
                'neg_mean_squared_error',
                on_train=True,
                name='nmse',
                lower_is_better=False,
            )],
            train_split=train_split,
            max_epochs=2,
        )
        net.fit(*data)

        expected = -np.mean([(0 - -1) ** 2, (2 - 0) ** 2])
        loss = net.history[:, 'nmse']
        assert np.allclose(loss, expected)

    def test_scoring_uses_score_when_none(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None)],
            max_epochs=5,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'score']
        # these values are the hard-coded side_effects from net.score
        expected = [10, 8, 6, 11, 7]
        assert result == expected

    @pytest.mark.parametrize('lower_is_better, expected', [
        (True, [True, True, True, False, False]),
        (False, [True, False, False, True, False]),
        (None, []),
    ])
    def test_best_score_when_lower_is_better(
            self, net_cls, module_cls, scoring_cls, train_split, data,
            lower_is_better, expected,
    ):
        # set scoring to None so that mocked net.score is used
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring=None,
                lower_is_better=lower_is_better)],
            train_split=train_split,
            max_epochs=5,
        )
        net.fit(*data)

        if lower_is_better is not None:
            is_best = net.history[:, 'score_best']
            assert is_best == expected
        else:
            # if lower_is_better==None, don't write score
            with pytest.raises(KeyError):
                # pylint: disable=pointless-statement
                net.history[:, 'score_best']

    def test_no_error_when_no_valid_data(
            self, net_cls, module_cls, mse_scoring, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[mse_scoring],
            max_epochs=3,
            train_split=train_split,
        )
        net.fit(*data)

        net.train_split = None
        # does not raise
        net.partial_fit(*data)

        # only the first 3 epochs wrote scores
        assert len(net.history[:, 'nmse']) == 3

    def test_with_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('accuracy')],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'accuracy']
        assert result == [0, 0]

    def test_with_score_nonexisting_string(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('does-not-exist')],
            max_epochs=2,
            train_split=train_split,
        )
        with pytest.raises(ValueError) as exc:
            net.fit(*data)
        msg = "'does-not-exist' is not a valid scoring value."
        assert exc.value.args[0].startswith(msg)

    def test_with_score_as_custom_func(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'score55']
        assert result == [55, 55]

    def test_with_name_none_returns_score_as_name(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score']

    def test_explicit_name_is_used_in_history(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name='myname')],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'myname']

    def test_with_scoring_str_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring='neg_mean_squared_error', name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'neg_mean_squared_error']

    def test_with_with_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55, name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_with_with_partial_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(partial(score55, foo=0), name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_target_extractor_is_called(
            self, net_cls, module_cls, train_split, scoring_cls, data):
        from skorch.utils import to_numpy

        X, y = data
        extractor = Mock(side_effect=to_numpy)
        scoring = scoring_cls(
            name='nmse',
            scoring='neg_mean_squared_error',
            target_extractor=extractor,
        )
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[scoring], max_epochs=2)
        net.fit(X, y)

        assert extractor.call_count == 2

    def test_without_target_data_works(
            self, net_cls, module_cls, scoring_cls, data,
    ):
        def myscore(_, X, y=None):
            assert y is None
            return np.mean(X)

        def mysplit(X, y):
            # set y_valid to None
            return X, X, y, None

        X, y = data
        net = net_cls(
            module=module_cls,
            callbacks=[scoring_cls(myscore)],
            train_split=mysplit,
            max_epochs=2,
        )
        net.fit(X, y)

        expected = np.mean(X)
        loss = net.history[:, 'myscore']
        assert np.allclose(loss, expected)


class TestBatchScoring:
    @pytest.fixture
    def scoring_cls(self):
        from skorch.callbacks import BatchScoring
        return BatchScoring

    @pytest.fixture
    def mse_scoring(self, scoring_cls):
        return scoring_cls(
            name='nmse',
            scoring='neg_mean_squared_error',
        ).initialize()

    @pytest.fixture
    def net(self, net_cls, module_cls, train_split, mse_scoring, data):
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[mse_scoring], max_epochs=2)
        return net.fit(*data)

    @pytest.fixture
    def train_loss(self, scoring_cls):
        from skorch.net import train_loss_score
        return scoring_cls(
            train_loss_score,
            name='train_loss',
            on_train=True,
        ).initialize()

    @pytest.fixture
    def valid_loss(self, scoring_cls):
        from skorch.net import valid_loss_score
        return scoring_cls(
            valid_loss_score,
            name='valid_loss',
        ).initialize()

    @pytest.fixture
    def history(self, net):
        return net.history

    def test_correct_train_loss_values(self, history):
        train_losses = history[:, 'train_loss']
        expected = np.mean([(0 - -1) ** 2, (2 - 0) ** 2])
        assert np.allclose(train_losses, expected)

    def test_correct_valid_loss_values(self, history):
        valid_losses = history[:, 'valid_loss']
        expected = np.mean([(3 - 5) ** 2, (0 - 4) ** 2])
        assert np.allclose(valid_losses, expected)

    def test_correct_mse_values_for_batches(self, history):
        nmse = history[:, 'batches', :, 'nmse']
        expected_per_epoch = [-(3 - 5) ** 2, -(0 - 4) ** 2]
        # for the 2 epochs, the loss is the same
        expected = [expected_per_epoch, expected_per_epoch]
        assert np.allclose(nmse, expected)

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

    @pytest.mark.parametrize('lower_is_better, expected', [
        (True, [True, True, True, False, False]),
        (False, [True, False, False, True, False]),
        (None, []),
    ])
    def test_best_score_when_lower_is_better(
            self, net_cls, module_cls, scoring_cls, train_split, data,
            lower_is_better, expected,
    ):
        # set scoring to None so that mocked net.score is used
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring=None,
                lower_is_better=lower_is_better)],
            train_split=train_split,
            max_epochs=5,
        )
        net.fit(*data)

        if lower_is_better is not None:
            is_best = net.history[:, 'score_best']
            assert is_best == expected
        else:
            # if lower_is_better==None, don't write score
            with pytest.raises(KeyError):
                # pylint: disable=pointless-statement
                net.history[:, 'score_best']

    def test_no_error_when_no_valid_data(
            self, net_cls, module_cls, mse_scoring, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[mse_scoring],
            max_epochs=1,
            train_split=None,
        )
        # does not raise
        net.fit(*data)

    def test_with_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('accuracy')],
            batch_size=1,
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        score_epochs = net.history[:, 'accuracy']
        assert np.allclose(score_epochs, [0, 0])

        score_batches = net.history[:, 'batches', :, 'accuracy']
        assert np.allclose(score_batches, [[0, 0], [0, 0]])

    def test_with_score_nonexisting_string(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('does-not-exist')],
            max_epochs=2,
            train_split=train_split,
        )
        with pytest.raises(ValueError) as exc:
            net.fit(*data)
        msg = "'does-not-exist' is not a valid scoring value."
        assert exc.value.args[0].startswith(msg)

    def test_with_score_as_custom_func(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        score_epochs = net.history[:, 'score55']
        assert np.allclose(score_epochs, [55, 55])

        score_batches = net.history[:, 'batches', :, 'score55']
        assert np.allclose(score_batches, [[55, 55], [55, 55]])

    def test_with_name_none_returns_score_as_name(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score']

    def test_explicit_name_is_used_in_history(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name='myname')],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'myname']

    def test_with_scoring_str_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring='neg_mean_squared_error', name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'neg_mean_squared_error']

    def test_with_with_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55, name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_with_with_partial_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(partial(score55, foo=0), name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_target_extractor_is_called(
            self, net_cls, module_cls, train_split, scoring_cls, data):
        from skorch.utils import to_numpy

        X, y = data
        extractor = Mock(side_effect=to_numpy)
        scoring = scoring_cls(
            name='nmse',
            scoring='neg_mean_squared_error',
            target_extractor=extractor,
        )
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[scoring], max_epochs=2)
        net.fit(X, y)

        assert extractor.call_count == 2 * 2


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
        from skorch.callbacks import EpochScoring
        return EpochScoring

    @pytest.fixture
    def mse_scoring(self, scoring_cls):
        return scoring_cls(
            'neg_mean_squared_error',
            name='nmse',
        ).initialize()

    @pytest.fixture
    def net(self, net_cls, module_cls, train_split, mse_scoring, print_log,
            data):
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[mse_scoring], max_epochs=2)
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
        keys_ignored = ['batches']  # 'dur' no longer ignored
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


class TestCheckpoint:
    @pytest.yield_fixture
    def checkpoint_cls(self):
        from skorch.callbacks import Checkpoint
        return Checkpoint

    @pytest.yield_fixture
    def save_params_mock(self):
        with patch('skorch.NeuralNet.save_params') as mock:
            mock.side_effect = lambda x: x
            yield mock

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 10 epochs"""
        from skorch.net import NeuralNetRegressor
        import torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Linear(1, 1)
            # pylint: disable=arguments-differ
            def forward(self, x):
                return self.p(x)

        return partial(
            NeuralNetRegressor,
            module=Module,
            max_epochs=10,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
        X = np.zeros((10, 1), dtype='float32')
        y = np.zeros((10, 1), dtype='float32')
        return X, y

    def test_none_monitor_saves_always(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        net = net_cls(callbacks=[
            checkpoint_cls(monitor=None),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == len(net.history)

    def test_default_without_validation_raises_meaningful_error(
            self, net_cls, checkpoint_cls, data):
        net = net_cls(
            callbacks=[
                checkpoint_cls(),
            ],
            train_split=None
        )
        from skorch.exceptions import SkorchException
        with pytest.raises(SkorchException) as e:
            net.fit(*data)
            expected = (
                "Monitor value '{}' cannot be found in history. "
                "Make sure you have validation data if you use "
                "validation scores for checkpointing.".format(
                    'valid_loss_best')
            )
            assert str(e.value) == expected

    def test_string_monitor_and_formatting(
            self, save_params_mock, net_cls, checkpoint_cls, data):
        def epoch_3_scorer(net, *_):
            return 1 if net.history[-1, 'epoch'] == 3 else 0

        from skorch.callbacks import EpochScoring
        scoring = EpochScoring(
            scoring=epoch_3_scorer, on_train=True)

        net = net_cls(callbacks=[
            ('my_score', scoring),
            checkpoint_cls(
                monitor='epoch_3_scorer',
                target='model_{last_epoch[epoch]}_{net.max_epochs}.pt'),
        ])
        net.fit(*data)

        assert save_params_mock.call_count == 1
        save_params_mock.assert_called_with('model_3_10.pt')


class TestProgressBar:
    @pytest.yield_fixture
    def progressbar_cls(self):
        from skorch.callbacks import ProgressBar
        return ProgressBar

    @pytest.fixture
    def net_cls(self):
        """very simple network that trains for 2 epochs"""
        from skorch.net import NeuralNetRegressor
        import torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Linear(1, 1)
            # pylint: disable=arguments-differ
            def forward(self, x):
                return self.p(x)

        return partial(
            NeuralNetRegressor,
            module=Module,
            max_epochs=2,
            batch_size=10)

    @pytest.fixture(scope='module')
    def data(self):
        # have 10 examples so we can do a nice CV split
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

    @pytest.mark.parametrize('scheme', [
        'count',
        'auto',
        None,
        2,  # correct number of batches_per_epoch (20 // 10)
        3,  # offset by +1, should still work
        1,  # offset by -1, should still work
    ])
    def test_different_count_schemes(
            self, scheme, net_cls, progressbar_cls, data):
        net = net_cls(callbacks=[
            progressbar_cls(batches_per_epoch=scheme),
        ])
        net.fit(*data)
