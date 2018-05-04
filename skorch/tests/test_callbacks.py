"""Tests for callbacks.py"""

from functools import partial
import itertools
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from skorch.utils import to_numpy


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
    def base_cls(self):
        from skorch.callbacks import Callback
        return Callback

    @pytest.fixture
    def on_x_methods(self):
        return [
            'on_train_begin',
            'on_train_end',
            'on_epoch_begin',
            'on_epoch_end',
            'on_batch_begin',
            'on_batch_end',
            'on_grad_computed',
        ]

    def test_on_x_methods_have_kwargs(self, callbacks, on_x_methods):
        import inspect
        for callback, method_name in itertools.product(
                callbacks, on_x_methods):
            method = getattr(callback, method_name)
            assert "kwargs" in inspect.signature(method).parameters

    def test_set_params_with_unknown_key_raises(self, base_cls):
        with pytest.raises(ValueError) as exc:
            base_cls().set_params(foo=123)

        # TODO: check error message more precisely, depending on what
        # the intended message shouldb e from sklearn side
        assert exc.value.args[0].startswith('Invalid parameter foo for')


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


class TestGradientNormClipping:
    @pytest.yield_fixture
    def grad_clip_cls_and_mock(self):
        with patch('skorch.callbacks.regularization.clip_grad_norm_') as cgn:
            from skorch.callbacks import GradientNormClipping
            yield GradientNormClipping, cgn

    def test_parameters_passed_correctly_to_torch_cgn(
            self, grad_clip_cls_and_mock):
        grad_norm_clip_cls, cgn = grad_clip_cls_and_mock

        clipping = grad_norm_clip_cls(
            gradient_clip_value=55, gradient_clip_norm_type=99)
        named_parameters = [('p1', 1), ('p2', 2), ('p3', 3)]
        parameter_values = [p for _, p in named_parameters]
        clipping.on_grad_computed(None, named_parameters=named_parameters)

        # Clip norm must receive values, not (name, value) pairs.
        assert list(cgn.call_args_list[0][0][0]) == parameter_values
        assert cgn.call_args_list[0][1]['max_norm'] == 55
        assert cgn.call_args_list[0][1]['norm_type'] == 99

    def test_no_parameter_updates_when_norm_0(
            self, classifier_module, classifier_data):
        from copy import deepcopy
        from skorch import NeuralNetClassifier
        from skorch.callbacks import GradientNormClipping

        net = NeuralNetClassifier(
            classifier_module,
            callbacks=[('grad_norm', GradientNormClipping(0))],
            train_split=None,
            warm_start=True,
            max_epochs=1,
        )
        net.initialize()

        params_before = deepcopy(list(net.module_.parameters()))
        net.fit(*classifier_data)
        params_after = net.module_.parameters()
        for p0, p1 in zip(params_before, params_after):
            p0, p1 = to_numpy(p0), to_numpy(p1)
            assert np.allclose(p0, p1)
