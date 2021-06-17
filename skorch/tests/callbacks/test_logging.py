"""Tests for callbacks/logging.py"""

from functools import partial
import os
from unittest.mock import Mock
from unittest.mock import call, patch

import numpy as np
import pytest
import torch
from torch import nn

from skorch.tests.conftest import neptune_installed
from skorch.tests.conftest import sacred_installed
from skorch.tests.conftest import wandb_installed
from skorch.tests.conftest import tensorboard_installed
from skorch.tests.conftest import mlflow_installed


@pytest.mark.skipif(
    not neptune_installed, reason='neptune is not installed')
class TestNeptune:
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
    def neptune_logger_cls(self):
        from skorch.callbacks import NeptuneLogger
        return NeptuneLogger

    @pytest.fixture
    def neptune_experiment_cls(self):
        import neptune
        neptune.init(project_qualified_name="tests/dry-run",
                     backend=neptune.OfflineBackend())
        return neptune.create_experiment

    @pytest.fixture
    def mock_experiment(self, neptune_experiment_cls):
        mock = Mock(spec=neptune_experiment_cls)
        mock.log_metric = Mock()
        mock.stop = Mock()
        return mock

    @pytest.fixture
    def net_fitted(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            mock_experiment,
    ):
        return net_cls(
            classifier_module,
            callbacks=[neptune_logger_cls(mock_experiment)],
            max_epochs=3,
        ).fit(*data)

    def test_experiment_closed_automatically(self, net_fitted, mock_experiment):
        assert mock_experiment.stop.call_count == 1

    def test_experiment_not_closed(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            mock_experiment,
    ):
        net_cls(
            classifier_module,
            callbacks=[
                neptune_logger_cls(mock_experiment, close_after_train=False)],
            max_epochs=2,
        ).fit(*data)
        assert mock_experiment.stop.call_count == 0

    def test_ignore_keys(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            mock_experiment,
    ):
        # ignore 'dur' and 'valid_loss', 'unknown' doesn't exist but
        # this should not cause a problem
        npt = neptune_logger_cls(
            mock_experiment, keys_ignored=['dur', 'valid_loss', 'unknown'])
        net_cls(
            classifier_module,
            callbacks=[npt],
            max_epochs=3,
        ).fit(*data)

        # 3 epochs x 2 epoch metrics = 6 calls
        assert mock_experiment.log_metric.call_count == 6
        call_args = [args[0][0] for args in mock_experiment.log_metric.call_args_list]
        assert 'valid_loss' not in call_args

    def test_keys_ignored_is_string(self, neptune_logger_cls, mock_experiment):
        npt = neptune_logger_cls(
            mock_experiment, keys_ignored='a-key').initialize()
        expected = {'a-key', 'batches'}
        assert npt.keys_ignored_ == expected

    def test_fit_with_real_experiment(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            neptune_experiment_cls,
    ):
        net = net_cls(
            classifier_module,
            callbacks=[neptune_logger_cls(neptune_experiment_cls())],
            max_epochs=5,
        )
        net.fit(*data)

    def test_log_on_batch_level_on(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            mock_experiment,
    ):
        net = net_cls(
            classifier_module,
            callbacks=[neptune_logger_cls(mock_experiment, log_on_batch_end=True)],
            max_epochs=5,
            batch_size=4,
            train_split=False
        )
        net.fit(*data)

        # 5 epochs x (40/4 batches x 2 batch metrics + 2 epoch metrics) = 110 calls
        assert mock_experiment.log_metric.call_count == 110
        mock_experiment.log_metric.assert_any_call('train_batch_size', 4)

    def test_log_on_batch_level_off(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            mock_experiment,
    ):
        net = net_cls(
            classifier_module,
            callbacks=[neptune_logger_cls(mock_experiment, log_on_batch_end=False)],
            max_epochs=5,
            batch_size=4,
            train_split=False
        )
        net.fit(*data)

        # 5 epochs x 2 epoch metrics = 10 calls
        assert mock_experiment.log_metric.call_count == 10
        call_args_list = mock_experiment.log_metric.call_args_list
        assert call('train_batch_size', 4) not in call_args_list

    def test_first_batch_flag(
            self,
            net_cls,
            classifier_module,
            data,
            neptune_logger_cls,
            neptune_experiment_cls,
    ):
        npt = neptune_logger_cls(neptune_experiment_cls())
        npt.initialize()
        assert npt.first_batch_ is True

        net = net_cls(
            classifier_module,
            callbacks=[npt],
            max_epochs=1,
        )

        npt.on_batch_end(net)
        assert npt.first_batch_ is False

@pytest.mark.skipif(
    not sacred_installed, reason='Sacred is not installed')
class TestSacred:
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
    def sacred_logger_cls(self):
        from skorch.callbacks import SacredLogger
        return SacredLogger

    @pytest.fixture
    def sacred_experiment_cls(self):
        from sacred import Experiment
        return Experiment

    @pytest.fixture
    def mock_experiment(self, sacred_experiment_cls):
        mock = Mock(spec=sacred_experiment_cls)
        mock.log_scalar = Mock()
        return mock

    @pytest.fixture
    def net_fitted(
            self,
            net_cls,
            classifier_module,
            data,
            sacred_logger_cls,
            mock_experiment,
    ):
        return net_cls(
            classifier_module,
            callbacks=[sacred_logger_cls(mock_experiment)],
            max_epochs=3,
        ).fit(*data)

    def test_ignore_keys(
            self,
            net_cls,
            classifier_module,
            data,
            sacred_logger_cls,
            mock_experiment,
    ):
        # ignore 'dur' and 'valid_loss', 'unknown' doesn't exist but
        # this should not cause a problem
        logger = sacred_logger_cls(
            mock_experiment, keys_ignored=['dur', 'valid_loss', 'unknown'])
        net_cls(
            classifier_module,
            callbacks=[logger],
            max_epochs=3,
        ).fit(*data)

        # 3 epochs x 2 epoch metrics = 6 calls
        assert mock_experiment.log_scalar.call_count == 6
        call_args = [args[0][0] for args in mock_experiment.log_scalar.call_args_list]
        assert 'valid_loss' not in call_args

    def test_keys_ignored_is_string(self, sacred_logger_cls, mock_experiment):
        logger = sacred_logger_cls(
            mock_experiment, keys_ignored='a-key').initialize()
        expected = {'a-key', 'batches'}
        assert logger.keys_ignored_ == expected

    def test_fit_with_real_experiment(
            self,
            net_cls,
            classifier_module,
            data,
            sacred_logger_cls,
            sacred_experiment_cls,
    ):
        experiment = sacred_experiment_cls()

        @experiment.main
        def experiment_main(_run):
            net = net_cls(
                classifier_module,
                callbacks=[sacred_logger_cls(_run)],
                max_epochs=5,
            )
            net.fit(*data)

        experiment.run()

    def test_log_on_batch_level_on(
            self,
            net_cls,
            classifier_module,
            data,
            sacred_logger_cls,
            mock_experiment,
    ):
        net = net_cls(
            classifier_module,
            callbacks=[sacred_logger_cls(mock_experiment, log_on_batch_end=True)],
            max_epochs=5,
            batch_size=4,
            train_split=False
        )
        net.fit(*data)

        # 5 epochs x (40/4 batches x 2 batch metrics + 2 epoch metrics) = 110 calls
        assert mock_experiment.log_scalar.call_count == 110
        mock_experiment.log_scalar.assert_any_call('train_batch_size_batch', 4)

        logged_keys = [
            call_args.args[0] for call_args in mock_experiment.log_scalar.call_args_list
        ]
        # This is a batch-only metric.
        assert 'train_batch_size_epoch' not in logged_keys

    def test_log_on_batch_level_off(
            self,
            net_cls,
            classifier_module,
            data,
            sacred_logger_cls,
            mock_experiment,
    ):
        net = net_cls(
            classifier_module,
            callbacks=[sacred_logger_cls(mock_experiment, log_on_batch_end=False)],
            max_epochs=5,
            batch_size=4,
            train_split=False
        )
        net.fit(*data)

        # 5 epochs x 2 epoch metrics = 10 calls
        assert mock_experiment.log_scalar.call_count == 10
        call_args_list = mock_experiment.log_scalar.call_args_list
        assert call('train_batch_size_batch', 4) not in call_args_list

@pytest.mark.skipif(
    not wandb_installed, reason='wandb is not installed')
class TestWandb:
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
    def wandb_logger_cls(self):
        from skorch.callbacks import WandbLogger
        return WandbLogger

    @pytest.fixture
    def wandb_run_cls(self):
        import wandb
        os.environ['WANDB_MODE'] = 'dryrun' # run offline
        wandb_version = tuple(map(int, wandb.__version__.split('.')[:2]))
        if wandb_version >= (0, 10):
            return wandb.init(anonymous="allow")
        else:
            with wandb.init(anonymous="allow") as run:
                return run

    @pytest.fixture
    def mock_run(self):
        mock = Mock()
        mock.log = Mock()
        mock.watch = Mock()
        mock.dir = '.'
        return mock

    def test_ignore_keys(
            self,
            net_cls,
            classifier_module,
            data,
            wandb_logger_cls,
            mock_run,
    ):
        # ignore 'dur' and 'valid_loss', 'unknown' doesn't exist but
        # this should not cause a problem
        wandb_cb = wandb_logger_cls(
            mock_run, keys_ignored=['dur', 'valid_loss', 'unknown'])
        net_cls(
            classifier_module,
            callbacks=[wandb_cb],
            max_epochs=3,
        ).fit(*data)

        # 3 epochs = 3 calls
        assert mock_run.log.call_count == 3
        assert mock_run.watch.call_count == 1
        call_args = [args[0][0] for args in mock_run.log.call_args_list]
        assert 'valid_loss' not in call_args

    def test_keys_ignored_is_string(self, wandb_logger_cls, mock_run):
        wandb_cb = wandb_logger_cls(
            mock_run, keys_ignored='a-key').initialize()
        expected = {'a-key', 'batches'}
        assert wandb_cb.keys_ignored_ == expected

    def test_fit_with_real_experiment(
            self,
            net_cls,
            classifier_module,
            data,
            wandb_logger_cls,
            wandb_run_cls,
    ):
        net = net_cls(
            classifier_module,
            callbacks=[wandb_logger_cls(wandb_run_cls)],
            max_epochs=5,
        )
        net.fit(*data)

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
        assert len(odd_row) == 6  # odd row has entries in every column
        assert odd_row[4] == '+'  # including '+' sign for the 'event_odd'
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

    def test_pickle(self, net_cls, progressbar_cls, data):
        # pickling was an issue since TQDM progress bar instances cannot
        # be pickled. Test pickling and restoration.
        import pickle

        net = net_cls(callbacks=[
            progressbar_cls(),
        ])
        net.fit(*data)
        dump = pickle.dumps(net)

        net = pickle.loads(dump)
        net.fit(*data)


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
        X_dict = {k: X[:, i:i + 4] for k, i in zip('cebad', range(0, X.shape[1], 4))}

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


@pytest.mark.skipif(
    not mlflow_installed, reason='mlflow is not installed')
class TestMLflowLogger:
    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def logger_cls(self):
        from skorch.callbacks import MlflowLogger
        return MlflowLogger

    @pytest.fixture
    def data(self, classifier_data):
        X, y = classifier_data
        X, y = X[:40], y[:40]
        return X, y

    @pytest.fixture
    def mock_run(self):
        from mlflow.entities import Run
        return Mock(Run)

    @pytest.fixture
    def mock_client(self):
        from mlflow.tracking import MlflowClient
        return Mock(MlflowClient)

    @pytest.fixture
    def logger_mock_cls(self, logger_cls, mock_run, mock_client):
        return partial(logger_cls, mock_run, mock_client)

    @pytest.fixture
    def net_builder_cls(self, net_cls, classifier_module, data):
        def builder(*args, **kwargs):
            return net_cls(classifier_module, *args, **kwargs).fit(*data)
        return builder

    @pytest.fixture
    def net_fitted(self, logger_mock_cls, net_builder_cls):
        return net_builder_cls(callbacks=[logger_mock_cls()], max_epochs=3)

    def test_run_default(self, monkeypatch, logger_cls, mock_run, mock_client):
        import mlflow
        mock_active_run = Mock(mlflow.active_run, return_value=mock_run)
        monkeypatch.setattr(mlflow, 'active_run', mock_active_run)
        logger = logger_cls(client=mock_client).initialize()
        assert mock_active_run.called
        assert logger.run_ == mock_run

    def test_client_default(self, monkeypatch, logger_cls, mock_run, mock_client):
        import mlflow.tracking
        monkeypatch.setattr(mlflow.tracking, 'MlflowClient', mock_client)
        logger = logger_cls(run=mock_run).initialize()
        assert mock_client.called
        assert logger.client_ == mock_client()

    def test_keys_from_history_logged(self, net_fitted, mock_client):
        assert mock_client.log_metric.call_count == 3 * 4
        keys = {call_args[0][1] for call_args in mock_client.log_metric.call_args_list}
        expected = {'dur', 'train_loss', 'valid_loss', 'valid_acc'}
        assert keys == expected

    def test_ignore_keys(self, logger_mock_cls, net_builder_cls):
        # ignore 'dur' and 'valid_loss', 'unknown' doesn't exist but
        # this should not cause a problem
        logger = logger_mock_cls(keys_ignored=['dur', 'valid_loss', 'unknown'])
        net_builder_cls(callbacks=[logger], max_epochs=3)
        keys = {
            call_args[0][1]
            for call_args in logger.client_.log_metric.call_args_list
        }
        expected = {'train_loss', 'valid_acc'}
        assert keys == expected

    def test_keys_ignored_is_string(self, logger_mock_cls):
        logger = logger_mock_cls(keys_ignored='a-key').initialize()
        expected = {'a-key', 'batches'}
        assert logger.keys_ignored_ == expected

    @pytest.mark.parametrize(
        'log_on_batch_end, log_on_epoch_end, batch_suffix, epoch_suffix',
        [(False, False, '', ''),
         (True, False, '', ''),
         (False, True, '', ''),
         (True, True, '_batch', '_epoch')]
    )
    def test_epoch_batch_suffixes_defaults(
            self,
            logger_mock_cls,
            log_on_batch_end,
            log_on_epoch_end,
            batch_suffix,
            epoch_suffix,
    ):
        logger = logger_mock_cls(
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end
        ).initialize()
        assert logger.batch_suffix_ == batch_suffix
        assert logger.epoch_suffix_ == epoch_suffix

    @pytest.mark.parametrize('batch_suffix', ['', '_foo'])
    @pytest.mark.parametrize('epoch_suffix', ['', '_bar'])
    def test_epoch_batch_custom_suffix(
            self,
            logger_mock_cls,
            batch_suffix,
            epoch_suffix,
    ):
        logger = logger_mock_cls(
            log_on_batch_end=True,
            log_on_epoch_end=True,
            batch_suffix=batch_suffix,
            epoch_suffix=epoch_suffix,
        ).initialize()
        assert logger.batch_suffix_ == batch_suffix
        assert logger.epoch_suffix_ == epoch_suffix

    def test_dont_log_epoch_metrics(self, logger_mock_cls, net_builder_cls):
        logger = logger_mock_cls(
            log_on_batch_end=True,
            log_on_epoch_end=False,
            batch_suffix='_batch',
            epoch_suffix='_epoch',
        )
        net_builder_cls(batch_size=10, callbacks=[logger], max_epochs=3)
        assert all(
            call[0][1].endswith('_batch')
            for call in logger.client_.log_metric.call_args_list
        )

    def test_log_epochs_with_step(self, net_fitted, mock_client):
        expected = [x for x in range(1, 4) for _ in range(4)]
        actual = [call[1].get('step') for call in mock_client.log_metric.call_args_list]
        assert expected == actual

    def test_log_batch_with_step(self, logger_mock_cls, net_builder_cls):
        logger = logger_mock_cls(log_on_batch_end=True, log_on_epoch_end=False)
        net_builder_cls(batch_size=10, callbacks=[logger], max_epochs=4)
        expected = [x for x in range(1, 21) for _ in range(2)]
        actual = [
            call[1].get('step')
            for call in logger.client_.log_metric.call_args_list
        ]
        assert expected == actual

    def test_artifact_filenames(self, net_fitted, mock_client):
        keys = {call_args[0][1].name
                for call_args in mock_client.log_artifact.call_args_list}
        expected = {'params.pth', 'optimizer.pth', 'criterion.pth', 'history.json'}
        assert keys == expected

    def test_artifact_in_temporary_directory(self, net_fitted, mock_client):
        for call_args in mock_client.log_artifact.call_args_list:
            assert str(call_args[0][1]).startswith('/tmp')

    def test_dont_create_artifact(self, net_builder_cls, logger_mock_cls):
        logger = logger_mock_cls(create_artifact=False)
        net_builder_cls(callbacks=[logger], max_epochs=3)
        assert not logger.client_.log_artifact.called

    def test_run_terminated_automatically(self, net_fitted, mock_client):
        assert mock_client.set_terminated.call_count == 1

    def test_run_not_closed(self, logger_mock_cls, mock_client, net_builder_cls):
        logger = logger_mock_cls(terminate_after_train=False)
        net_builder_cls(callbacks=[logger], max_epochs=2)
        assert logger.client_.set_terminated.call_count == 0

    def test_fit_with_real_run_and_client(self, tmp_path, logger_cls, net_builder_cls):
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tmp_path.as_uri())
        experiment_name = 'foo'
        experiment_id = client.create_experiment(experiment_name)
        run = client.create_run(experiment_id)
        logger = logger_cls(run, client, create_artifact=False)
        net_builder_cls(callbacks=[logger], max_epochs=3)
        assert os.listdir(tmp_path)
