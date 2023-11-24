""" Callbacks for printing, logging and log information."""

import sys
import time
import tempfile
from contextlib import suppress
from numbers import Number
from itertools import cycle
from pathlib import Path

import numpy as np
import tqdm
from tabulate import tabulate

from skorch.utils import Ansi
from skorch.dataset import get_len
from skorch.callbacks import Callback

__all__ = ['EpochTimer', 'NeptuneLogger', 'WandbLogger', 'PrintLog', 'ProgressBar',
           'TensorBoard', 'SacredLogger', 'MlflowLogger']


def filter_log_keys(keys, keys_ignored=None):
    """Filter out keys that are generally to be ignored.

    This is used by several callbacks to filter out keys from history
    that should not be logged.

    Parameters
    ----------
    keys : iterable of str
      All keys.

    keys_ignored : iterable of str or None (default=None)
      If not None, collection of extra keys to be ignored.

    """
    keys_ignored = keys_ignored or ()
    for key in keys:
        if not (
                key == 'epoch' or
                (key in keys_ignored) or
                key.endswith('_best') or
                key.endswith('_batch_count') or
                key.startswith('event_')
        ):
            yield key


class EpochTimer(Callback):
    """Measures the duration of each epoch and writes it to the
    history with the name ``dur``.

    """
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', time.time() - self.epoch_start_time_)


class NeptuneLogger(Callback):
    """Logs model metadata and training metrics to Neptune.

    Neptune is a lightweight experiment-tracking tool.
    You can read more about it here: https://neptune.ai

    Use this callback to automatically log all interesting values from
    your net's history to Neptune.

    The best way to log additional information is to log directly to the
    run object.

    To monitor resource consumption, install psutil:

    $ python -m pip install psutil

    You can view example experiment logs here:
    https://app.neptune.ai/o/common/org/skorch-integration/e/SKOR-32/all

    Examples
    --------
    $ # Install Neptune
    $ python -m pip install neptune

    >>> # Create a Neptune run
    >>> import neptune
    >>> from neptune.types import File
    >>> # This example uses the API token for anonymous users.
    >>> # For your own projects, use the token associated with your neptune.ai account.
    >>> run = neptune.init_run(
    ...     api_token=neptune.ANONYMOUS_API_TOKEN,
    ...     project='shared/skorch-integration',
    ...     name='skorch-basic-example',
    ...     source_files=['skorch_example.py'],
    ... )

    >>> # Create a NeptuneLogger callback
    >>> neptune_logger = NeptuneLogger(run, close_after_train=False)

    >>> # Pass the logger to the net callbacks argument
    >>> net = NeuralNetClassifier(
    ...           ClassifierModule,
    ...           max_epochs=20,
    ...           lr=0.01,
    ...           callbacks=[neptune_logger, Checkpoint(dirname="./checkpoints")])
    >>> net.fit(X, y)

    >>> # Save the checkpoints to Neptune
    >>> neptune_logger.run["checkpoints"].upload_files("./checkpoints")

    >>> # Log additional metrics after training has finished
    >>> from sklearn.metrics import roc_auc_score
    >>> y_proba = net.predict_proba(X)
    >>> auc = roc_auc_score(y, y_proba[:, 1])
    >>> neptune_logger.run["roc_auc_score"].log(auc)

    >>> # Log charts, such as an ROC curve
    >>> from sklearn.metrics import RocCurveDisplay
    >>> roc_plot = RocCurveDisplay.from_estimator(net, X, y)
    >>> neptune_logger.run["roc_curve"].upload(File.as_html(roc_plot.figure_))

    >>> # Log the net object after training
    >>> net.save_params(f_params='basic_model.pkl')
    >>> neptune_logger.run["basic_model"].upload(File('basic_model.pkl'))

    >>> # Close the run
    >>> neptune_logger.run.stop()

    Parameters
    ----------
    run : neptune.Run or neptune.handler.Handler
      Instantiated ``Run`` or ``Handler`` class.

    log_on_batch_end : bool (default=False)
      Whether to log loss and other metrics on batch level.

    close_after_train : bool (default=True)
      Whether to close the ``Run`` object once training
      finishes. Set this parameter to False if you want to continue
      logging to the same run or if you use it as a context
      manager.

    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be logged to Neptune. Note that in
      addition to the keys provided by the user, keys such as those starting
      with ``'event_'`` or ending on ``'_best'`` are ignored by default.

    base_namespace: str
      Namespace (folder) under which all metadata logged by the ``NeptuneLogger``
      will be stored. Defaults to "training".

    Attributes
    ----------
    .. _Neptune: https://www.neptune.ai

    """

    def __init__(
            self,
            run,
            *,
            log_on_batch_end=False,
            close_after_train=True,
            keys_ignored=None,
            base_namespace='training',
    ):
        self.run = run
        self.log_on_batch_end = log_on_batch_end
        self.close_after_train = close_after_train
        self.keys_ignored = keys_ignored
        self.base_namespace = base_namespace

    def _log_integration_version(self) -> None:
        from skorch import __version__

        self.run['source_code/integrations/skorch'] = __version__

    @property
    def _metric_logger(self):
        return self.run[self._base_namespace]

    @staticmethod
    def _get_obj_name(obj):
        return type(obj).__name__

    def initialize(self):
        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')

        if self.base_namespace.endswith("/"):
            self._base_namespace = self.base_namespace[:-1]
        else:
            self._base_namespace = self.base_namespace

        self._log_integration_version()

        return self

    def on_train_begin(self, net, X, y, **kwargs):
        # TODO: we might want to improve logging of the multi-module net objects, see:
        #       https://github.com/skorch-dev/skorch/pull/906#discussion_r993514643

        self._metric_logger['model/model_type'] = self._get_obj_name(net.module_)
        self._metric_logger['model/summary'] = self._model_summary_file(net.module_)

        self._metric_logger['config/optimizer'] = self._get_obj_name(net.optimizer_)
        self._metric_logger['config/criterion'] = self._get_obj_name(net.criterion_)
        self._metric_logger['config/lr'] = net.lr
        self._metric_logger['config/epochs'] = net.max_epochs
        self._metric_logger['config/batch_size'] = net.batch_size
        self._metric_logger['config/device'] = net.device

    def on_batch_end(self, net, **kwargs):
        if self.log_on_batch_end:
            batch_logs = net.history[-1]['batches'][-1]

            for key in filter_log_keys(batch_logs.keys(), self.keys_ignored_):
                self._log_metric(key, batch_logs, batch=True)

    def on_epoch_end(self, net, **kwargs):
        """Automatically log values from the last history step."""
        epoch_logs = net.history[-1]

        for key in filter_log_keys(epoch_logs.keys(), self.keys_ignored_):
            self._log_metric(key, epoch_logs, batch=False)

    def on_train_end(self, net, **kwargs):
        try:
            self._metric_logger['train/epoch/event_lr'].append(net.history[:, 'event_lr'])
        except KeyError:
            pass
        if self.close_after_train:
            try:  # >1.0 package structure
                from neptune.handler import Handler
            except ImportError:  # <1.0 package structure
                from neptune.new.handler import Handler

            # Neptune integrations now accept passing Handler object
            # to an integration.
            # Ref: https://docs.neptune.ai/api/field_types/#handler
            # Example of getting an handler from a `Run` object.
            # handler = run["foo"]
            # handler['bar'] = 1  # Logs to `foo/bar`
            # NOTE: Handler provides most of the functionality of `Run`
            # for logging, however it doesn't implement a few methods like
            # `stop`, `wait`, etc.
            root_obj = self.run
            if isinstance(self.run, Handler):
                root_obj = self.run.get_root_object()

            root_obj.stop()

    def _log_metric(self, name, logs, batch):
        kind, _, key = name.partition('_')

        if not key:
            key = 'epoch_duration' if kind == 'dur' else kind
            self._metric_logger[key].append(logs[name])
        else:
            if kind == 'valid':
                kind = 'validation'

            if batch:
                granularity = 'batch'
            else:
                granularity = 'epoch'

            # for example:     train /   epoch   / loss
            self._metric_logger[kind][granularity][key].append(logs[name])

    @staticmethod
    def _model_summary_file(model):
        try:
            # neptune-client>=1.0.0 package structure
            from neptune.types import File
        except ImportError:
            # neptune-client=0.9.0+ package structure
            from neptune.new.types import File

        return File.from_content(str(model), extension='txt')


class WandbLogger(Callback):
    """Logs best model and metrics to `Weights & Biases <https://docs.wandb.com/>`_

    Use this callback to automatically log best trained model, all metrics from
    your net's history, model topology and computer resources to Weights & Biases
    after each epoch.

    Every file saved in `wandb_run.dir` is automatically logged to W&B servers.

    See `example run
    <https://app.wandb.ai/borisd13/skorch/runs/s20or4ct/overview?workspace=user-borisd13>`_

    Examples
    --------
    >>> # Install wandb
    ... python -m pip install wandb

    >>> import wandb
    >>> from skorch.callbacks import WandbLogger

    >>> # Create a wandb Run
    ... wandb_run = wandb.init()
    >>> # Alternative: Create a wandb Run without having a W&B account
    ... wandb_run = wandb.init(anonymous="allow)

    >>> # Log hyper-parameters (optional)
    ... wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

    >>> net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
    >>> net.fit(X, y)

    Parameters
    ----------
    wandb_run : wandb.wandb_run.Run
      wandb Run used to log data.

    save_model : bool (default=True)
      Whether to save a checkpoint of the best model and upload it
      to your Run on W&B servers.

    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be logged to wandb. Note that in
      addition to the keys provided by the user, keys such as those starting
      with ``'event_'`` or ending on ``'_best'`` are ignored by default.

    """

    def __init__(
            self,
            wandb_run,
            save_model=True,
            keys_ignored=None,
    ):
        self.wandb_run = wandb_run
        self.save_model = save_model
        self.keys_ignored = keys_ignored

    def initialize(self):
        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def on_train_begin(self, net, **kwargs):
        """Log model topology and add a hook for gradients"""
        self.wandb_run.watch(net.module_)

    def on_epoch_end(self, net, **kwargs):
        """Log values from the last history step and save best model"""
        hist = net.history[-1]
        keys_kept = filter_log_keys(hist, keys_ignored=self.keys_ignored_)
        logged_vals = {k: hist[k] for k in keys_kept}
        self.wandb_run.log(logged_vals)

        # save best model
        if self.save_model and hist['valid_loss_best']:
            model_path = Path(self.wandb_run.dir) / 'best_model.pth'
            with model_path.open('wb') as model_file:
                net.save_params(f_params=model_file)


class PrintLog(Callback):
    """Print useful information from the model's history as a table.

    By default, ``PrintLog`` prints everything from the history except
    for ``'batches'``.

    To determine the best loss, ``PrintLog`` looks for keys that end on
    ``'_best'`` and associates them with the corresponding loss. E.g.,
    ``'train_loss_best'`` will be matched with ``'train_loss'``. The
    :class:`skorch.callbacks.EpochScoring` callback takes care of
    creating those entries, which is why ``PrintLog`` works best in
    conjunction with that callback.

    ``PrintLog`` treats keys with the ``'event_'`` prefix in a special
    way. They are assumed to contain information about occasionally
    occuring events. The ``False`` or ``None`` entries (indicating
    that an event did not occur) are not printed, resulting in empty
    cells in the table, and ``True`` entries are printed with ``+``
    symbol. ``PrintLog`` groups all event columns together and pushes
    them to the right, just before the ``'dur'`` column.

    *Note*: ``PrintLog`` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.

    Parameters
    ----------
    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be part of the printed table. Note
      that in addition to the keys provided by the user, keys such as those
      starting with ``'event_'`` or ending on ``'_best'`` are ignored by
      default.

    sink : callable (default=print)
      The target that the output string is sent to. By default, the
      output is printed to stdout, but the sink could also be a
      logger, etc.

    tablefmt : str (default='simple')
      The format of the table. See the documentation of the ``tabulate``
      package for more detail. Can be 'plain', 'grid', 'pipe', 'html',
      'latex', among others.

    floatfmt : str (default='.4f')
      The number formatting. See the documentation of the ``tabulate``
      package for more details.

    stralign : str (default='right')
      The alignment of columns with strings. Can be 'left', 'center',
      'right', or ``None`` (disable alignment). Default is 'right' (to
      be consistent with numerical columns).

    """
    def __init__(
            self,
            keys_ignored=None,
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
            stralign='right',
    ):
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign

    def initialize(self):
        self.first_iteration_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable).

        """
        value = row[key]

        if isinstance(value, bool) or value is None:
            return '+' if value else ''

        if not isinstance(value, Number):
            return value

        # determine if integer value
        is_integer = float(value).is_integer()
        template = '{}' if is_integer else '{:' + self.floatfmt + '}'

        # if numeric, there could be a 'best' key
        key_best = key + '_best'
        if (key_best in row) and row[key_best]:
            template = color + template + Ansi.ENDC.value
        return template.format(value)

    def _sorted_keys(self, keys):
        """Sort keys, dropping the ones that should be ignored.

        The keys that are in ``self.ignored_keys`` or that end on
        '_best' are dropped. Among the remaining keys:
          * 'epoch' is put first;
          * 'dur' is put last;
          * keys that start with 'event_' are put just before 'dur';
          * all remaining keys are sorted alphabetically.
        """
        sorted_keys = []

        # make sure 'epoch' comes first
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
            sorted_keys.append('epoch')

        # ignore keys like *_best or event_*
        for key in filter_log_keys(sorted(keys), keys_ignored=self.keys_ignored_):
            if key != 'dur':
                sorted_keys.append(key)

        # add event_* keys
        for key in sorted(keys):
            if key.startswith('event_') and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # make sure 'dur' comes last
        if ('dur' in keys) and ('dur' not in self.keys_ignored_):
            sorted_keys.append('dur')

        return sorted_keys

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        for key, color in zip(self._sorted_keys(row.keys()), colors):
            formatted = self.format_row(row, key, color=color)
            if key.startswith('event_'):
                key = key[6:]
            yield key, formatted

    def table(self, row):
        headers = []
        formatted = []
        for key, formatted_row in self._yield_keys_formatted(row):
            headers.append(key)
            formatted.append(formatted_row)

        return tabulate(
            [formatted],
            headers=headers,
            tablefmt=self.tablefmt,
            floatfmt=self.floatfmt,
            stralign=self.stralign,
        )

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        data = net.history[-1]
        verbose = net.verbose
        tabulated = self.table(data)

        if self.first_iteration_:
            header, lines = tabulated.split('\n', 2)[:2]
            self._sink(header, verbose)
            self._sink(lines, verbose)
            self.first_iteration_ = False

        self._sink(tabulated.rsplit('\n', 1)[-1], verbose)
        if self.sink is print:
            sys.stdout.flush()


class ProgressBar(Callback):
    """Display a progress bar for each epoch.

    The progress bar includes elapsed and estimated remaining time for
    the current epoch, the number of batches processed, and other
    user-defined metrics. The progress bar is erased once the epoch is
    completed.

    ``ProgressBar`` needs to know the total number of batches per
    epoch in order to display a meaningful progress bar. By default,
    this number is determined automatically using the dataset length
    and the batch size. If this heuristic does not work for some
    reason, you may either specify the number of batches explicitly
    or let the ``ProgressBar`` count the actual number of batches in
    the previous epoch.

    For jupyter notebooks a non-ASCII progress bar can be printed
    instead. To use this feature, you need to have `ipywidgets
    <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`_
    installed.

    Parameters
    ----------

    batches_per_epoch : int, str (default='auto')
      Either a concrete number or a string specifying the method used
      to determine the number of batches per epoch automatically.
      ``'auto'`` means that the number is computed from the length of
      the dataset and the batch size. ``'count'`` means that the
      number is determined by counting the batches in the previous
      epoch. Note that this will leave you without a progress bar at
      the first epoch.

    detect_notebook : bool (default=True)
      If enabled, the progress bar determines if its current environment
      is a jupyter notebook and switches to a non-ASCII progress bar.

    postfix_keys : list of str (default=['train_loss', 'valid_loss'])
      You can use this list to specify additional info displayed in the
      progress bar such as metrics and losses. A prerequisite to this is
      that these values are residing in the history on batch level already,
      i.e. they must be accessible via

      >>> net.history[-1, 'batches', -1, key]
    """
    def __init__(
            self,
            batches_per_epoch='auto',
            detect_notebook=True,
            postfix_keys=None
    ):
        self.batches_per_epoch = batches_per_epoch
        self.detect_notebook = detect_notebook
        self.postfix_keys = postfix_keys or ['train_loss', 'valid_loss']

    def in_ipynb(self):
        try:
            return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except NameError:
            return False

    def _use_notebook(self):
        return self.in_ipynb() if self.detect_notebook else False

    def _get_batch_size(self, net, training):
        name = 'iterator_train' if training else 'iterator_valid'
        net_params = net.get_params()
        return net_params.get(name + '__batch_size', net_params['batch_size'])

    def _get_batches_per_epoch_phase(self, net, dataset, training):
        if dataset is None:
            return 0
        batch_size = self._get_batch_size(net, training)
        return int(np.ceil(get_len(dataset) / batch_size))

    def _get_batches_per_epoch(self, net, dataset_train, dataset_valid):
        return (self._get_batches_per_epoch_phase(net, dataset_train, True) +
                self._get_batches_per_epoch_phase(net, dataset_valid, False))

    def _get_postfix_dict(self, net):
        postfix = {}
        for key in self.postfix_keys:
            try:
                postfix[key] = net.history[-1, 'batches', -1, key]
            except KeyError:
                pass
        return postfix

    # pylint: disable=attribute-defined-outside-init
    def on_batch_end(self, net, **kwargs):
        self.pbar_.set_postfix(self._get_postfix_dict(net), refresh=False)
        self.pbar_.update()

    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # Assume it is a number until proven otherwise.
        batches_per_epoch = self.batches_per_epoch

        if self.batches_per_epoch == 'auto':
            batches_per_epoch = self._get_batches_per_epoch(
                net, dataset_train, dataset_valid
            )
        elif self.batches_per_epoch == 'count':
            if len(net.history) <= 1:
                # No limit is known until the end of the first epoch.
                batches_per_epoch = None
            else:
                batches_per_epoch = len(net.history[-2, 'batches'])

        if self._use_notebook():
            self.pbar_ = tqdm.tqdm_notebook(total=batches_per_epoch, leave=False)
        else:
            self.pbar_ = tqdm.tqdm(total=batches_per_epoch, leave=False)

    def on_epoch_end(self, net, **kwargs):
        self.pbar_.close()

    def __getstate__(self):
        # don't save away the temporary pbar_ object which gets created on
        # epoch begin anew anyway. This avoids pickling errors with tqdm.
        state = self.__dict__.copy()
        state.pop('pbar_', None)
        return state


def rename_tensorboard_key(key):
    """Rename keys from history to keys in TensorBoard

    Specifically, prefixes all names with "Loss/" if they seem to be
    losses.

    """
    if key.startswith('train') or key.startswith('valid'):
        key = 'Loss/' + key
    return key


class TensorBoard(Callback):
    """Logs results from history to TensorBoard

    "TensorBoard provides the visualization and tooling needed for machine
    learning experimentation" (`offical docs
    <https://www.tensorflow.org/tensorboard/>`_).

    Use this callback to automatically log all interesting values from your
    net's history to tensorboard after each epoch.

    Examples
    --------
    Here is the standard way of using the callback:

    >>> # Example: normal usage
    >>> from skorch.callbacks import TensorBoard
    >>> from torch.utils.tensorboard import SummaryWriter
    >>> writer = SummaryWriter(...)
    >>> net = NeuralNet(..., callbacks=[TensorBoard(writer)])
    >>> net.fit(X, y)

    The best way to log additional information is to subclass this
    callback and add your code to one of the ``on_*`` methods.

    >>> # Example: log the bias parameter as a histogram
    >>> def extract_bias(module):
    ...     return module.hidden.bias
    >>> # override on_epoch_end
    >>> class MyTensorBoard(TensorBoard):
    ...     def on_epoch_end(self, net, **kwargs):
    ...         bias = extract_bias(net.module_)
    ...         epoch = net.history[-1, 'epoch']
    ...         self.writer.add_histogram('bias', bias, global_step=epoch)
    ...         super().on_epoch_end(net, **kwargs)  # call super last
    >>> # other code
    >>> net = NeuralNet(..., callbacks=[MyTensorBoard(writer)])

    Parameters
    ----------
    writer : torch.utils.tensorboard.writer.SummaryWriter
      Instantiated ``SummaryWriter`` class.

    close_after_train : bool (default=True)
      Whether to close the ``SummaryWriter`` object once training
      finishes. Set this parameter to False if you want to continue
      logging with the same writer or if you use it as a context
      manager.

    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be logged to tensorboard. Note that in
      addition to the keys provided by the user, keys such as those starting
      with ``'event_'`` or ending on ``'_best'`` are ignored by default.

    key_mapper : callable or function (default=rename_tensorboard_key)
      This function maps a key name from the history to a tag in
      tensorboard. This is useful because tensorboard can
      automatically group similar tags if their names start with the
      same prefix, followed by a forward slash. By default, this
      callback will prefix all keys that start with "train" or "valid"
      with the "Loss/" prefix.

    """
    def __init__(
            self,
            writer,
            close_after_train=True,
            keys_ignored=None,
            key_mapper=rename_tensorboard_key,
    ):
        self.writer = writer
        self.close_after_train = close_after_train
        self.keys_ignored = keys_ignored
        self.key_mapper = key_mapper

    def initialize(self):
        self.first_batch_ = True

        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        return self

    def on_batch_end(self, net, **kwargs):
        self.first_batch_ = False

    def add_scalar_maybe(self, history, key, tag, global_step=None):
        """Add a scalar value from the history to TensorBoard

        Will catch errors like missing keys or wrong value types.

        Parameters
        ----------
        history : skorch.History
          History object saved as attribute on the neural net.

        key : str
          Key of the desired value in the history.

        tag : str
          Name of the tag used in TensorBoard.

        global_step : int or None
          Global step value to record.

        """
        hist = history[-1]
        val = hist.get(key)
        if val is None:
            return

        global_step = global_step if global_step is not None else hist['epoch']
        with suppress(NotImplementedError):
            # pytorch raises NotImplementedError on wrong types
            self.writer.add_scalar(
                tag=tag,
                scalar_value=val,
                global_step=global_step,
            )

    def on_epoch_end(self, net, **kwargs):
        """Automatically log values from the last history step."""
        history = net.history
        hist = history[-1]
        epoch = hist['epoch']

        for key in filter_log_keys(hist, keys_ignored=self.keys_ignored_):
            tag = self.key_mapper(key)
            self.add_scalar_maybe(history, key=key, tag=tag, global_step=epoch)

    def on_train_end(self, net, **kwargs):
        if self.close_after_train:
            self.writer.close()


class SacredLogger(Callback):
    """Logs results from history to Sacred.

    Sacred is a tool to help you configure, organize, log and reproduce
    experiments. Developed at IDSIA. See https://github.com/IDSIA/sacred.

    Use this callback to automatically log all interesting values from
    your net's history to Sacred.

    If you want to log additional information, you can simply add it to
    ``History``. See the documentation on ``Callbacks``, and ``Scoring`` for
    more information. Alternatively you can subclass this callback and extend
    the ``on_*`` methods.

    To use this logger, you first have to install Sacred:

    .. code-block:: bash

        python -m pip install sacred

    You might also install pymongo to use a mongodb backend. See the `upstream
    documentation <https://github.com/IDSIA/sacred#installing>`_ for more
    details. Once you have installed it, you can set up a simple experiment and
    pass this Logger as a callback to your skorch estimator:

    Examples
    --------
    >>> # contents of sacred-experiment.py
    >>> import numpy as np
    >>> from sacred import Experiment
    >>> from sklearn.datasets import make_classification
    >>> from skorch.callbacks.logging import SacredLogger
    >>> from skorch.callbacks.scoring import EpochScoring
    >>> from skorch import NeuralNetClassifier
    >>> from skorch.toy import make_classifier
    >>> ex = Experiment()
    >>> @ex.config
    >>> def my_config():
    ...     max_epochs = 20
    ...     lr = 0.01
    >>> X, y = make_classification()
    >>> X, y = X.astype(np.float32), y.astype(np.int64)
    >>> @ex.automain
    >>> def main(_run, max_epochs, lr):
    ...     # Take care to add additional scoring callbacks *before* the logger.
    ...     net = NeuralNetClassifier(
    ...         make_classifier(),
    ...         max_epochs=max_epochs,
    ...         lr=0.01,
    ...         callbacks=[EpochScoring("f1"), SacredLogger(_run)]
    ...     )
    ...     # now fit your estimator to your data
    ...     net.fit(X, y)

    Then call this from the command line, e.g. like this:

    .. code-block:: bash

        python sacred-script.py with max_epochs=15

    You can also change other options on the command line and optionally
    specify a backend.

    Parameters
    ----------
    experiment : sacred.Experiment
      Instantiated ``Experiment`` class.

    log_on_batch_end : bool (default=False)
      Whether to log loss and other metrics on batch level.

    log_on_epoch_end : bool (default=True)
      Whether to log loss and other metrics on epoch level.

    batch_suffix : str (default=None)
      A string that will be appended to all logged keys. By default (if set to
      ``None``) "_batch" is used if batch and epoch logging are both enabled
      and no suffix is used otherwise.

    epoch_suffix : str (default=None)
      A string that will be appended to all logged keys. By default (if set to
      ``None``) "_epoch" is used if batch and epoch logging are both enabled
      and no suffix is used otherwise.

    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be logged to Sacred. Note that in
      addition to the keys provided by the user, keys such as those starting
      with ``'event_'`` or ending on ``'_best'`` are ignored by default.

    """

    def __init__(
        self,
        experiment,
        log_on_batch_end=False,
        log_on_epoch_end=True,
        batch_suffix=None,
        epoch_suffix=None,
        keys_ignored=None,
    ):
        self.experiment = experiment
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end
        self.batch_suffix = batch_suffix
        self.epoch_suffix = epoch_suffix
        self.keys_ignored = keys_ignored

    def initialize(self):
        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add("batches")

        self.batch_suffix_ = self.batch_suffix
        self.epoch_suffix_ = self.epoch_suffix
        if self.batch_suffix_ is None:
            self.batch_suffix_ = (
                "_batch" if self.log_on_batch_end and self.log_on_epoch_end else ""
            )
        if self.epoch_suffix_ is None:
            self.epoch_suffix_ = (
                "_epoch" if self.log_on_batch_end and self.log_on_epoch_end else ""
            )
        return self

    def on_batch_end(self, net, **kwargs):
        if not self.log_on_batch_end:
            return
        batch_logs = net.history[-1]["batches"][-1]

        for key in filter_log_keys(batch_logs.keys(), self.keys_ignored_):
            # skorch does not keep a batch count, but sacred will
            # automatically associate the results with a counter.
            self.experiment.log_scalar(key + self.batch_suffix_, batch_logs[key])

    def on_epoch_end(self, net, **kwargs):
        """Automatically log values from the last history step."""
        if not self.log_on_epoch_end:
            return
        epoch_logs = net.history[-1]
        epoch = epoch_logs["epoch"]

        for key in filter_log_keys(epoch_logs.keys(), self.keys_ignored_):
            self.experiment.log_scalar(key + self.epoch_suffix_, epoch_logs[key], epoch)


class MlflowLogger(Callback):
    """Logs results from history and artifact to Mlflow

    "MLflow is an open source platform for managing
    the end-to-end machine learning lifecycle" (:doc:`mlflow:index`)

    Use this callback to automatically log your metrics
    and create/log artifacts to mlflow.

    The best way to log additional information is to log directly to the
    experiment object or subclass the ``on_*`` methods.

    To use this logger, you first have to install Mlflow:

    .. code-block::

      $ python -m pip install mlflow

    Examples
    --------

    Mlflow :doc:`fluent API <mlflow:python_api/mlflow>`:

    >>> import mlflow
    >>> net = NeuralNetClassifier(net, callbacks=[MLflowLogger()])
    >>> with mlflow.start_run():
    ...     net.fit(X, y)

    Custom :py:class:`run <mlflow.entities.Run>` and
    :py:class:`client <mlflow.tracking.MlflowClient>`:

    >>> from mlflow.tracking import MlflowClient
    >>> client = MlflowClient()
    >>> experiment = client.get_experiment_by_name('Default')
    >>> run = client.create_run(experiment.experiment_id)
    >>> net = NeuralNetClassifier(..., callbacks=[MlflowLogger(run, client)])
    >>> net.fit(X, y)

    Parameters
    ----------

    run : mlflow.entities.Run (default=None)
      Instantiated :py:class:`mlflow.entities.Run` class.
      By default (if set to ``None``),
      :py:func:`mlflow.active_run` is used to get the current run.

    client : mlflow.tracking.MlflowClient (default=None)
      Instantiated :py:class:`mlflow.tracking.MlflowClient` class.
      By default (if set to ``None``),
      ``MlflowClient()`` is used, which by default has:

      - the tracking URI set by :py:func:`mlflow.set_tracking_uri`
      - the registry URI set by :py:func:`mlflow.set_registry_uri`

    create_artifact : bool (default=True)
      Whether to create artifacts for the network's
      params, optimizer, criterion and history.
      See :ref:`save_load`

    terminate_after_train : bool (default=True)
      Whether to terminate the ``Run`` object once training finishes.

    log_on_batch_end : bool (default=False)
      Whether to log loss and other metrics on batch level.

    log_on_epoch_end : bool (default=True)
      Whether to log loss and other metrics on epoch level.

    batch_suffix : str (default=None)
      A string that will be appended to all logged keys. By default (if set to
      ``None``) ``'_batch'`` is used if batch and epoch logging are both enabled
      and no suffix is used otherwise.

    epoch_suffix : str (default=None)
      A string that will be appended to all logged keys. By default (if set to
      ``None``) ``'_epoch'`` is used if batch and epoch logging are both enabled
      and no suffix is used otherwise.

    keys_ignored : str or list of str (default=None)
      Key or list of keys that should not be logged to Mlflow. Note that in
      addition to the keys provided by the user, keys such as those starting
      with ``'event_'`` or ending on ``'_best'`` are ignored by default.
    """
    def __init__(
        self,
        run=None,
        client=None,
        create_artifact=True,
        terminate_after_train=True,
        log_on_batch_end=False,
        log_on_epoch_end=True,
        batch_suffix=None,
        epoch_suffix=None,
        keys_ignored=None,
    ):
        self.run = run
        self.client = client
        self.create_artifact = create_artifact
        self.terminate_after_train = terminate_after_train
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end
        self.batch_suffix = batch_suffix
        self.epoch_suffix = epoch_suffix
        self.keys_ignored = keys_ignored

    def initialize(self):
        self.run_ = self.run
        if self.run_ is None:
            import mlflow
            self.run_ = mlflow.active_run()
        self.client_ = self.client
        if self.client_ is None:
            from mlflow.tracking import MlflowClient
            self.client_ = MlflowClient()
        keys_ignored = self.keys_ignored
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored_ = set(keys_ignored or [])
        self.keys_ignored_.add('batches')
        self.batch_suffix_ = self._init_suffix(self.batch_suffix, '_batch')
        self.epoch_suffix_ = self._init_suffix(self.epoch_suffix, '_epoch')
        return self

    def _init_suffix(self, suffix, default):
        if suffix is not None:
            return suffix
        return default if self.log_on_batch_end and self.log_on_epoch_end else ''

    def on_train_begin(self, net, **kwargs):
        self._batch_count = 0

    def on_batch_end(self, net, training, **kwargs):
        if not self.log_on_batch_end:
            return
        self._batch_count += 1
        batch_logs = net.history[-1]['batches'][-1]
        self._iteration_log(batch_logs, self.batch_suffix_, self._batch_count)

    def on_epoch_end(self, net, **kwargs):
        if not self.log_on_epoch_end:
            return
        epoch_logs = net.history[-1]
        self._iteration_log(epoch_logs, self.epoch_suffix_, len(net.history))

    def _iteration_log(self, logs, suffix, step):
        for key in filter_log_keys(logs.keys(), self.keys_ignored_):
            self.client_.log_metric(
                self.run_.info.run_id,
                key + suffix,
                logs[key],
                step=step,
            )

    def on_train_end(self, net, **kwargs):
        try:
            self._log_artifacts(net)
        finally:
            if self.terminate_after_train:
                self.client_.set_terminated(self.run_.info.run_id)

    def _log_artifacts(self, net):
        if not self.create_artifact:
            return
        with tempfile.TemporaryDirectory(prefix='skorch_mlflow_logger_') as dirpath:
            dirpath = Path(dirpath)
            params_filepath = dirpath / 'params.pth'
            optimizer_filepath = dirpath / 'optimizer.pth'
            criterion_filepath = dirpath / 'criterion.pth'
            history_filepath = dirpath / 'history.json'
            net.save_params(
                f_params=params_filepath,
                f_optimizer=optimizer_filepath,
                f_criterion=criterion_filepath,
                f_history=history_filepath,
            )
            self.client_.log_artifact(self.run_.info.run_id, params_filepath)
            self.client_.log_artifact(self.run_.info.run_id, optimizer_filepath)
            self.client_.log_artifact(self.run_.info.run_id, criterion_filepath)
            self.client_.log_artifact(self.run_.info.run_id, history_filepath)
