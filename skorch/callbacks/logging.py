""" Callbacks for printing, logging and log information."""

import sys
import time
from contextlib import suppress
from numbers import Number
from itertools import cycle

import numpy as np
import tqdm
from tabulate import tabulate

from skorch.utils import Ansi
from skorch.dataset import get_len
from skorch.callbacks import Callback


__all__ = ['EpochTimer', 'PrintLog', 'ProgressBar', 'TensorBoard']


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
      Key or list of keys that should not be part of the printed
      table. Note that in addition to the keys provided by the user,
      keys such as those starting with 'event_' or ending on '_best'
      are ignored by default.

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
        self.pbar.set_postfix(self._get_postfix_dict(net), refresh=False)
        self.pbar.update()

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
            self.pbar = tqdm.tqdm_notebook(total=batches_per_epoch, leave=False)
        else:
            self.pbar = tqdm.tqdm(total=batches_per_epoch, leave=False)

    def on_epoch_end(self, net, **kwargs):
        self.pbar.close()


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

    "TensorBoard provides the visualization and tooling needed for
    machine learning experimentation" (tensorboard_)

    Use this callback to automatically log all interesting values from
    your net's history to tensorboard after each epoch.

    The best way to log additional information is to subclass this
    callback and add your code to one of the ``on_*`` methods.

    Examples
    --------
    >>> # Example to log the bias parameter as a histogram
    >>> def extract_bias(module):
    ...     return module.hidden.bias

    >>> class MyTensorBoard(TensorBoard):
    ...     def on_epoch_end(self, net, **kwargs):
    ...         bias = extract_bias(net.module_)
    ...         epoch = net.history[-1, 'epoch']
    ...         self.writer.add_histogram('bias', bias, global_step=epoch)
    ...         super().on_epoch_end(net, **kwargs)  # call super last

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
      Key or list of keys that should not be logged to
      tensorboard. Note that in addition to the keys provided by the
      user, keys such as those starting with 'event_' or ending on
      '_best' are ignored by default.

    key_mapper : callable or function (default=rename_tensorboard_key)
      This function maps a key name from the history to a tag in
      tensorboard. This is useful because tensorboard can
      automatically group similar tags if their names start with the
      same prefix, followed by a forward slash. By default, this
      callback will prefix all keys that start with "train" or "valid"
      with the "Loss/" prefix.

    .. _tensorboard: https://www.tensorflow.org/tensorboard/

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
