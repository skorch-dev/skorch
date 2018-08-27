""" Callbacks for printing, logging and log information."""

import sys
import time
from numbers import Number
from itertools import cycle

import numpy as np
import tqdm
from tabulate import tabulate

from skorch.utils import Ansi
from skorch.dataset import get_len
from skorch.callbacks import Callback


__all__ = ['EpochTimer', 'PrintLog', 'ProgressBar']


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
    ``Scoring`` callback takes care of creating those entries, which is
    why ``PrintLog`` works best in conjunction with that callback.

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
      table. Note that keys ending on '_best' are also ignored.

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
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt
        self.stralign = stralign

    def initialize(self):
        self.first_iteration_ = True
        self.keys_ignored_ = set(self.keys_ignored or [])
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
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored_):
            sorted_keys.append('epoch')

        for key in sorted(keys):
            if not (
                    (key in ('epoch', 'dur')) or
                    (key in self.keys_ignored_) or
                    key.endswith('_best') or
                    key.startswith('event_')
            ):
                sorted_keys.append(key)

        for key in sorted(keys):
            if key.startswith('event_') and (key not in self.keys_ignored_):
                sorted_keys.append(key)
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
