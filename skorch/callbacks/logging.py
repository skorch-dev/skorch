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
    """Print out useful information from the model's history.

    By default, ``PrintLog`` prints everything from the history except
    for ``'batches'``.

    To determine the best loss, ``PrintLog`` looks for keys that end on
    ``'_best'`` and associates them with the corresponding loss. E.g.,
    ``'train_loss_best'`` will be matched with ``'train_loss'``. The
    ``Scoring`` callback takes care of creating those entries, which is
    why ``PrintLog`` works best in conjunction with that callback.

    *Note*: ``PrintLog`` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.

    Parameters
    ----------
    keys_ignored : str or list of str (default='batches')
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

    """
    def __init__(
            self,
            keys_ignored='batches',
            sink=print,
            tablefmt='simple',
            floatfmt='.4f',
    ):
        if isinstance(keys_ignored, str):
            keys_ignored = [keys_ignored]
        self.keys_ignored = keys_ignored
        self.sink = sink
        self.tablefmt = tablefmt
        self.floatfmt = floatfmt

    def initialize(self):
        self.first_iteration_ = True
        return self

    def format_row(self, row, key, color):
        """For a given row from the table, format it (i.e. floating
        points and color if applicable.

        """
        value = row[key]
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
        """Sort keys alphabetically, but put 'epoch' first and 'dur'
        last.

        Ignore keys that are in ``self.ignored_keys`` or that end on
        '_best'.

        """
        sorted_keys = []
        if ('epoch' in keys) and ('epoch' not in self.keys_ignored):
            sorted_keys.append('epoch')

        for key in sorted(keys):
            if not (
                    (key in ('epoch', 'dur')) or
                    (key in self.keys_ignored) or
                    key.endswith('_best')
            ):
                sorted_keys.append(key)

        if ('dur' in keys) and ('dur' not in self.keys_ignored):
            sorted_keys.append('dur')
        return sorted_keys

    def _yield_keys_formatted(self, row):
        colors = cycle([color.value for color in Ansi if color != color.ENDC])
        color = next(colors)
        for key in self._sorted_keys(row.keys()):
            formatted = self.format_row(row, key, color=color)
            yield key, formatted
            color = next(colors)

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
    """Display a progress bar for each epoch including duration, estimated
    remaining time and user-defined metrics.

    For jupyter notebooks a non-ASCII progress bar is printed instead.
    To use this feature, you need to have `ipywidgets
    <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`
    installed.

    Parameters
    ----------

    batches_per_epoch : int, str (default='count')
      The progress bar determines the number of batches per epoch
      by itself in ``'count'`` mode where the number of iterations is
      determined after one epoch which will leave you without a progress
      bar at the first epoch. To fix that you can provide this number manually
      or set ``'auto'`` where the callback attempts to compute the
      number of batches per epoch beforehand.

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
            batches_per_epoch='count',
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

    def _get_batches_per_epoch_phase(self, net, X, training):
        if X is None:
            return 0
        batch_size = self._get_batch_size(net, training)
        return int(np.ceil(get_len(X) / batch_size))

    def _get_batches_per_epoch(self, net, X, X_valid):
        return (self._get_batches_per_epoch_phase(net, X, True) +
                self._get_batches_per_epoch_phase(net, X_valid, False))

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
        self.pbar.set_postfix(self._get_postfix_dict(net))
        self.pbar.update()

    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def on_epoch_begin(self, net, X=None, X_valid=None, **kwargs):
        # Assume it is a number until proven otherwise.
        batches_per_epoch = self.batches_per_epoch

        if self.batches_per_epoch == 'auto':
            batches_per_epoch = self._get_batches_per_epoch(net, X, X_valid)
        elif self.batches_per_epoch == 'count':
            # No limit is known until the end of the first epoch.
            batches_per_epoch = None

        if self._use_notebook():
            self.pbar = tqdm.tqdm_notebook(total=batches_per_epoch)
        else:
            self.pbar = tqdm.tqdm(total=batches_per_epoch)

    def on_epoch_end(self, net, **kwargs):
        if self.batches_per_epoch == 'count':
            self.batches_per_epoch = self.pbar.n
        self.pbar.close()
