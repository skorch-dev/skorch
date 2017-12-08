"""Contains callback base class and callbacks."""

from functools import partial
from itertools import cycle
from numbers import Number
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _score
from tabulate import tabulate
import tqdm

from skorch.utils import Ansi
from skorch.utils import to_numpy
from skorch.exceptions import SkorchException


class Callback:
    """Base class for callbacks.

    All custom callbacks should inherit from this class. The subclass
    may override any of the ``on_...`` methods. It is, however, not
    necessary to override all of them, since it's okay if they don't
    have any effect.

    Classes that inherit from this also gain the ``get_params`` and
    ``set_params`` method.

    """
    def initialize(self):
        """(Re-)Set the initial state of the callback. Use this
        e.g. if the callback tracks some state that should be reset
        when the model is re-initialized.

        This method should return self.

        """
        return self

    def on_train_begin(self, net, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, net, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, net, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, net, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, net, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, net, **kwargs):
        """Called at the end of each batch."""
        pass

    def _get_param_names(self):
        return (key for key in self.__dict__ if not key.endswith('_'))

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self


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


class ScoringBase(Callback):
    """Base class for scoring.

    Subclass and implement an ``on_*`` method before using.
    """
    def __init__(
            self,
            scoring,
            lower_is_better=True,
            on_train=False,
            name=None,
            target_extractor=to_numpy,
    ):
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.on_train = on_train
        self.name = name
        self.target_extractor = target_extractor

    def _get_name(self):
        if self.name is not None:
            return self.name
        if self.scoring is None:
            return 'score'
        if isinstance(self.scoring, str):
            return self.scoring
        if isinstance(self.scoring, partial):
            return self.scoring.func.__name__
        return self.scoring.__name__

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        self.name_ = self._get_name()
        return self

    def _scoring(self, net, X_test, y_test):
        """Resolve scoring and apply it to data."""
        scorer = check_scoring(net, self.scoring)
        scores = _score(
            estimator=net,
            X_test=X_test,
            y_test=y_test,
            scorer=scorer,
            is_multimetric=False,
        )
        return scores

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_


class BatchScoring(ScoringBase):
    """Callback that performs generic scoring on batches.

    This callback determines the score after each batch and stores it
    in the net's history in the column given by ``name``. At the end
    of the epoch, the average of the scores are determined and also
    stored in the history. Furthermore, it is determined whether this
    average score is the best score yet and that information is also
    stored in the history.

    In contrast to ``EpochScoring``, this callback determines the
    score for each batch and then averages the score at the end of the
    epoch. This can be disadvantageous for some scores if the batch
    size is small -- e.g. area under the ROC will return incorrect
    scores in this case. Therefore, it is recommnded to use
    ``EpochScoring`` unless you really need the scores for each batch.

    Parameters
    ----------
    scoring : None, str, or callable
      If None, use the ``score`` method of the model. If str, it should
      be a valid sklearn metric (e.g. "f1_score", "accuracy_score"). If
      a callable, it should have the signature (model, X, y), and it
      should return a scalar. This works analogously to the ``scoring``
      parameter in sklearn's ``GridSearchCV`` et al.

    lower_is_better : bool (default=True)
      Whether lower (e.g. log loss) or higher (e.g. accuracy) scores
      are better

    on_train : bool (default=False)
      Whether this should be called during train or validation.

    name : str or None (default=None)
      If not an explicit string, tries to infer the name from the
      ``scoring`` argument.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    """
    # pylint: disable=unused-argument,arguments-differ
    def on_batch_end(self, net, X, y, training, **kwargs):
        if training != self.on_train:
            return

        y = self.target_extractor(y)
        try:
            score = self._scoring(net, X, y)
            net.history.record_batch(self.name_, score)
        except KeyError:
            pass

    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name_]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    # pylint: disable=unused-argument
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:
            history[-1, 'batches', :, self.name_]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name_, score_avg)
        if is_best is not None:
            history.record(self.name_ + '_best', is_best)


class EpochScoring(ScoringBase):
    """Callback that performs generic scoring on predictions.

    At the end of each epoch, this callback makes a prediction on
    train or validation data, determines the score for that prediction
    and whether it is the best yet, and stores the result in the net's
    history.

    In case you already computed a score value for each batch you
    can omit the score computation step by return the value from
    the history. For example:

        >>> def my_score(net, X=None, y=None):
        ...     losses = net.history[-1, 'batches', :, 'my_score']
        ...     batch_sizes = net.history[-1, 'batches', :, 'valid_batch_size']
        ...     return np.average(losses, weights=batch_sizes)
        >>> net = MyNet(callbacks=[
        ...     ('my_score', Scoring(my_score, name='my_score'))

    Parameters
    ----------
    scoring : None, str, or callable (default=None)
      If None, use the ``score`` method of the model. If str, it
      should be a valid sklearn scorer (e.g. "f1", "accuracy"). If a
      callable, it should have the signature (model, X, y), and it
      should return a scalar. This works analogously to the
      ``scoring`` parameter in sklearn's ``GridSearchCV`` et al.

    lower_is_better : bool (default=True)
      Whether lower scores should be considered better or worse.

    on_train : bool (default=False)
      Whether this should be called during train or validation data.

    name : str or None (default=None)
      If not an explicit string, tries to infer the name from the
      ``scoring`` argument.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    """
    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(
            self,
            net,
            X,
            y,
            X_valid,
            y_valid,
            **kwargs):
        history = net.history

        if self.on_train:
            X_test, y_test = X, y
        else:
            X_test, y_test = X_valid, y_valid

        if X_test is None:
            return

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)
        current_score = self._scoring(net, X_test, y_test)
        history.record(self.name_, current_score)

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + '_best', is_best)
        if is_best:
            self.best_score_ = current_score


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


class Checkpoint(Callback):
    """Save the model during training if the given metric improved.

    This callback works by default in conjunction with the validation
    scoring callback since it creates a ``valid_loss_best`` value
    in the history which the callback uses to determine if this
    epoch is save-worthy.

    You can also specify your own metric to monitor or supply a
    callback that dynamically evaluates whether the model should
    be saved in this epoch.

    Example:

    >>> net = MyNet(callbacks=[Checkpoint()])
    >>> net.fit(X, y)

    Example using a custom monitor where only models are saved in
    epochs where the validation *and* the train loss is best:

    >>> monitor = lambda net: all(net.history[-1, (
    ...     'train_loss_best', 'valid_loss_best')])
    >>> net = MyNet(callbacks=[Checkpoint(monitor=monitor)])
    >>> net.fit(X, y)

    Parameters
    ----------
    target : file-like object, str
      File path to the file or file-like object.
      See NeuralNet.save_params for details what this value may be.

      If the value is a string you can also use format specifiers
      to, for example, indicate the current epoch. Accessible format
      values are ``net``, ``last_epoch`` and ``last_batch``.
      Example to include last epoch number in file name:

      >>> cb = Checkpoint(target="target_{last_epoch[epoch]}.pt")

    monitor : str, function, None
      Value of the history to monitor or callback that determines whether
      this epoch should to a checkpoint. The callback takes the network
      instance as parameter.

      In case ``monitor`` is set to ``None``, the callback will save
      the network at every epoch.

      **Note:** If you supply a lambda expression as monitor, you cannot
      pickle the wrapper anymore as lambdas cannot be pickled. You can
      mitigate this problem by using importable functions instead.
    """
    def __init__(
            self,
            target='model.pt',
            monitor='valid_loss_best',
    ):
        self.monitor = monitor
        self.target = target

    def on_epoch_end(self, net, **kwargs):
        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                raise SkorchException(
                    "Monitor value '{}' cannot be found in history. "
                    "Make sure you have validation data if you use "
                    "validation scores for checkpointing.".format(e.args[0]))

        if do_checkpoint:
            target = self.target
            if isinstance(self.target, str):
                target = self.target.format(
                    net=net,
                    last_epoch=net.history[-1],
                    last_batch=net.history[-1, 'batches', -1],
                )
            if net.verbose > 0:
                print("Checkpoint! Saving model to {}.".format(target))
            net.save_params(target)


class ProgressBar(Callback):
    """Display a progress bar for each epoch including duration, estimated
    remaining time and user-defined metrics.

    For jupyter notebooks a non-ASCII progress bar is printed instead.
    To use this feature, you need to have `ipywidgets
    <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`
    installed.

    Parameters:
    -----------

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
        return int(np.ceil(len(X) / batch_size))

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

    # pylint: disable=attribute-defined-outside-init
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
