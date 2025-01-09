""" Callbacks for calculating scores."""

from contextlib import contextmanager
from contextlib import suppress
from functools import partial
import warnings

import numpy as np
import sklearn
from sklearn.metrics import make_scorer, check_scoring

from skorch.callbacks import Callback
from skorch.dataset import unpack_data
from sklearn.metrics._scorer import _BaseScorer
from skorch.utils import data_from_dataset
from skorch.utils import is_skorch_dataset
from skorch.utils import to_numpy
from skorch.utils import check_indexing
from skorch.utils import to_device


__all__ = ['BatchScoring', 'EpochScoring', 'PassthroughScoring']


@contextmanager
def _cache_net_forward_iter(net, use_caching, y_preds):
    """Caching context for ``skorch.NeuralNet`` instance.

    Returns a modified version of the net whose ``forward_iter``
    method will subsequently return cached predictions. Leaving the
    context will undo the overwrite of the ``forward_iter`` method.

    Note that the net may override the use of caching.

    """
    if net.use_caching != 'auto':
        use_caching = net.use_caching

    if not use_caching:
        yield net
        return
    y_preds = iter(y_preds)

    # pylint: disable=unused-argument
    def cached_forward_iter(*args, device=net.device, **kwargs):
        for yp in y_preds:
            yield to_device(yp, device=device)

    net.forward_iter = cached_forward_iter
    try:
        yield net
    finally:
        # By setting net.forward_iter we define an attribute
        # `forward_iter` that precedes the bound method
        # `forward_iter`. By deleting the entry from the attribute
        # dict we undo this.
        del net.__dict__['forward_iter']


def convert_sklearn_metric_function(scoring):
    """If ``scoring`` is a sklearn metric function, convert it to a
    sklearn scorer and return it. Otherwise, return ``scoring`` unchanged."""
    if callable(scoring):
        module = getattr(scoring, '__module__', None)

        # those are scoring objects returned by make_scorer starting
        # from sklearn 0.22
        scorer_names = ('_PredictScorer', '_ProbaScorer', '_ThresholdScorer', '_Scorer')
        if (
                hasattr(module, 'startswith') and
                module.startswith('sklearn.metrics.') and
                not module.startswith('sklearn.metrics.scorer') and
                not module.startswith('sklearn.metrics.tests.') and
                not scoring.__class__.__name__ in scorer_names
        ):
            return make_scorer(scoring)
    return scoring


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
            use_caching=True,
    ):
        self.scoring = scoring
        self.lower_is_better = lower_is_better
        self.on_train = on_train
        self.name = name
        self.target_extractor = target_extractor
        self.use_caching = use_caching

    # pylint: disable=protected-access
    def _get_name(self):
        """Find name of scoring function."""
        if self.name is not None:
            return self.name
        if self.scoring_ is None:
            return 'score'
        if isinstance(self.scoring_, str):
            return self.scoring_
        if isinstance(self.scoring_, partial):
            return self.scoring_.func.__name__
        if isinstance(self.scoring_, _BaseScorer):
            if hasattr(self.scoring_._score_func, '__name__'):
                # sklearn < 0.22
                return self.scoring_._score_func.__name__
            # sklearn >= 0.22
            return self.scoring_._score_func._score_func.__name__
        if isinstance(self.scoring_, dict):
            raise ValueError("Dict not supported as scorer for multi-metric scoring."
                             " Register multiple scoring callbacks instead.")
        return self.scoring_.__name__

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        self.scoring_ = convert_sklearn_metric_function(self.scoring)
        self.name_ = self._get_name()
        return self

    # pylint: disable=attribute-defined-outside-init,arguments-differ
    def on_train_begin(self, net, X, y, **kwargs):
        self.X_indexing_ = check_indexing(X)
        self.y_indexing_ = check_indexing(y)

        # Looks for the right most index where `*_best` is True
        # That index is used to get the best score in `net.history`
        with suppress(ValueError, IndexError, KeyError):
            best_name_history = net.history[:, '{}_best'.format(self.name_)]
            idx_best_reverse = best_name_history[::-1].index(True)
            idx_best = len(best_name_history) - idx_best_reverse - 1
            self.best_score_ = net.history[idx_best, self.name_]

    def _scoring(self, net, X_test, y_test):
        """Resolve scoring and apply it to data. Use cached prediction
        instead of running inference again, if available."""
        scorer = check_scoring(net, self.scoring_)
        return scorer(net, X_test, y_test)

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

    In contrast to :class:`.EpochScoring`, this callback determines
    the score for each batch and then averages the score at the end of
    the epoch. This can be disadvantageous for some scores if the
    batch size is small -- e.g. area under the ROC will return
    incorrect scores in this case. Therefore, it is recommnded to use
    :class:`.EpochScoring` unless you really need the scores for each
    batch.

    If ``y`` is None, the ``scoring`` function with signature (model, X, y)
    must be able to handle ``X`` as a ``Tensor`` and ``y=None``.

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
      are better.

    on_train : bool (default=False)
      Whether this should be called during train or validation.

    name : str or None (default=None)
      If not an explicit string, tries to infer the name from the
      ``scoring`` argument.

    target_extractor : callable (default=to_numpy)
      This is called on y before it is passed to scoring.

    use_caching : bool (default=True)
      Re-use the model's prediction for computing the loss to calculate
      the score. Turning this off will result in an additional inference
      step for each batch. Note that the net may override the use of
      caching.

    """
    # pylint: disable=unused-argument,arguments-differ

    def on_batch_end(self, net, batch, training, **kwargs):
        if training != self.on_train:
            return

        X, y = unpack_data(batch)
        y_preds = [kwargs['y_pred']]
        with _cache_net_forward_iter(net, self.use_caching, y_preds) as cached_net:
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y=None.
            y = None if y is None else self.target_extractor(y)
            try:
                score = self._scoring(cached_net, X, y)
                cached_net.history.record_batch(self.name_, score)
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
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name_]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name_, score_avg)
        if is_best is not None:
            history.record(self.name_ + '_best', bool(is_best))


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

    If you fit with a custom dataset, this callback should work as
    expected as long as ``use_caching=True`` which enables the
    collection of ``y`` values from the dataset. If you decide to
    disable the caching of predictions and ``y`` values, you need
    to write your own scoring function that is able to deal with the
    dataset and returns a scalar, for example:

        >>> def ds_accuracy(net, ds, y=None):
        ...     # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
        ...     y_true = [y for _, y in ds]
        ...     y_pred = net.predict(ds)
        ...     return sklearn.metrics.accuracy_score(y_true, y_pred)
        >>> net = MyNet(callbacks=[
        ...     EpochScoring(ds_accuracy, use_caching=False)])
        >>> ds = torchvision.datasets.MNIST(root=mnist_path)
        >>> net.fit(ds)

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

    use_caching : bool (default=True)
      Collect labels and predictions (``y_true`` and ``y_pred``)
      over the course of one epoch and use the cached values for
      computing the score. The cached values are shared between
      all ``EpochScoring`` instances. Disabling this will result
      in an additional inference step for each epoch and an
      inability to use arbitrary datasets as input (since we
      don't know how to extract ``y_true`` from an arbitrary
      dataset). Note that the net may override the use of caching.

    """
    def _initialize_cache(self):
        self.y_trues_ = []
        self.y_preds_ = []

    def initialize(self):
        super().initialize()
        self._initialize_cache()
        return self

    # pylint: disable=arguments-differ,unused-argument
    def on_epoch_begin(self, net, dataset_train, dataset_valid, **kwargs):
        self._initialize_cache()

    # pylint: disable=arguments-differ
    def on_batch_end(
            self, net, batch, y_pred, training, **kwargs):
        use_caching = self.use_caching
        if net.use_caching !=  'auto':
            use_caching = net.use_caching

        if (not use_caching) or (training != self.on_train):
            return

        # We collect references to the prediction and target data
        # emitted by the training process. Since we don't copy the
        # data, all *Scoring callback instances use the same
        # underlying data. This is also the reason why we don't run
        # self.target_extractor(y) here but on epoch end, so that
        # there are no copies of parts of y hanging around during
        # training.
        _X, y = unpack_data(batch)
        if y is not None:
            self.y_trues_.append(y)
        self.y_preds_.append(y_pred)

    def get_test_data(self, dataset_train, dataset_valid, use_caching):
        """Return data needed to perform scoring.

        This is a convenience method that handles picking of
        train/valid, different types of input data, use of cache,
        etc. for you.

        Parameters
        ----------
        dataset_train
          Incoming training data or dataset.

        dataset_valid
          Incoming validation data or dataset.

        use_caching : bool
          Whether caching of inference is being used.

        Returns
        -------
        X_test
          Input data used for making the prediction.

        y_test
          Target ground truth. If caching was enabled, return cached
          y_test.

        y_pred : list
          The predicted targets. If caching was disabled, the list is
          empty. If caching was enabled, the list contains the batches
          of the predictions. It may thus be necessary to concatenate
          the output before working with it:
          ``y_pred = np.concatenate(y_pred)``

        """
        dataset = dataset_train if self.on_train else dataset_valid

        if use_caching:
            X_test = dataset
            y_pred = self.y_preds_
            y_test = [self.target_extractor(y) for y in self.y_trues_]
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y_test=None.
            y_test = np.concatenate(y_test) if y_test else None
            return X_test, y_test, y_pred

        if is_skorch_dataset(dataset):
            X_test, y_test = data_from_dataset(
                dataset,
                X_indexing=self.X_indexing_,
                y_indexing=self.y_indexing_,
            )
        else:
            X_test, y_test = dataset, None

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)
        return X_test, y_test, []

    def _record_score(self, history, current_score):
        """Record the current store and, if applicable, if it's the best score
        yet.

        """
        history.record(self.name_, current_score)

        is_best = self._is_best_score(current_score)
        if is_best is None:
            return

        history.record(self.name_ + '_best', bool(is_best))
        if is_best:
            self.best_score_ = current_score

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(
            self,
            net,
            dataset_train,
            dataset_valid,
            **kwargs):
        use_caching = self.use_caching
        if net.use_caching !=  'auto':
            use_caching = net.use_caching

        X_test, y_test, y_pred = self.get_test_data(
            dataset_train,
            dataset_valid,
            use_caching=use_caching,
        )
        if X_test is None:
            return

        with _cache_net_forward_iter(net, self.use_caching, y_pred) as cached_net:
            current_score = self._scoring(cached_net, X_test, y_test)

        self._record_score(net.history, current_score)

    def on_train_end(self, *args, **kwargs):
        self._initialize_cache()


class PassthroughScoring(Callback):
    """Creates scores on epoch level based on batch level scores

    This callback doesn't calculate any new scores but instead passes
    through a score that was created on the batch level. Based on that
    score, an average across the batch is created (honoring the batch
    size) and recorded in the history for the given epoch.

    Use this callback when there already is a score calculated on the
    batch level. If that score has yet to be calculated, use
    :class:`.BatchScoring` instead.

    Parameters
    ----------
    name : str
      Name of the score recorded on a batch level in the history.

    lower_is_better : bool (default=True)
      Whether lower (e.g. log loss) or higher (e.g. accuracy) scores
      are better.

    on_train : bool (default=False)
      Whether this should be called during train or validation.

    """
    def __init__(
            self,
            name,
            lower_is_better=True,
            on_train=False,
    ):
        self.name = name
        self.lower_is_better = lower_is_better
        self.on_train = on_train

    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        return self

    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_

    def get_avg_score(self, history):
        if self.on_train:
            bs_key = 'train_batch_size'
        else:
            bs_key = 'valid_batch_size'

        weights, scores = list(zip(
            *history[-1, 'batches', :, [bs_key, self.name]]))
        score_avg = np.average(scores, weights=weights)
        return score_avg

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, net, **kwargs):
        history = net.history
        try:  # don't raise if there is no valid data
            history[-1, 'batches', :, self.name]
        except KeyError:
            return

        score_avg = self.get_avg_score(history)
        is_best = self._is_best_score(score_avg)
        if is_best:
            self.best_score_ = score_avg

        history.record(self.name, score_avg)
        if is_best is not None:
            history.record(self.name + '_best', bool(is_best))
