"""NeuralNet subclasses for classification tasks."""

import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from skorch import NeuralNet
from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import EpochScoring
from skorch.callbacks import BatchScoring
from skorch.dataset import CVSplit
from skorch.utils import get_dim
from skorch.utils import is_dataset
from skorch.utils import noop
from skorch.utils import to_numpy
from skorch.utils import train_loss_score
from skorch.utils import valid_loss_score


neural_net_clf_doc_start = """NeuralNet for classification tasks

    Use this specifically if you have a standard classification task,
    with input data X and target y.

"""

neural_net_clf_criterion_text = """

    criterion : torch criterion (class, default=torch.nn.NLLLoss)
      Negative log likelihood loss. Note that the module should return
      probabilities, the log is applied during ``get_loss``."""


def get_neural_net_clf_doc(doc):
    doc = neural_net_clf_doc_start + " " + doc.split("\n ", 4)[-1]
    pattern = re.compile(r'(\n\s+)(criterion .*\n)(\s.+){1,99}')
    start, end = pattern.search(doc).span()
    doc = doc[:start] + neural_net_clf_criterion_text + doc[end:]
    return doc


# pylint: disable=missing-docstring
class NeuralNetClassifier(NeuralNet):
    __doc__ = get_neural_net_clf_doc(NeuralNet.__doc__)

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.NLLLoss,
            train_split=CVSplit(5, stratified=True),
            **kwargs
    ):
        super(NeuralNetClassifier, self).__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', BatchScoring(
                train_loss_score,
                name='train_loss',
                on_train=True,
                target_extractor=noop,
            )),
            ('valid_loss', BatchScoring(
                valid_loss_score,
                name='valid_loss',
                target_extractor=noop,
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            msg = ("No y-values are given (y=None). You must either supply a "
                   "Dataset as X or implement your own DataLoader for "
                   "training (and your validation) and supply it using the "
                   "``iterator_train`` and ``iterator_valid`` parameters "
                   "respectively.")
            raise ValueError(msg)

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(self.criterion_, torch.nn.NLLLoss):
            y_pred = torch.log(y_pred)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    # pylint: disable=signature-differs
    def fit(self, X, y, **fit_params):
        """See ``NeuralNet.fit``.

        In contrast to ``NeuralNet.fit``, ``y`` is non-optional to
        avoid mistakenly forgetting about ``y``. However, ``y`` can be
        set to ``None`` in case it is derived dynamically from
        ``X``.

        """
        # pylint: disable=useless-super-delegation
        # this is actually a pylint bug:
        # https://github.com/PyCQA/pylint/issues/1085
        return super(NeuralNetClassifier, self).fit(X, y, **fit_params)

    def predict_proba(self, X):
        """Where applicable, return probability estimates for
        samples.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_proba : numpy ndarray

        """
        # Only the docstring changed from parent.
        # pylint: disable=useless-super-delegation
        return super().predict_proba(X)

    def predict(self, X):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_pred : numpy ndarray

        """
        y_preds = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_preds.append(to_numpy(yp.max(-1)[-1]))
        y_pred = np.concatenate(y_preds, 0)
        return y_pred


neural_net_binary_clf_doc_start = """NeuralNet for binary classification tasks

    Use this specifically if you have a binary classification task,
    with input data X and target y. y must be 1d.

"""

neural_net_binary_clf_criterion_text = """

    criterion : torch criterion (class, default=torch.nn.BCEWithLogitsLoss)
      Binary cross entropy loss with logits. Note that the module should return
      the logit of probabilities with shape (batch_size, ).

    threshold : float (default=0.5)
      Probabilities above this threshold is classified as 1. ``threshold``
      is used by ``predict`` and ``predict_proba`` for classification."""


def get_neural_net_binary_clf_doc(doc):
    doc = neural_net_binary_clf_doc_start + " " + doc.split("\n ", 4)[-1]
    pattern = re.compile(r'(\n\s+)(criterion .*\n)(\s.+){1,99}')
    start, end = pattern.search(doc).span()
    doc = doc[:start] + neural_net_binary_clf_criterion_text + doc[end:]
    return doc


class NeuralNetBinaryClassifier(NeuralNet):
    # pylint: disable=missing-docstring
    __doc__ = get_neural_net_binary_clf_doc(NeuralNet.__doc__)

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.BCEWithLogitsLoss,
            train_split=CVSplit(5, stratified=True),
            threshold=0.5,
            **kwargs
    ):
        super().__init__(
            module,
            criterion=criterion,
            train_split=train_split,
            *args,
            **kwargs
        )
        self.threshold = threshold

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', BatchScoring(
                train_loss_score,
                name='train_loss',
                on_train=True,
                target_extractor=noop,
            )),
            ('valid_loss', BatchScoring(
                valid_loss_score,
                name='valid_loss',
                target_extractor=noop,
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        super().check_data(X, y)
        if get_dim(y) != 1:
            raise ValueError("The target data should be 1-dimensional.")

    # pylint: disable=signature-differs
    def fit(self, X, y, **fit_params):
        """See ``NeuralNet.fit``.

        In contrast to ``NeuralNet.fit``, ``y`` is non-optional to
        avoid mistakenly forgetting about ``y``. However, ``y`` can be
        set to ``None`` in case it is derived dynamically from
        ``X``.

        """
        # pylint: disable=useless-super-delegation
        # this is actually a pylint bug:
        # https://github.com/PyCQA/pylint/issues/1085
        return super().fit(X, y, **fit_params)

    def predict(self, X):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_pred : numpy ndarray

        """
        return (self.predict_proba(X) > self.threshold).astype('uint8')

    # pylint: disable=missing-docstring
    def predict_proba(self, X):
        """Where applicable, return probability estimates for
        samples.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_proba : numpy ndarray

        """
        y_probas = []
        bce_logits_loss = isinstance(
            self.criterion_, torch.nn.BCEWithLogitsLoss)

        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            if bce_logits_loss:
                yp = torch.sigmoid(yp)
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba
