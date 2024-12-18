"""NeuralNet subclasses for classification tasks."""

import re

import numpy as np
from sklearn.base import ClassifierMixin
import torch
from torch.utils.data import DataLoader

from skorch import NeuralNet
from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import EpochScoring
from skorch.callbacks import PassthroughScoring
from skorch.dataset import ValidSplit
from skorch.utils import data_from_dataset, is_dataset, get_dim, to_numpy

neural_net_clf_doc_start = """NeuralNet for classification tasks

    Use this specifically if you have a standard classification task,
    with input data X and target y.

"""

neural_net_clf_additional_text = """

    criterion : torch criterion (class, default=torch.nn.NLLLoss)
      Negative log likelihood loss. Note that the module should return
      probabilities, the log is applied during ``get_loss``.

    classes : None or list (default=None)
      If None, the ``classes_`` attribute will be inferred from the
      ``y`` data passed to ``fit``. If a non-empty list is passed,
      that list will be returned as ``classes_``. If the initial
      skorch behavior should be restored, i.e. raising an
      ``AttributeError``, pass an empty list."""

neural_net_clf_additional_attribute = """classes_ : array, shape (n_classes, )
      A list of class labels known to the classifier.

"""


def get_neural_net_clf_doc(doc):
    doc = neural_net_clf_doc_start + " " + doc.split("\n ", 4)[-1]
    pattern = re.compile(r'(\n\s+)(criterion .*\n)(\s.+){1,99}')
    start, end = pattern.search(doc).span()
    doc = doc[:start] + neural_net_clf_additional_text + doc[end:]
    doc = doc + neural_net_clf_additional_attribute
    return doc


# pylint: disable=missing-docstring
class NeuralNetClassifier(ClassifierMixin, NeuralNet):
    __doc__ = get_neural_net_clf_doc(NeuralNet.__doc__)

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.NLLLoss,
            train_split=ValidSplit(5, stratified=True),
            classes=None,
            **kwargs
    ):
        super(NeuralNetClassifier, self).__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )
        self.classes = classes

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]

    @property
    def classes_(self):
        if self.classes is not None:
            if not len(self.classes):
                raise AttributeError("{} has no attribute 'classes_'".format(
                    self.__class__.__name__))
            return np.asarray(self.classes)

        try:
            return self.classes_inferred_
        except AttributeError as exc:
            # It's not easily possible to track exactly what circumstances led
            # to this, so try to make an educated guess and provide a possible
            # solution.
            msg = (
                f"{self.__class__.__name__} could not infer the classes from y; "
                "this error probably occurred because the net was trained without y "
                "and some function tried to access the '.classes_' attribute; "
                "a possible solution is to provide the 'classes' argument when "
                f"initializing {self.__class__.__name__}"
            )
            raise AttributeError(msg) from exc

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

        if (y is None) and is_dataset(X):
            try:
                _, y_ds = data_from_dataset(X)
                self.classes_inferred_ = np.unique(to_numpy(y_ds))
            except AttributeError:
                # If this fails, we might still be good to go, so don't raise
                pass

        if y is not None:
            # pylint: disable=attribute-defined-outside-init
            self.classes_inferred_ = np.unique(to_numpy(y))

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # we can assume that the attribute criterion_ exists; if users define
        # custom criteria, they have to override get_loss anyway
        if isinstance(self.criterion_, torch.nn.NLLLoss):
            eps = torch.finfo(y_pred.dtype).eps
            y_pred = torch.log(y_pred + eps)
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
        return self.predict_proba(X).argmax(axis=1)


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


class NeuralNetBinaryClassifier(ClassifierMixin, NeuralNet):
    # pylint: disable=missing-docstring
    __doc__ = get_neural_net_binary_clf_doc(NeuralNet.__doc__)

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.BCEWithLogitsLoss,
            train_split=ValidSplit(5, stratified=True),
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
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]

    @property
    def classes_(self):
        return np.array([0, 1])

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        super().check_data(X, y)
        if (not is_dataset(X)) and (get_dim(y) != 1):
            raise ValueError("The target data should be 1-dimensional.")

    def infer(self, x, **fit_params):
        """Perform an inference step

        The first output of the ``module`` must be a single array that
        has either shape (n,) or shape (n, 1). In the latter case, the
        output will be reshaped to become 1-dim.

        """
        y_infer = super().infer(x, **fit_params)
        rest = None
        if isinstance(y_infer, tuple):
            y_infer, *rest = y_infer

        if (y_infer.dim() > 2) or ((y_infer.dim() == 2) and (y_infer.shape[1] != 1)):
            raise ValueError(
                "Expected module output to have shape (n,) or "
                "(n, 1), got {} instead".format(tuple(y_infer.shape)))

        y_infer = y_infer.reshape(-1)
        if rest is None:
            return y_infer
        return (y_infer,) + tuple(rest)

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
        y_proba = self.predict_proba(X)
        return (y_proba[:, 1] > self.threshold).astype('uint8')
