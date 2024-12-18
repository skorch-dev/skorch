"""NeuralNet subclasses for regression tasks."""

import re

from sklearn.base import RegressorMixin
import torch
from torch.utils.data import DataLoader

from skorch import NeuralNet
from skorch.utils import get_dim
from skorch.utils import is_dataset


neural_net_reg_doc_start = """NeuralNet for regression tasks

    Use this specifically if you have a standard regression task,
    with input data X and target y. y must be 2d.

"""

neural_net_reg_criterion_text = """

    criterion : torch criterion (class, default=torch.nn.MSELoss)
      Mean squared error loss."""


def get_neural_net_reg_doc(doc):
    doc = neural_net_reg_doc_start + " " + doc.split("\n ", 4)[-1]
    pattern = re.compile(r'(\n\s+)(criterion .*\n)(\s.+){1,99}')
    start, end = pattern.search(doc).span()
    doc = doc[:start] + neural_net_reg_criterion_text + doc[end:]
    return doc


# pylint: disable=missing-docstring
class NeuralNetRegressor(RegressorMixin, NeuralNet):
    __doc__ = get_neural_net_reg_doc(NeuralNet.__doc__)

    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.MSELoss,
            **kwargs
    ):
        super(NeuralNetRegressor, self).__init__(
            module,
            *args,
            criterion=criterion,
            **kwargs
        )

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            raise ValueError("No y-values are given (y=None). You must "
                             "implement your own DataLoader for training "
                             "(and your validation) and supply it using the "
                             "``iterator_train`` and ``iterator_valid`` "
                             "parameters respectively.")
        if y is None:
            # The user implements its own mechanism for generating y.
            return

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
        return super(NeuralNetRegressor, self).fit(X, y, **fit_params)
