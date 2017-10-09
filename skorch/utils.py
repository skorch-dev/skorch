"""skorch utilities.

Should not have any dependency on other skorch packages.

"""

from collections.abc import Sequence
from enum import Enum
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class Ansi(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


def to_var(X, use_cuda=False):
    """Generic function to convert a input data to pytorch Variables.

    Returns X when it already is a pytorch Variable.

    """
    if isinstance(X, (Variable, nn.utils.rnn.PackedSequence)):
        return X

    X = to_tensor(X, use_cuda=use_cuda)
    if isinstance(X, dict):
        return {k: to_var(v) for k, v in X.items()}

    if isinstance(X, (tuple, list)):
        return [to_var(x) for x in X]

    return Variable(X)


def to_tensor(X, use_cuda=False):
    """Turn to torch Variable.

    Handles the cases:
      * Variable
      * PackedSequence
      * numpy array
      * torch Tensor
      * list or tuple of one of the former
      * dict of one of the former

    """
    to_tensor_ = partial(to_tensor, use_cuda=use_cuda)

    if isinstance(X, (Variable, nn.utils.rnn.PackedSequence)):
        return X

    if isinstance(X, dict):
        return {key: to_tensor_(val) for key, val in X.items()}

    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if isinstance(X, Sequence):
        X = torch.from_numpy(np.array(X))
    elif np.isscalar(X):
        X = torch.from_numpy(np.array([X]))

    if use_cuda:
        X = X.cuda()
    return X


def to_numpy(X):
    """Generic function to convert a pytorch tensor or variable to
    numpy.

    Returns X when it already is a numpy array.

    """
    if isinstance(X, np.ndarray):
        return X

    if is_pandas_ndframe(X):
        return X.values

    if X.is_cuda:
        X = X.cpu()

    if isinstance(X, Variable):
        data = X.data
    else:
        data = X
    return data.numpy()


def check_history_slice(history, sl):
    # Note: May extend this for more cases.
    try:
        # pylint: disable=pointless-statement
        history[sl]
        return
    except KeyError as exc:
        msg = ("Key '{}' could not be found in history; "
               "maybe there was a typo? To make this key optional, "
               "add it to the 'keys_optional' parameter.".format(exc.args[0]))
        raise KeyError(msg)


def get_dim(y):
    """Return the number of dimensions of a torch tensor or numpy
    array-like object.

    """
    try:
        return y.ndim
    except AttributeError:
        return y.dim()


def is_pandas_ndframe(x):
    # the sklearn way of determining this
    return hasattr(x, 'iloc')


def flatten(arr):
    for item in arr:
        if isinstance(item, (tuple, list, dict)):
            yield from flatten(item)
        else:
            yield item


def duplicate_items(*collections):
    """Search for duplicate items in all collections.

    Examples
    --------
    >>> duplicate_items([1, 2], [3])
    set()
    >>> duplicate_items({1: 'a', 2: 'a'})
    set()
    >>> duplicate_items(['a', 'b', 'a'])
    {'a'}
    >>> duplicate_items([1, 2], {3: 'hi', 4: 'ha'}, (2, 3))
    {2, 3}

    """
    duplicates = set()
    seen = set()
    for item in flatten(collections):
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates
