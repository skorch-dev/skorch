from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import collections.abc


class Ansi(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


def to_var(X, use_cuda=False):
    X = to_tensor(X, use_cuda=use_cuda)
    return Variable(X)


def to_tensor(X, use_cuda=False):
    """Turn to torch Variable.

    Handles the cases:
      * Variable
      * PackedSequence
      * dict
      * numpy array
      * torch Tensor

    """
    if isinstance(X, (Variable, nn.utils.rnn.PackedSequence)):
        return X

    if isinstance(X, dict):
        return {key: to_tensor(val) for key, val in X.items()}

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if isinstance(X, collections.abc.Sequence):
        X = torch.from_numpy(np.array(X))

    if use_cuda:
        X = X.cuda()
    return X


def to_numpy(X):
    if X.is_cuda:
        X = X.cpu()
    try:
        data = X.data
    except AttributeError:
        data = X
    return data.numpy()


def check_history_slice(history, sl):
    # Note: May extend this for more cases.
    try:
        history[sl]
        return
    except KeyError as exc:
        msg = ("Key '{}' could not be found in history; "
               "maybe there was a typo?".format(exc.args[0]))
        raise KeyError(msg)
