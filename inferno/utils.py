import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


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
