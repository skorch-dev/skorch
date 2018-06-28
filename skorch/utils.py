"""skorch utilities.

Should not have any dependency on other skorch packages.

"""

from collections.abc import Sequence
from contextlib import contextmanager
from enum import Enum
from functools import partial
import pathlib

import numpy as np
from sklearn.utils import safe_indexing
import torch
from torch import nn
from torch.utils.data.dataset import Subset


class Ansi(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, torch.Tensor)


def is_dataset(x):
    return isinstance(x, torch.utils.data.Dataset)


# pylint: disable=not-callable
def to_tensor(X, device):
    """Turn to torch tensor.

    Handles the cases:
      * PackedSequence
      * numpy array
      * torch Tensor
      * list or tuple of one of the former
      * dict of one of the former

    """
    assert not isinstance(device, bool), (
        "possible bug: used `device` parameter like `use_cuda`. "
        "Set `device='cuda'` instead.")
    to_tensor_ = partial(to_tensor, device=device)

    if is_torch_data_type(X):
        return X.to(device)
    elif isinstance(X, dict):
        return {key: to_tensor_(val) for key, val in X.items()}
    elif isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]
    elif np.isscalar(X):
        return torch.tensor(X).to(device)
    elif isinstance(X, Sequence):
        return torch.tensor(np.array(X)).to(device)
    elif isinstance(X, np.ndarray):
        return torch.tensor(X).to(device)
    elif isinstance(X, nn.utils.rnn.PackedSequence):
        return X
    else:
        raise TypeError("Cannot convert this data type to a torch tensor.")


def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy.

    Returns X when it already is a numpy array.

    """
    if isinstance(X, np.ndarray):
        return X

    if is_pandas_ndframe(X):
        return X.values

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


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


# pylint: disable=unused-argument
def _indexing_none(data, i):
    return None


def _indexing_dict(data, i):
    return {k: v[i] for k, v in data.items()}


def _indexing_list_tuple_of_data(data, i, indexings=None):
    """Data is a list/tuple of data structures (e.g. list of numpy arrays).

    ``indexings`` are the indexing functions for each element of
    ``data``. If ``indexings`` are not given, the indexing functions
    for the individual structures have to be determined ad hoc, which
    is slower.

    """
    if not indexings:
        return [multi_indexing(x, i) for x in data]
    return [multi_indexing(x, i, indexing)
            for x, indexing in zip(data, indexings)]


def _indexing_ndframe(data, i):
    # During fit, DataFrames are converted to dict, which is why we
    # might need _indexing_dict.
    if hasattr(data, 'iloc'):
        return data.iloc[i]
    return _indexing_dict(data, i)


def _indexing_other(data, i):
    if isinstance(i, (int, np.integer, slice)):
        return data[i]
    return safe_indexing(data, i)


def check_indexing(data):
    """Perform a check how incoming data should be indexed and return an
    appropriate indexing function with signature f(data, index).

    This is useful for determining upfront how data should be indexed
    instead of doing it repeatedly for each batch, thus saving some
    time.

    """
    if data is None:
        return _indexing_none

    if isinstance(data, dict):
        # dictionary of containers
        return _indexing_dict

    if isinstance(data, (list, tuple)):
        try:
            # list or tuple of containers
            # TODO: Is there a better way than just to try to index? This
            # is error prone (e.g. if one day list of strings are
            # possible).
            multi_indexing(data[0], 0)
            indexings = [check_indexing(x) for x in data]
            return partial(_indexing_list_tuple_of_data, indexings=indexings)
        except TypeError:
            # list or tuple of values
            return _indexing_other

    if is_pandas_ndframe(data):
        # pandas NDFrame, will be transformed to dict
        return _indexing_ndframe

    # torch tensor, numpy ndarray, list
    return _indexing_other


def _normalize_numpy_indices(i):
    """Normalize the index in case it is a numpy integer or boolean
    array."""
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
    return i


def multi_indexing(data, i, indexing=None):
    """Perform indexing on multiple data structures.

    Currently supported data types:

    * numpy arrays
    * torch tensors
    * pandas NDFrame
    * a dictionary of the former three
    * a list/tuple of the former three

    ``i`` can be an integer or a slice.

    Examples
    --------
    >>> multi_indexing(np.asarray([1, 2, 3]), 0)
    1

    >>> multi_indexing(np.asarray([1, 2, 3]), np.s_[:2])
    array([1, 2])

    >>> multi_indexing(torch.arange(0, 4), np.s_[1:3])
    tensor([ 1.,  2.])

    >>> multi_indexing([[1, 2, 3], [4, 5, 6]], np.s_[:2])
    [[1, 2], [4, 5]]

    >>> multi_indexing({'a': [1, 2, 3], 'b': [4, 5, 6]}, np.s_[-2:])
    {'a': [2, 3], 'b': [5, 6]}

    >>> multi_indexing(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [1, 2])
       a  b
    1  2  5
    2  3  6

    Parameters
    ----------
    data
      Data of a type mentioned above.

    i : int or slice
      Slicing index.

    indexing : function/callable or None (default=None)
      If not None, use this function for indexing into the data. If
      None, try to automatically determine how to index data.

    """
    # in case of i being a numpy array
    i = _normalize_numpy_indices(i)

    # If we already know how to index, use that knowledge
    if indexing is not None:
        return indexing(data, i)

    # If we don't know how to index, find out and apply
    return check_indexing(data)(data, i)


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


def params_for(prefix, kwargs):
    """Extract parameters that belong to a given sklearn module prefix from
    ``kwargs``. This is useful to obtain parameters that belong to a
    submodule.

    Examples
    --------
    >>> kwargs = {'encoder__a': 3, 'encoder__b': 4, 'decoder__a': 5}
    >>> params_for('encoder', kwargs)
    {'a': 3, 'b': 4}

    """
    if not prefix.endswith('__'):
        prefix += '__'
    return {key[len(prefix):]: val for key, val in kwargs.items()
            if key.startswith(prefix)}


# pylint: disable=invalid-name
class _none:
    pass


def data_from_dataset(dataset, X_indexing=None, y_indexing=None):
    """Try to access X and y attribute from dataset.

    Also works when dataset is a subset.

    Parameters
    ----------
    dataset : skorch.dataset.Dataset or torch.utils.data.dataset.Subset
      The incoming dataset should be a ``skorch.dataset.Dataset`` or a
      ``torch.utils.data.dataset.Subset`` of a
      ``skorch.dataset.Dataset``.

    X_indexing : function/callable or None (default=None)
      If not None, use this function for indexing into the X data. If
      None, try to automatically determine how to index data.

    y_indexing : function/callable or None (default=None)
      If not None, use this function for indexing into the y data. If
      None, try to automatically determine how to index data.

    """
    X, y = _none, _none

    if isinstance(dataset, Subset):
        X, y = data_from_dataset(
            dataset.dataset, X_indexing=X_indexing, y_indexing=y_indexing)
        X = multi_indexing(X, dataset.indices, indexing=X_indexing)
        y = multi_indexing(y, dataset.indices, indexing=y_indexing)
    elif hasattr(dataset, 'X') and hasattr(dataset, 'y'):
        X, y = dataset.X, dataset.y

    if (X is _none) or (y is _none):
        raise AttributeError("Could not access X and y from dataset.")
    return X, y


def is_skorch_dataset(ds):
    """Checks if the supplied dataset is an instance of
    ``skorch.dataset.Dataset`` even when it is nested inside
    ``torch.util.data.dataset.Subset``."""
    from skorch.dataset import Dataset
    if isinstance(ds, Subset):
        return is_skorch_dataset(ds.dataset)
    return isinstance(ds, Dataset)


# pylint: disable=unused-argument
def noop(*args, **kwargs):
    """No-op function that does nothing and returns ``None``.

    This is useful for defining scoring callbacks that do not need a
    target extractor.
    """
    pass


@contextmanager
def open_file_like(f, mode):
    """Wrapper for opening a file"""
    new_fd = isinstance(f, (str, pathlib.Path))
    if new_fd:
        f = open(f, mode)
    try:
        yield f
    finally:
        if new_fd:
            f.close()


# pylint: disable=unused-argument
def train_loss_score(net, X=None, y=None):
    return net.history[-1, 'batches', -1, 'train_loss']


# pylint: disable=unused-argument
def valid_loss_score(net, X=None, y=None):
    return net.history[-1, 'batches', -1, 'valid_loss']


class FirstStepAccumulator:
    """Store and retrieve the train step data.

    This class simply stores the first step value and returns it.

    For most uses, ``skorch.utils.FirstStepAccumulator`` is what you
    want, since the optimizer calls the train step exactly
    once. However, some optimizerss such as LBFGSs make more than one
    call. If in that case, you don't want the first value to be
    returned (but instead, say, the last value), implement your own
    accumulator and make sure it is returned by
    ``NeuralNet.get_train_step_accumulator`` method.

    """
    def __init__(self):
        self.step = None

    def store_step(self, step):
        """Store the first step."""
        if self.step is None:
            self.step = step

    def get_step(self):
        """Return the stored step."""
        return self.step


def _filter_requires_grad(pgroups):
    """Returns parameter groups where parameters with
    ``requires_grad==False`` are filtered out.
    """
    for pgroup in pgroups:
        output = {k: v for k, v in pgroup.items() if k != 'params'}
        output['params'] = (p for p in pgroup['params'] if p.requires_grad)
        yield output
