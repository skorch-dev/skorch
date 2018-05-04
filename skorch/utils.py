"""skorch utilities.

Should not have any dependency on other skorch packages.

"""

from collections.abc import Sequence
from enum import Enum
from functools import partial

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

    if isinstance(X, nn.utils.rnn.PackedSequence):
        return X

    if isinstance(X, dict):
        return {key: to_tensor_(val) for key, val in X.items()}

    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]

    if isinstance(X, np.ndarray):
        X = torch.tensor(X)

    if np.isscalar(X):
        # ugly work-around - torch constructor does not accept np scalars
        X = torch.tensor(np.array([X]))[0]

    if isinstance(X, Sequence):
        X = torch.tensor(np.array(X))

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a torch tensor.")

    return X.to(device)


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


def multi_indexing(data, i):
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
     1
     2
    [torch.FloatTensor of size 2]

    >>> multi_indexing([[1, 2, 3], [4, 5, 6]], np.s_[:2])
    [[1, 2], [4, 5]]

    >>> multi_indexing({'a': [1, 2, 3], 'b': [4, 5, 6]}, np.s_[-2:])
    {'a': [2, 3], 'b': [5, 6]}

    >>> multi_indexing(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), [1, 2])
       a  b
    1  2  5
    2  3  6

    """
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
        else:
            raise IndexError("arrays used as indices must be of integer "
                             "(or boolean) type")

    if isinstance(data, dict):
        # dictionary of containers
        return {k: v[i] for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        # list or tuple of containers
        try:
            return [multi_indexing(x, i) for x in data]
        except TypeError:
            pass
    if is_pandas_ndframe(data):
        # pandas NDFrame
        return data.iloc[i]
    # torch tensor, numpy ndarray, list
    if isinstance(i, (int, np.integer, slice)):
        return data[i]
    return safe_indexing(data, i)


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


def data_from_dataset(dataset):
    """Try to access X and y attribute from dataset.

    Also works when dataset is a subset.

    """
    X, y = _none, _none

    if isinstance(dataset, Subset):
        X, y = data_from_dataset(dataset.dataset)
        X = multi_indexing(X, dataset.indices)
        y = multi_indexing(y, dataset.indices) if y is not None else y
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
