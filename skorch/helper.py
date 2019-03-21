"""Helper functions and classes for users.

They should not be used in skorch directly.

"""
from functools import partial
import warnings

import numpy as np
from torch.utils.data.dataloader import default_collate

from skorch.utils import _make_split
from skorch.utils import _make_optimizer
from skorch.cli import parse_args
from skorch.utils import is_torch_data_type


class SliceDict(dict):
    """Wrapper for Python dict that makes it sliceable across values.

    Use this if your input data is a dictionary and you have problems
    with sklearn not being able to slice it. Wrap your dict with
    SliceDict and it should usually work.

    Note:
    * SliceDict cannot be indexed by integers, if you want one row,
      say row 3, use `[3:4]`.
    * SliceDict accepts numpy arrays and torch tensors as values.

    Examples
    --------
    >>> X = {'key0': val0, 'key1': val1}
    >>> search = GridSearchCV(net, params, ...)
    >>> search.fit(X, y)  # raises error
    >>> Xs = SliceDict(key0=val0, key1=val1)  # or Xs = SliceDict(**X)
    >>> search.fit(Xs, y)  # works

    """
    def __init__(self, **kwargs):
        lengths = [value.shape[0] for value in kwargs.values()]
        lengths_set = set(lengths)
        if lengths_set and (len(lengths_set) != 1):
            raise ValueError(
                "Initialized with items of different lengths: {}"
                "".format(', '.join(map(str, sorted(lengths_set)))))

        if not lengths:
            self._len = 0
        else:
            self._len = lengths[0]

        super(SliceDict, self).__init__(**kwargs)

    def __len__(self):
        return self._len

    def __getitem__(self, sl):
        if isinstance(sl, int):
            # Indexing with integers is not well-defined because that
            # recudes the dimension of arrays by one, messing up
            # lengths and shapes.
            raise ValueError("SliceDict cannot be indexed by integers.")
        if isinstance(sl, str):
            return super(SliceDict, self).__getitem__(sl)
        return SliceDict(**{k: v[sl] for k, v in self.items()})

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Key must be str, not {}.".format(type(key)))

        length = value.shape[0]
        if not self.keys():
            self._len = length

        if self._len != length:
            raise ValueError(
                "Cannot set array with shape[0] != {}"
                "".format(self._len))

        super(SliceDict, self).__setitem__(key, value)

    def update(self, kwargs):
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def __repr__(self):
        out = super(SliceDict, self).__repr__()
        return "SliceDict(**{})".format(out)

    @property
    def shape(self):
        return (self._len,)

    def copy(self):
        return type(self)(**self)

    def fromkeys(self, *args, **kwargs):
        """fromkeys method makes no sense with SliceDict and is thus not
        supported."""
        raise TypeError("SliceDict does not support fromkeys.")

    def __eq__(self, other):
        if self.keys() != other.keys():
            return False

        for key, val in self.items():
            val_other = other[key]

            # torch tensors
            if is_torch_data_type(val):
                if not is_torch_data_type(val_other):
                    return False
                if not (val == val_other).all():
                    return False
                continue

            # numpy arrays
            if isinstance(val, np.ndarray):
                if not isinstance(val_other, np.ndarray):
                    return False
                if not (val == val_other).all():
                    return False
                continue

            # rest
            if val != val_other:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)


def _unpack_first(x):
    if isinstance(x, tuple):
        return x[0]
    return x


class SliceDatasetX:
    """Helper class that wraps a torch dataset to make it work with
    sklearn.

    Sometimes, sklearn will touch the input data, e.g. when splitting
    the data for a grid search. This will fail when the input data is
    a torch dataset. To prevent this, use this wrapper class for your
    dataset.

    Note that this class will only return the X value (i.e. the first
    value returned by indexing the original dataset). Sklearn, and
    hence skorch, always require 2 values, X and y. Therefore, you
    still need to provide the y data separately.

    Examples
    --------
    >>> X = MyCustomDataset()
    >>> search = GridSearchCV(net, params, ...)
    >>> search.fit(X, y)  # raises error
    >>> ds = SliceDatasetX(X)  # or Xs = SliceDict(**X)
    >>> search.fit(ds, y)  # works

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
      A valid torch dataset.

    collate_fn : callable (default=torch.utils.data.dataloader.default_collate)
      A function that merges a list of samples to form a
      mini-batch. This is typically the same function that torch's
      DataLoader class uses to collate batches.

    """
    def __init__(self, dataset, collate_fn=default_collate):
        self.dataset = dataset
        self.collate_fn = collate_fn

        self._indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    @property
    def shape(self):
        return (len(self),)

    def transform(self, X):
        return X

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xi = _unpack_first(self.dataset[i])
            return self.transform(Xi)

        if isinstance(i, slice):
            i = self._indices[i]

        if isinstance(i, np.ndarray):
            if i.ndim != 1:
                raise IndexError("SliceDatasetX only supports slicing with 1 "
                                 "dimensional arrays, got {} dimensions instead."
                                 "".format(i.ndim))
            if i.dtype == np.bool:
                i = np.flatnonzero(i)

        Xi = self.collate_fn([
            self.transform(_unpack_first(self.dataset[j])) for j in i])
        return Xi


# TODO: remove in 0.5.0
def filter_requires_grad(pgroups):
    """Returns parameter groups where parameters
    that don't require a gradient are filtered out.

    Parameters
    ----------
    pgroups : dict
      Parameter groups to be filtered

    """
    warnings.warn(
        "For filtering gradients, please use skorch.callbacks.Freezer.",
        DeprecationWarning)

    for pgroup in pgroups:
        output = {k: v for k, v in pgroup.items() if k != 'params'}
        output['params'] = (p for p in pgroup['params'] if p.requires_grad)
        yield output


# TODO: remove in 0.5.0
def filtered_optimizer(optimizer, filter_fn):
    """Wraps an optimizer that filters out parameters where
    ``filter_fn`` over ``pgroups`` returns ``False``.
    This function can be used, for example, to filter parameters
    that do not require a gradient:

    >>> from skorch.helper import filtered_optimizer, filter_requires_grad
    >>> optimizer = filtered_optimizer(torch.optim.SGD, filter_requires_grad)
    >>> net = NeuralNetClassifier(module, optimizer=optimizer)

    Parameters
    ----------
    optimizer : torch optim (class)
      The uninitialized optimizer that is wrapped

    filter_fn : function
      Use this function to filter parameter groups before passing
      it to ``optimizer``.

    """
    warnings.warn(
        "For filtering gradients, please use skorch.callbacks.Freezer.",
        DeprecationWarning)

    return partial(_make_optimizer, optimizer=optimizer, filter_fn=filter_fn)


def predefined_split(dataset):
    """Uses ``dataset`` for validiation in ``NeutralNet``.

    Examples
    --------
    >>> valid_ds = skorch.Dataset(X, y)
    >>> net = NeutralNet(..., train_split=predefined_split(valid_ds))

    Parameters
    ----------
    dataset: torch Dataset
       Validiation dataset

    """
    return partial(_make_split, valid_ds=dataset)
