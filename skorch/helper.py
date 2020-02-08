"""Helper functions and classes for users.

They should not be used in skorch directly.

"""
from collections import Sequence
from functools import partial

import numpy as np

from skorch.cli import parse_args
from skorch.utils import _make_split
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


# This class must be an instance of Sequence and have an ndim
# attribute because sklearn will test this.
class SliceDataset(Sequence):
    # pylint: disable=anomalous-backslash-in-string
    """Helper class that wraps a torch dataset to make it work with
    sklearn.

    Sometimes, sklearn will touch the input data, e.g. when splitting
    the data for a grid search. This will fail when the input data is
    a torch dataset. To prevent this, use this wrapper class for your
    dataset.

    Note: This class will only return the X value by default (i.e. the
    first value returned by indexing the original dataset). Sklearn,
    and hence skorch, always require 2 values, X and y. Therefore, you
    still need to provide the y data separately.

    Note: This class behaves similarly to a PyTorch
    :class:`~torch.utils.data.Subset` when it is indexed by a slice or
    numpy array: It will return another ``SliceDataset`` that
    references the subset instead of the actual values. Only when it
    is indexed by an int does it return the actual values. The reason
    for this is to avoid loading all data into memory when sklearn,
    for instance, creates a train/validation split on the
    dataset. Data will only be loaded in batches during the fit loop.

    Examples
    --------
    >>> X = MyCustomDataset()
    >>> search = GridSearchCV(net, params, ...)
    >>> search.fit(X, y)  # raises error
    >>> ds = SliceDataset(X)
    >>> search.fit(ds, y)  # works

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
      A valid torch dataset.

    idx : int (default=0)
      Indicates which element of the dataset should be
      returned. Typically, the dataset returns both X and y
      values. SliceDataset can only return 1 value. If you want to
      get X, choose idx=0 (default), if you want y, choose idx=1.

    indices : list, np.ndarray, or None (default=None)
      If you only want to return a subset of the dataset, indicate
      which subset that is by passing this argument. Typically, this
      can be left to be None, which returns all the data. See also
      :class:`~torch.utils.data.Subset`.

    """
    def __init__(self, dataset, idx=0, indices=None):
        self.dataset = dataset
        self.idx = idx
        self.indices = indices

        self.indices_ = (self.indices if self.indices is not None
                         else np.arange(len(self.dataset)))
        self.ndim = 1

    def __len__(self):
        return len(self.indices_)

    @property
    def shape(self):
        return (len(self),)

    def transform(self, data):
        """Additional transformations on ``data``.

        Note: If you use this in conjuction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``data`` is a single rows.

        """
        return data

    def _select_item(self, Xn):
        # Raise a custom error message when accessing out of
        # bounds. However, this will only trigger as soon as this is
        # indexed by an integer.
        try:
            return Xn[self.idx]
        except IndexError:
            name = self.__class__.__name__
            msg = ("{} is trying to access element {} but there are only "
                   "{} elements.".format(name, self.idx, len(Xn)))
            raise IndexError(msg)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xn = self.dataset[self.indices_[i]]
            Xi = self._select_item(Xn)
            return self.transform(Xi)

        if isinstance(i, slice):
            return SliceDataset(self.dataset, idx=self.idx, indices=self.indices_[i])

        if isinstance(i, np.ndarray):
            if i.ndim != 1:
                raise IndexError("SliceDataset only supports slicing with 1 "
                                 "dimensional arrays, got {} dimensions instead."
                                 "".format(i.ndim))
            if i.dtype == np.bool:
                i = np.flatnonzero(i)

        return SliceDataset(self.dataset, idx=self.idx, indices=self.indices_[i])


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
