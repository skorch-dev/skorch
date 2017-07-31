from functools import partial

import numpy as np
from sklearn.utils import safe_indexing
from sklearn.model_selection import check_cv
import torch
import torch.utils.data

from inferno.utils import is_pandas_ndframe
from inferno.utils import to_numpy
from inferno.utils import to_tensor


def _apply_to_data(data, func, unpack_dict=False):
    _apply = partial(_apply_to_data, func=func, unpack_dict=unpack_dict)
    if isinstance(data, dict):
        if unpack_dict:
            return [_apply(v) for v in data.values()]
        else:
            return {k: _apply(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        try:
            # e.g.list/tuple of arrays
            return [_apply(x) for x in data]
        except TypeError:
            return func(data)
    return func(data)


def _flatten(arr):
    for item in arr:
        if isinstance(item, (tuple, list)):
            yield from _flatten(item)
        else:
            yield item


def get_len(data):
    lens = [_apply_to_data(data, len, unpack_dict=True)]
    lens = list(_flatten(lens))
    len_set = set(lens)
    if len(len_set) != 1:
        raise ValueError("Dataset does not have consistent lengths.")
    return list(len_set)[0]


def multi_indexing(data, i):
    """Perform indexing on multiple data structures.

    Currently supported data types:

    * numpy arrays
    * torch tensors
    * pandas NDFrame
    * a dictionary of the former three
    * a list/tuple of the former three

    `i` can be an integer or a slice.

    Example
    -------
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

    >>> multi_indexing(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
       a  b
    1  2  5
    2  3  6

    """
    if isinstance(i, np.ndarray):
        if i.dtype == bool:
            i = i.nonzero()[0]
        if i.dtype == int:
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
    if torch.is_tensor(data):
        # torch tensor-like
        if isinstance(i, list):
            i = torch.LongTensor(i)
        return data[i]
    if is_pandas_ndframe(data):
        # pandas NDFrame
        return data.iloc[i]
    # numpy ndarray, list
    if isinstance(i, (int, slice)):
        return data[i]
    return safe_indexing(data, i)


class Dataset(torch.utils.data.Dataset):
    """General dataset wrapper that can be used in conjunction with
    pytorch's DataLoader.

    The dataset will always yield a tuple of two values, the first
    from the data (`X`) and the second from the target
    (`y`). However, the target is allowed to be None. In that case,
    Dataset will currently return a dummy tensor, since DataLoader
    does not work with Nones.

    Dataset currently works with the following data types:

    * numpy arrays
    * torch tensors
    * pandas NDFrame
    * a dictionary of the former three
    * a list/tuple of the former three

    Parameters
    ----------
    X : see above
      Everything pertaining to the input data.

    y : see above or None (default=None)
      Everything pertaining to the target, if there is anything.

    use_cuda : bool (default=False)
      Whether to use cuda.

    length : int or None (default=None)
      If not None, determines the length (`len`) of the data. Should
      usually be left at None, in which case the length is determined
      by the data itself.

    """
    def __init__(
            self,
            X,
            y=None,
            use_cuda=False,
            length=None,
    ):
        self.X = X
        self.y = y
        self.use_cuda = use_cuda

        if length is not None:
            self._length = length
            return

        len_X = get_len(X)
        if y is not None:
            len_y = get_len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        X, y = self.X, self.y
        if is_pandas_ndframe(X):
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        xi = multi_indexing(X, i)
        if y is None:
            # pytorch DataLoader cannot deal with None
            yi = torch.zeros(get_len(xi))
        else:
            yi = multi_indexing(y, i)
        return (
            to_tensor(xi, use_cuda=self.use_cuda),
            to_tensor(yi, use_cuda=self.use_cuda),
        )


class CVSplit(object):
    """Class that performs the internal train/valid split.

    The `cv` argument here works similarly to the regular sklearn `cv`
    parameter in, e.g., `GridSearchCV`. However, instead of cycling
    through all splits, only one fixed split (the first one) is
    used. To get a full cycle through the splits, don't use
    `NeuralNet`'s internal validation but instead the corresponding
    sklearn functions (e.g. `cross_val_score`).

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, optional
      (Refer sklearn's User Guide for cross_validation for the various
      cross-validation strategies that can be used here.)

      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

    classifier : bool (default=False)
      For integer/None inputs to cv, if the estimator is a classifier
      (e.g. `NeuralNetClassifier`) and `y` is either binary or
      multiclass, sklearn's StratifiedKFold is used. In all other
      cases, KFold is used.

    Yields
    ------
    TODO

    """
    def __init__(self, cv=5, classifier=False):
        self.cv = cv
        self.classifier = classifier

    def regular_cv(self, X, y, cv):
        """Use the normal `.split` interface for data types are
        supported by sklearn.

        """
        idx_train, idx_valid = next(iter(cv.split(X, y)))
        X_train = safe_indexing(X, idx_train)
        X_valid = safe_indexing(X, idx_valid)
        y_train = None if y is None else safe_indexing(y, idx_train)
        y_valid = None if y is None else safe_indexing(y, idx_valid)
        return X_train, X_valid, y_train, y_valid

    def special_cv(self, X, y, cv, stratified=False):
        """For data types not directly supported by sklearn, use
        custom split function.

        """
        dataset = Dataset(X, y)
        num_samples = len(dataset)
        args = (np.arange(num_samples),)
        if stratified:
            y_arr = to_numpy(y)
            args = args + (y_arr,)
        idx_train, idx_valid = next(iter(cv.split(*args)))

        X_train, y_train = dataset[idx_train]
        X_valid, y_valid = dataset[idx_valid]
        return X_train, X_valid, y_train, y_valid

    def _check_cv(self, y):
        # TODO: Find a better solution for this mess
        stratified = False
        if y is None:
            cv = check_cv(self.cv, classifier=self.classifier)
            return cv, stratified

        try:
            # for stratified split, y must be a numpy array
            y_arr = to_numpy(y)
        except AttributeError:
            y_arr = y

        try:
            cv = check_cv(self.cv, y_arr, classifier=self.classifier)
            stratified = True
        except ValueError:
            cv = check_cv(self.cv, classifier=self.classifier)
        return cv, stratified

    def __call__(self, X, y):
        cv, stratified = self._check_cv(y)
        if isinstance(X, np.ndarray) or is_pandas_ndframe(X):
            # regular sklearn case
            return self.regular_cv(X, y, cv)
        else:
            # sklearn cannot properly split
            return self.special_cv(X, y, cv, stratified=stratified)

    def __repr__(self):
        # TODO
        return super(CVSplit, self).__repr__()
