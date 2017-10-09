"""Contains custom skorch Dataset and CVSplit."""

from functools import partial
from numbers import Number

import numpy as np
from sklearn.utils import safe_indexing
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import check_cv
import torch
import torch.utils.data

from skorch.utils import flatten
from skorch.utils import is_pandas_ndframe
from skorch.utils import to_numpy
from skorch.utils import to_tensor


def _apply_to_data(data, func, unpack_dict=False):
    """Apply a function to data, trying to unpack different data
    types.

    """
    apply_ = partial(_apply_to_data, func=func, unpack_dict=unpack_dict)
    if isinstance(data, dict):
        if unpack_dict:
            return [apply_(v) for v in data.values()]
        return {k: apply_(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        try:
            # e.g.list/tuple of arrays
            return [apply_(x) for x in data]
        except TypeError:
            return func(data)
    return func(data)


def get_len(data):
    lens = [_apply_to_data(data, len, unpack_dict=True)]
    lens = list(flatten(lens))
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
            i = tuple(j.tolist() for j in i.nonzero())
        elif i.dtype == int:
            i = i.tolist()
        else:
            raise IndexError("arrays used as indices must be of integer "
                             "(or boolean) type")
    if isinstance(i, list):
        i = (i,)

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

        # pylint: disable=invalid-name
        len_X = get_len(X)
        if y is not None:
            len_y = get_len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    def transform(self, X, y):
        """Additional transformations on X and y.

        By default, they are cast to torch tensors. Override this if
        you want a different behavior.

        Note: If you use this in conjuction with pytorch's DataLoader,
        the latter will call the dataset for each row separately,
        which means that the incoming X and y each are single rows.

        """
        # pytorch DataLoader cannot deal with None so we use 0 as a
        # placeholder value. We only return a Tensor with one value
        # (as opposed to `batchsz` values) since the pytorch
        # DataLoader calls __getitem__ for each row in the batch
        # anyway, which results in a dummy `y` value for each row in
        # the batch.
        y = torch.Tensor([0]) if y is None else y

        return (
            to_tensor(X, use_cuda=self.use_cuda),
            to_tensor(y, use_cuda=self.use_cuda),
        )

    def __getitem__(self, i):
        X, y = self.X, self.y
        if is_pandas_ndframe(X):
            X = {k: X[k].values.reshape(-1, 1) for k in X}

        xi = multi_indexing(X, i)
        yi = y if y is None else multi_indexing(y, i)
        return self.transform(xi, yi)


class CVSplit(object):
    """Class that performs the internal train/valid split.

    The `cv` argument here works similarly to the regular sklearn `cv`
    parameter in, e.g., `GridSearchCV`. However, instead of cycling
    through all splits, only one fixed split (the first one) is
    used. To get a full cycle through the splits, don't use
    `NeuralNet`'s internal validation but instead the corresponding
    sklearn functions (e.g. `cross_val_score`).

    We additionally support a float, similar to sklearn's
    `train_test_split`.

    Parameters
    ----------
    cv : int, float, cross-validation generator or an iterable, optional
      (Refer sklearn's User Guide for cross_validation for the various
      cross-validation strategies that can be used here.)

      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - float, to represent the proportion of the dataset to include
          in the test split.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

    stratified : bool (default=False)
      Whether the split should be stratified. Only works if `y` is
      either binary or multiclass classification.

    random_state : int, RandomState instance, or None (default=None)
      Control the random state in case that `(Stratified)ShuffleSplit`
      is used (which is when a float is passed to `cv`). For more
      information, look at the sklearn documentation of
      `(Stratified)ShuffleSplit`.

      """
    def __init__(
            self,
            cv=5,
            stratified=False,
            random_state=None,
    ):
        self.stratified = stratified
        self.random_state = random_state

        if isinstance(cv, Number) and (cv <= 0):
            raise ValueError("Numbers less than 0 are not allowed for cv "
                             "but CVSplit got {}".format(cv))
        self.cv = cv

    def _is_stratified(self, cv):
        return isinstance(cv, (StratifiedKFold, StratifiedShuffleSplit))

    def _is_float(self, x):
        if not isinstance(x, Number):
            return False
        return not float(x).is_integer()

    def _check_cv_float(self):
        cv_cls = StratifiedShuffleSplit if self.stratified else ShuffleSplit
        return cv_cls(test_size=self.cv, random_state=self.random_state)

    def _check_cv_non_float(self, y):
        return check_cv(
            self.cv,
            y=y,
            classifier=self.stratified,
        )

    def check_cv(self, y):
        """Resolve which cross validation strategy is used."""
        y_arr = None
        if self.stratified:
            # Try to convert y to numpy for sklearn's check_cv; if conversion
            # doesn't work, still try.
            try:
                y_arr = to_numpy(y)
            except AttributeError:
                y_arr = y

        if self._is_float(self.cv):
            return self._check_cv_float()
        return self._check_cv_non_float(y_arr)

    def _is_regular(self, x):
        return (x is None) or isinstance(x, np.ndarray) or is_pandas_ndframe(x)

    def __call__(self, X, y):
        bad_y_error = ValueError("Stratified CV not possible with given y.")
        if (y is None) and self.stratified:
            raise bad_y_error

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise bad_y_error

        # pylint: disable=invalid-name
        len_X = get_len(X)
        if y is not None:
            len_y = get_len(y)
            if len_X != len_y:
                raise ValueError("Cannot perform a CV split if X and y "
                                 "have different lengths.")

        args = (np.arange(len_X),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args)))
        X_train = multi_indexing(X, idx_train)
        X_valid = multi_indexing(X, idx_valid)
        y_train = None if y is None else multi_indexing(y, idx_train)
        y_valid = None if y is None else multi_indexing(y, idx_valid)
        return X_train, X_valid, y_train, y_valid

    def __repr__(self):
        # TODO
        # pylint: disable=useless-super-delegation
        return super(CVSplit, self).__repr__()
