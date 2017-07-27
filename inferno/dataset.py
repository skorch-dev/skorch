from functools import partial

from sklearn.utils import safe_indexing
import torch

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
        return data[i]
    if hasattr(data, 'iloc'):
        # pandas NDFrame
        return data.iloc[i]
    # numpy ndarray, list
    if isinstance(i, (int, slice)):
        return data[i]
    return safe_indexing(data, i)


class Dataset(object):
    """General dataset wrapper that can be used in conjunction with
    pytorch's DataLoader.

    The dataset will always yield two values, the data and the
    target. However, the target is allowed to be None. In that case,
    Dataset will currently return a dummy tensor, since DataLoader
    does not work with Nones.

    Currently works with the following data types:

    * numpy arrays
    * torch tensors
    * a dictionary of the former two
    * a list/tuple of the former two

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
        if hasattr(X, 'iloc'):
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
