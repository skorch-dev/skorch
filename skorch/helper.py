"""Helper functions and classes for users.

They should not be used in skorch directly.

"""
from collections.abc import Sequence
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import torch

from skorch.cli import parse_args  # pylint: disable=unused-import
from skorch.dataset import unpack_data
from skorch.utils import _make_split
from skorch.utils import is_torch_data_type
from skorch.utils import to_tensor


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
    """Uses ``dataset`` for validiation in :class:`.NeuralNet`.

    Examples
    --------
    >>> valid_ds = skorch.dataset.Dataset(X, y)
    >>> net = NeuralNet(..., train_split=predefined_split(valid_ds))

    Parameters
    ----------
    dataset: torch Dataset
       Validiation dataset

    """
    return partial(_make_split, valid_ds=dataset)


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Transform a DataFrame into a dict useful for working with skorch.

    Transforms cardinal data to floats and categorical data to vectors
    of ints so that they can be embedded.

    Although skorch can deal with pandas DataFrames, the default
    behavior is often not very useful. Use this transformer to
    transform the DataFrame into a dict with all float columns
    concatenated using the key "X" and all categorical values encoded
    as integers, using their respective column names as keys.

    Your module must have a matching signature for this to work. It
    must accept an argument ``X`` for all cardinal
    values. Additionally, for all categorical values, it must accept
    an argument with the same name as the corresponding column (see
    example below). If you need help with the required signature, use
    the ``describe_signature`` method of this class and pass it your
    data.

    You can choose whether you want to treat int columns the same as
    float columns (default) or as categorical values.

    To one-hot encode categorical features, initialize their
    corresponding embedding layers using the identity matrix.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'col_floats': np.linspace(0, 1, 12),
    ...     'col_ints': [11, 11, 10] * 4,
    ...     'col_cats': ['a', 'b', 'a'] * 4,
    ... })
    >>> # cast to category dtype to later learn embeddings
    >>> df['col_cats'] = df['col_cats'].astype('category')
    >>> y = np.asarray([0, 1, 0] * 4)

    >>> class MyModule(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.reset_params()

    >>>     def reset_params(self):
    ...         self.embedding = nn.Embedding(2, 10)
    ...         self.linear = nn.Linear(2, 10)
    ...         self.out = nn.Linear(20, 2)
    ...         self.nonlin = nn.Softmax(dim=-1)

    >>>     def forward(self, X, col_cats):
    ...         # "X" contains the values from col_floats and col_ints
    ...         # "col_cats" contains the values from "col_cats"
    ...         X_lin = self.linear(X)
    ...         X_cat = self.embedding(col_cats)
    ...         X_concat = torch.cat((X_lin, X_cat), dim=1)
    ...         return self.nonlin(self.out(X_concat))

    >>> net = NeuralNetClassifier(MyModule)
    >>> pipe = Pipeline([
    ...     ('transform', DataFrameTransformer()),
    ...     ('net', net),
    ... ])
    >>> pipe.fit(df, y)

    Parameters
    ----------
    treat_int_as_categorical : bool (default=False)
      Whether to treat integers as categorical values or as cardinal
      values, i.e. the same as floats.

    float_dtype : numpy dtype or None (default=np.float32)
      The dtype to cast the cardinal values to. If None, don't change
      them.

    int_dtype : numpy dtype or None (default=np.int64)
      The dtype to cast the categorical values to. If None, don't
      change them. If you do this, it can happen that the categorical
      values will have different dtypes, reflecting the number of
      unique categories.

    Notes
    -----
    The value of X will always be 2-dimensional, even if it only
    contains 1 column.

    """
    import pandas as pd

    def __init__(
            self,
            treat_int_as_categorical=False,
            float_dtype=np.float32,
            int_dtype=np.int64,
    ):
        self.treat_int_as_categorical = treat_int_as_categorical
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

    def _check_dtypes(self, df):
        """Perform a check on the DataFrame to detect wrong dtypes or keys.

        Makes sure that there are no conflicts in key names.

        If dtypes are found that cannot be dealt with, raises a
        TypeError with a message indicating which ones caused trouble.

        Raises
        ------
        ValueError
          If there already is a column named 'X'.

        TypeError
          If a wrong dtype is found.

        """
        if 'X' in df:
            raise ValueError(
                "DataFrame contains a column named 'X', which clashes "
                "with the name chosen for cardinal features; consider "
                "renaming that column.")

        wrong_dtypes = []

        for col, dtype in zip(df, df.dtypes):
            if isinstance(dtype, self.pd.api.types.CategoricalDtype):
                continue
            if np.issubdtype(dtype, np.integer):
                continue
            if np.issubdtype(dtype, np.floating):
                continue
            wrong_dtypes.append((col, dtype))

        if not wrong_dtypes:
            return

        wrong_dtypes = sorted(wrong_dtypes, key=lambda tup: tup[0])
        msg_dtypes = ", ".join(
            "{} ({})".format(col, dtype) for col, dtype in wrong_dtypes)
        msg = ("The following columns have dtypes that cannot be "
               "interpreted as numerical dtypes: {}".format(msg_dtypes))
        raise TypeError(msg)

    # pylint: disable=unused-argument
    def fit(self, df, y=None, **fit_params):
        self._check_dtypes(df)
        return self

    def transform(self, df):
        """Transform DataFrame to become a dict that works well with skorch.

        Parameters
        ----------
        df : pd.DataFrame
          Incoming DataFrame.

        Returns
        -------
        X_dict: dict
          Dictionary with all floats concatenated using the key "X"
          and all categorical values encoded as integers, using their
          respective column names as keys.

        """
        self._check_dtypes(df)

        X_dict = {}
        Xf = []  # floats

        for col, dtype in zip(df, df.dtypes):
            X_col = df[col]

            if isinstance(dtype, self.pd.api.types.CategoricalDtype):
                x = X_col.cat.codes.values
                if self.int_dtype is not None:
                    x = x.astype(self.int_dtype)
                X_dict[col] = x
                continue

            if (
                    np.issubdtype(dtype, np.integer)
                    and self.treat_int_as_categorical
            ):
                x = X_col.astype('category').cat.codes.values
                if self.int_dtype is not None:
                    x = x.astype(self.int_dtype)
                X_dict[col] = x
                continue

            Xf.append(X_col.values)

        if not Xf:
            return X_dict

        X = np.stack(Xf, axis=1)
        if self.float_dtype is not None:
            X = X.astype(self.float_dtype)
        X_dict['X'] = X
        return X_dict

    def describe_signature(self, df):
        """Describe the signature required for the given data.

        Pass the DataFrame to receive a description of the signature
        required for the module's forward method. The description
        consists of three parts:

        1. The names of the arguments that the forward method
        needs.
        2. The dtypes of the torch tensors passed to forward.
        3. The number of input units that are required for the
        corresponding argument. For the float parameter, this is just
        the number of dimensions of the tensor. For categorical
        parameters, it is the number of unique elements.

        Returns
        -------
        signature : dict
          Returns a dict with each key corresponding to one key
          required for the forward method. The values are dictionaries
          of two elements. The key "dtype" describes the torch dtype
          of the resulting tensor, the key "input_units" describes the
          required number of input units.

        """
        X_dict = self.fit_transform(df)
        signature = {}

        X = X_dict.get('X')
        if X is not None:
            signature['X'] = dict(
                dtype=to_tensor(X, device='cpu').dtype,
                input_units=X.shape[1],
            )

        for key, val in X_dict.items():
            if key == 'X':
                continue

            tensor = to_tensor(val, device='cpu')
            nunique = len(torch.unique(tensor))
            signature[key] = dict(
                dtype=tensor.dtype,
                input_units=nunique,
            )

        return signature


class AccelerateMixin:
    """Mixin class to add support for huggingface accelerate

    This is an *experimental* feature.

    Use this mixin class with one of the neural net classes (e.g. ``NeuralNet``,
    ``NeuralNetClassifier``, or ``NeuralNetRegressor``) and pass an instance of
    ``Accelerator`` for mixed precision, multi-GPU, or TPU training.

    Install the accelerate library using:

    .. code-block::

      python -m pip install accelerate

    skorch does not itself provide any facilities to enable these training
    features. A lot of them can still be implemented by the user with a little
    bit of extra work but it can be a daunting task. That is why this helper
    class was added: Using this mixin in conjunction with the accelerate library
    should cover a lot of common use cases.

    .. note::

        Under the hood, accelerate uses :class:`~torch.cuda.amp.GradScaler`,
        which does not support passing the training step as a closure.
        Therefore, if your optimizer requires that (e.g.
        :class:`torch.optim.LBFGS`), you cannot use accelerate.

    .. warning::

        Since accelerate is still quite young and backwards compatiblity
        breaking features might be added, we treat its integration as an
        experimental feature. When accelerate's API stabilizes, we will consider
        adding it to skorch proper.

    Examples
    --------
    >>> from skorch import NeuralNetClassifier
    >>> from skorch.helper import AccelerateMixin
    >>> from accelerate import Accelerator
    >>>
    >>> class AcceleratedNet(AccelerateMixin, NeuralNetClassifier):
    ...     '''NeuralNetClassifier with accelerate support'''
    >>>
    >>> accelerator = Accelerator(...)
    >>> net = AcceleratedNet(MyModule,  accelerator=accelerator)
    >>> net.fit(X, y)

    The same approach works with all the other skorch net classes.

    Parameters
    ----------
    accelerator : accelerate.Accelerator
      In addition to the usual parameters, pass an instance of
      ``accelerate.Accelerator`` with the desired settings.

    device : str, torch.device, or None (default=None)
      The compute device to be used. When using accelerate, it is recommended to
      leave device handling to accelerate. Therefore, it is best to leave this
      argument to be None, which means that skorch does not set the device.

    callbacks__print_log__sink : 'auto' or callable
      If 'auto', uses the ``print`` function of the accelerator, if it has one.
      This avoids printing the same output multiple times when training
      concurrently on multiple machines. If the accelerator does not have a
      ``print`` function, use Python's ``print`` function instead.

    """
    def __init__(
            self,
            *args,
            accelerator,
            device=None,
            callbacks__print_log__sink='auto',
            **kwargs
    ):
        super().__init__(
            *args,
            device=device,
            callbacks__print_log__sink=callbacks__print_log__sink,
            **kwargs
        )
        self.accelerator = accelerator

    def _check_kwargs(self, kwargs):
        super()._check_kwargs(kwargs)

        if self.accelerator.device_placement and (self.device is not None):
            raise ValueError(
                "When device placement is performed by the accelerator, set device=None"
            )

    def _initialize_callbacks(self):
        if self.callbacks__print_log__sink == 'auto':
            print_func = getattr(self.accelerator, 'print', print)
            self.callbacks__print_log__sink = print_func
        super()._initialize_callbacks()
        return self

    def _initialize_criterion(self, *args, **kwargs):
        super()._initialize_criterion(*args, **kwargs)

        with self._current_init_context('criterion'):
            for name in self._criteria:
                criterion = getattr(self, name + '_')
                if isinstance(criterion, torch.nn.Module):
                    setattr(self, name + '_', self.accelerator.prepare(criterion))

        return self

    def _initialize_module(self, *args, **kwargs):
        super()._initialize_module(*args, **kwargs)

        with self._current_init_context('module'):
            for name in self._modules:
                module = getattr(self, name + '_')
                if isinstance(module, torch.nn.Module):
                    setattr(self, name + '_', self.accelerator.prepare(module))

        return self

    def _initialize_optimizer(self, *args, **kwargs):
        super()._initialize_optimizer(*args, **kwargs)

        with self._current_init_context('optimizer'):
            for name in self._optimizers:
                optimizer = getattr(self, name + '_')
                if isinstance(optimizer, torch.optim.Optimizer):
                    setattr(self, name + '_', self.accelerator.prepare(optimizer))

        return self

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        self.accelerator.backward(loss)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def get_iterator(self, *args, **kwargs):
        iterator = super().get_iterator(*args, **kwargs)
        iterator = self.accelerator.prepare(iterator)
        return iterator

    def _step_optimizer(self, step_fn):
        # We cannot step_fn as a 'closure' to .step because GradScaler doesn't
        # suppor it:
        # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step
        # Therefore, we need to call step_fn explicitly and step without
        # argument.
        step_fn()
        for name in self._optimizers:
            optimizer = getattr(self, name + '_')
            optimizer.step()

    # pylint: disable=unused-argument
    def on_train_end(self, net, X=None, y=None, **kwargs):
        super().on_train_end(net, X=X, y=y, **kwargs)
        self.module_ = self.accelerator.unwrap_model(self.module_)
