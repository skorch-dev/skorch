"""skorch utilities.

Should not have any dependency on other skorch packages.

"""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from functools import partial
import io
from itertools import tee
import pathlib
import warnings

import numpy as np
from scipy import sparse
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.utils import _safe_indexing as safe_indexing
from sklearn.utils.validation import check_is_fitted as sk_check_is_fitted
import torch
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data.dataset import Subset

from skorch.exceptions import DeviceWarning
from skorch.exceptions import NotInitializedError
from ._version import Version

try:
    import torch_geometric
    TORCH_GEOMETRIC_INSTALLED = True
except ImportError:
    TORCH_GEOMETRIC_INSTALLED = False


class Ansi(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, (torch.Tensor, PackedSequence))


def is_dataset(x):
    return isinstance(x, torch.utils.data.Dataset)


def is_geometric_data_type(x):
    from torch_geometric.data import Data
    return isinstance(x, Data)


# pylint: disable=not-callable
def to_tensor(X, device, accept_sparse=False):
    """Turn input data to torch tensor.

    Parameters
    ----------
    X : input data
      Handles the cases:
        * PackedSequence
        * numpy array
        * torch Tensor
        * scipy sparse CSR matrix
        * list or tuple of one of the former
        * dict with values of one of the former

    device : str, torch.device
      The compute device to be used. If set to 'cuda', data in torch
      tensors will be pushed to cuda tensors before being sent to the
      module.

    accept_sparse : bool (default=False)
      Whether to accept scipy sparse matrices as input. If False,
      passing a sparse matrix raises an error. If True, it is
      converted to a torch COO tensor.

    Returns
    -------
    output : torch Tensor

    """
    to_tensor_ = partial(to_tensor, device=device)

    if is_torch_data_type(X):
        return to_device(X, device)
    if TORCH_GEOMETRIC_INSTALLED and is_geometric_data_type(X):
        return to_device(X, device)
    if hasattr(X, 'convert_to_tensors'):
        # huggingface transformers BatchEncoding
        return X.convert_to_tensors('pt')
    if isinstance(X, Mapping):
        return {key: to_tensor_(val) for key, val in X.items()}
    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]
    if np.isscalar(X):
        return torch.as_tensor(X, device=device)
    if isinstance(X, Sequence):
        return torch.as_tensor(np.array(X), device=device)
    if isinstance(X, np.ndarray):
        return torch.as_tensor(X, device=device)
    if sparse.issparse(X):
        if accept_sparse:
            return torch.sparse_coo_tensor(
                X.nonzero(), X.data, size=X.shape).to(device)
        raise TypeError("Sparse matrices are not supported. Set "
                        "accept_sparse=True to allow sparse matrices.")

    raise TypeError("Cannot convert this data type to a torch tensor.")


def _is_slicedataset(X):
    # Cannot use isinstance because we don't want to depend on helper.py.
    return hasattr(X, 'dataset') and hasattr(X, 'idx') and hasattr(X, 'indices')


def to_numpy(X):
    """Generic function to convert a pytorch tensor to numpy.

    This function tries to unpack the tensor(s) from supported
    data structures (e.g., dicts, lists, etc.) but doesn't go
    beyond.

    Returns X when it already is a numpy array.

    """
    if isinstance(X, np.ndarray):
        return X

    if isinstance(X, Mapping):
        return {key: to_numpy(val) for key, val in X.items()}

    if is_pandas_ndframe(X):
        return X.values

    if isinstance(X, (tuple, list)):
        return type(X)(to_numpy(x) for x in X)

    if _is_slicedataset(X):
        return np.asarray(X)

    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if hasattr(X, 'is_mps') and X.is_mps:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()


def to_device(X, device):
    """Generic function to modify the device type of the tensor(s) or module.

    PyTorch distribution objects are left untouched, since they don't support an
    API to move between devices.

    Parameters
    ----------
    X : input data
        Deals with X being a:

         * torch tensor
         * tuple of torch tensors
         * dict of torch tensors
         * PackSequence instance
         * torch.nn.Module

    device : str, torch.device
        The compute device to be used. If device=None, return the input
        unmodified

    """
    if device is None:
        return X

    if isinstance(X, Mapping):
        # dict-like but not a dict
        return type(X)({key: to_device(val, device) for key, val in X.items()})

    # PackedSequence class inherits from a namedtuple
    if isinstance(X, (tuple, list)) and (type(X) != PackedSequence):
        return type(X)(to_device(x, device) for x in X)

    if isinstance(X, torch.distributions.distribution.Distribution):
        return X

    return X.to(device)


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
        if isinstance(item, (tuple, list, Mapping)):
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
    # sklearn's safe_indexing doesn't work with tuples since 0.22
    if isinstance(i, (int, np.integer, slice, tuple)):
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

    if isinstance(data, Mapping):
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
    dataset : skorch.dataset.Dataset or torch.utils.data.Subset
      The incoming dataset should be a ``skorch.dataset.Dataset`` or a
      ``torch.utils.data.Subset`` of a
      ``skorch.dataset.Dataset``.

    X_indexing : function/callable or None (default=None)
      If not None, use this function for indexing into the X data. If
      None, try to automatically determine how to index data.

    y_indexing : function/callable or None (default=None)
      If not None, use this function for indexing into the y data. If
      None, try to automatically determine how to index data.

    Raises
    ------
    AttributeError
      If X and y could not be accessed from the dataset.

    """
    X, y = _none, _none

    if isinstance(dataset, Subset):
        X, y = data_from_dataset(
            dataset.dataset, X_indexing=X_indexing, y_indexing=y_indexing)
        X = multi_indexing(X, dataset.indices, indexing=X_indexing)
        y = multi_indexing(y, dataset.indices, indexing=y_indexing)
    elif hasattr(dataset, 'X') and hasattr(dataset, 'y'):
        X, y = dataset.X, dataset.y
    elif isinstance(dataset, torch.utils.data.dataset.TensorDataset):
        if len(items := dataset.tensors) == 2:
            X, y = items

    if (X is _none) or (y is _none):
        raise AttributeError("Could not access X and y from dataset.")
    return X, y


def is_skorch_dataset(ds):
    """Checks if the supplied dataset is an instance of
    ``skorch.dataset.Dataset`` even when it is nested inside
    ``torch.util.data.Subset``."""
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


@contextmanager
def open_file_like(f, mode):
    """Wrapper for opening a file"""
    if isinstance(f, (str, pathlib.Path)):
        file_like = open(f, mode)
    else:
        file_like = f

    try:
        yield file_like
    finally:
        file_like.close()


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


def _make_split(X, y=None, valid_ds=None, **kwargs):
    """Used by ``predefined_split`` to allow for pickling"""
    return X, valid_ds


def freeze_parameter(param):
    """Convenience function to freeze a passed torch parameter.
    Used by ``skorch.callbacks.Freezer``
    """
    param.requires_grad = False


def unfreeze_parameter(param):
    """Convenience function to unfreeze a passed torch parameter.
    Used by ``skorch.callbacks.Unfreezer``
    """
    param.requires_grad = True


def get_map_location(target_device, fallback_device='cpu'):
    """Determine the location to map loaded data (e.g., weights)
    for a given target device (e.g. 'cuda').
    """
    if target_device is None:
        target_device = fallback_device

    map_location = torch.device(target_device)

    # The user wants to use CUDA but there is no CUDA device
    # available, thus fall back to CPU.
    if map_location.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn(
            'Requested to load data to CUDA but no CUDA devices '
            'are available. Loading on device "{}" instead.'.format(
                fallback_device,
            ), DeviceWarning)
        map_location = torch.device(fallback_device)
    return map_location


def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """Checks whether the net is initialized.

    Note: This calls ``sklearn.utils.validation.check_is_fitted``
    under the hood, using exactly the same arguments and logic. The
    only difference is that this function has an adapted error message
    and raises a ``skorch.exception.NotInitializedError`` instead of
    an ``sklearn.exceptions.NotFittedError``.

    """
    try:
        sk_check_is_fitted(estimator, attributes, msg=msg, all_or_any=all_or_any)
    except NotFittedError as exc:
        if msg is None:
            msg = ("This %(name)s instance is not initialized yet. Call "
                   "'initialize' or 'fit' with appropriate arguments "
                   "before using this method.")

        raise NotInitializedError(msg % {'name': type(estimator).__name__}) from exc


def _identity(x):
    """Return input as is, the identity operation"""
    return x


def _make_2d_probs(prob):
    """Create a 2d probability array from a 1d vector

    This is needed because by convention, even for binary classification
    problems, sklearn expects 2 probabilities to be returned per row, one for
    class 0 and one for class 1.

    """
    y_proba = torch.stack((1 - prob, prob), 1)
    return y_proba


def _sigmoid_then_2d(x):
    """Transform 1-dim logits to valid y_proba

    Sigmoid is applied to x to transform it to probabilities. Then
    concatenate the probabilities with 1 - these probabilities to
    return a correctly formed ``y_proba``. This is required for
    sklearn, which expects probabilities to be 2d arrays whose sum
    along axis 1 is 1.0.

    Parameters
    ----------
    x : torch.tensor
      A 1 dimensional float torch tensor containing raw logits.

    Returns
    -------
    y_proba : torch.tensor
      A 2 dimensional float tensor of probabilities that sum up to 1
      on axis 1.

    """
    prob = torch.sigmoid(x)
    return _make_2d_probs(prob)


# TODO only needed if multiclass GP classfication is added
# def _transpose(x):
    # return x.T


# pylint: disable=protected-access
def _infer_predict_nonlinearity(net):
    """Infers the correct nonlinearity to apply for this net

    The nonlinearity is applied only when calling
    :func:`~skorch.classifier.NeuralNetClassifier.predict` or
    :func:`~skorch.classifier.NeuralNetClassifier.predict_proba`.

    """
    # Implementation: At the moment, this function "dispatches" only
    # based on the criterion, not the class of the net. We still pass
    # the whole net as input in case we want to modify this at a
    # future point in time.
    if len(net._criteria) != 1:
        # don't know which criterion to consider, don't try to guess
        return _identity

    criterion = getattr(net, net._criteria[0] + '_')
    # unwrap optimizer in case of torch.compile being used
    criterion = getattr(criterion, '_orig_mod', criterion)

    if isinstance(criterion, CrossEntropyLoss):
        return partial(torch.softmax, dim=-1)

    if isinstance(criterion, BCEWithLogitsLoss):
        return _sigmoid_then_2d

    if isinstance(criterion, BCELoss):
        return _make_2d_probs

    return _identity

    # TODO: Add the code below to _infer_predict_nonlinearity if multiclass GP
    # classfication is added.
    # likelihood = getattr(net, 'likelihood_', None)
    # if likelihood is None:
    #     return _identity
    # nonlin = _identity
    # try:
    #     import gpytorch
    #     if isinstance(likelihood, gpytorch.likelihoods.SoftmaxLikelihood):
    #         # SoftmaxLikelihood returns batch second order
    #         nonlin = _transpose
    # except ImportError:
    #     # there is no gpytorch install
    #     pass
    # except AttributeError:
    #     # gpytorch and pytorch are incompatible
    #     msg = (
    #         "Importing gpytorch failed. This is probably because its version is "
    #         "incompatible with the installed torch version. Please visit "
    #         "https://github.com/cornellius-gp/gpytorch#installation to check "
    #         "which versions are compatible"
    #     )
    #     warnings.warn(msg)
    # return nonlin


class TeeGenerator:
    """Stores a generator and calls ``tee`` on it to create new generators
    when ``TeeGenerator`` is iterated over to let you iterate over the given
    generator more than once.

    """
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.gen, it = tee(self.gen)
        yield from it


def _check_f_arguments(caller_name, **kwargs):
    """Check file name arguments and return them

    This is used for checking if arguments to, e.g., ``save_params``
    are correct.

    Parameters
    ----------
    caller_name : str
      Name of caller, is only required for the error message.

    kwargs : dict
      Keyword arguments that are intended to be checked.

    Returns
    -------
    kwargs_module : dict
      Keyword arguments for saving/loading modules.

    kwargs_other : dict
      Keyword arguments for saving/loading everything else.

    Raises
    ------
    TypeError
      There are two possibilities for arguments to be
      incorrect. First, if they're not called 'f_*'. Second, if both
      'f_params' and 'f_module' are passed, since those designate the
      same thing.

    """
    if kwargs.get('f_params') and kwargs.get('f_module'):
        raise TypeError("{} called with both f_params and f_module, please choose one"
                        .format(caller_name))

    kwargs_module = {}
    kwargs_other = {}
    keys_other = {'f_history', 'f_pickle'}
    for key, val in kwargs.items():
        if not key.startswith('f_'):
            raise TypeError(
                "{name} got an unexpected argument '{key}', did you mean 'f_{key}'?"
                .format(name=caller_name, key=key))

        if val is None:
            continue
        if key in keys_other:
            kwargs_other[key] = val
        else:
            # strip 'f_' prefix and attach '_', and normalize 'params' to 'module'
            # e.g. 'f_optimizer' becomes 'optimizer_', 'f_params' becomes 'module_'
            key = 'module_' if key == 'f_params' else key[2:] + '_'
            kwargs_module[key] = val
    return kwargs_module, kwargs_other


def get_default_torch_load_kwargs():
    """Returns the kwargs passed to torch.load that correspond to the current
    torch version.

    The plan is to switch from weights_only=False to True in PyTorch version
    2.6.0, but depending on what happens, this may require updating.

    """
    version_torch = Version(torch.__version__)
    version_default_switch = Version('2.6.0')
    if version_torch >= version_default_switch:
        return {"weights_only": True}
    return {"weights_only": False}
