import numpy as np
from skorch.net import NeuralNet
from skorch.dataset import unpack_data


def loss_scoring(net: NeuralNet, X, y=None):
    """
    loss_scoring(net, X, y=None)

    Computes the loss of ``net`` on data (``X``, ``y``).

    This function can be used to implement the ``score`` method for a
    :class:`.NeuralNet` through sub-classing. This is useful, for example, when
    combining skorch models with sklearn objects that rely on the model's
    ``score`` method. Below is an example using GridSearchCV.

    >>> class ScoredNet(skorch.NeuralNetClassifier):
    ...     def score(self, X, y=None, lower_is_better=False):
    ...         loss_value = loss_scoring(self, X, y)
    ...         if lower_is_better:
    ...             return loss_value
    ...         return -loss_value
    ...
    >>> X = np.random.randn(250, 25).astype('float32')
    >>> y = (X.dot(np.ones(25)) > 0).astype(int)
    >>> module = nn.Sequential(
    ...     nn.Linear(25, 25),
    ...     nn.ReLU(),
    ...     nn.Linear(25,
    ...     2),
    ...     nn.Softmax(dim=1)
    ... )
    >>> net = ScoredNet(module)
    >>> grid_searcher = GridSearchCV(
    ...     net, {'lr': [1e-2, 1e-3], 'batch_size': [8, 16]}
    ... )
    >>> grid_searcher.fit(X, y)

    Parameters
    ----------
    net : skorch.NeuralNet
        A fitted Skorch :class:`.NeuralNet` object.
    X : input data, compatible with skorch.dataset.Dataset
        By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
        If this doesn't work with your data, you have to pass a
        ``Dataset`` that can deal with the data.
    y : target data, compatible with skorch.dataset.Dataset
        The same data types as for ``X`` are supported. If your X is a Dataset
        that contains the target, ``y`` may be set to None.

    Returns
    -------
    loss_value : float32 or np.ndarray
        Return type depends on net.criterion_.reduction, and will be a float if
        reduction is ``'sum'`` or ``'mean'``. If reduction is ``'none'`` then
        this function returns a ``np.ndarray`` object.

    """
    net.check_is_fitted()

    dataset = net.get_dataset(X, y)
    iterator = net.get_iterator(dataset, training=False)
    history = {"loss": [], "batch_size": []}
    reduction = net.criterion_.reduction
    for data in iterator:
        Xi, yi = unpack_data(data)
        yp = net.evaluation_step(Xi, training=False)
        loss = net.criterion_(yp, yi)
        if reduction == "none":
            loss_value = loss.detach().cpu().numpy()
        else:
            loss_value = loss.item()
        history["loss"].append(loss_value)
        history["batch_size"].append(yi.size(0))

    if reduction == "none":
        return np.concatenate(history["loss"], 0)
    loss_value = np.average(history["loss"], weights=history["batch_size"])
    if reduction == "sum":
        loss_value *= len(history["batch_size"])
    return loss_value
