import numpy as np
from skorch.net import NeuralNet
from skorch.dataset import unpack_data


class _CriterionAccumulator:
    def __init__(self, criterion):
        """
        _CriterionAccumulator(criterion)

        Parameters
        ----------
        criterion: PyTorch loss function
            e.g., nn.MSELoss or nn.NLLLoss
        """
        self.criterion = criterion
        reduction = self._get_reduction()
        assert reduction in ["mean", "sum", "none",], (
            "Expected reduction to be one of 'mean', 'sum' "
            + "or 'none' but got {reduction}.".format(reduction=reduction)
        )
        self.log = {"loss": [], "batch_size": []}
        self._reduce = {
            "sum": self._reduce_sum,
            "mean": self._reduce_mean,
            "none": self._reduce_none,
        }

    def __call__(self, input, target):
        loss = self.criterion(input, target)

        batch_size = target.size(0)
        self.log["batch_size"].append(batch_size)

        # If criterion.reduction is 'none', loss.item() can fail,
        # because item method requires scalar input.
        if self._get_reduction() == "none":
            loss = loss.detach().cpu().numpy()
            self.log["loss"].append(loss)
        else:
            self.log["loss"].append(loss.item())

    def reduce(self):
        reduction = self._get_reduction()
        return self._reduce[reduction]()

    def _get_reduction(self):
        return self.criterion.reduction

    def _reduce_sum(self):
        return np.sum(self.log["loss"])

    def _reduce_mean(self):
        losses = np.array(self.log["loss"])
        batch_sizes = np.array(self.log["batch_size"])
        n_examples = batch_sizes.sum()
        loss_value = np.sum(losses * batch_sizes) / n_examples
        return loss_value

    def _reduce_none(self):
        return np.concatenate(self.log["loss"], 0)


def loss_scoring(net: NeuralNet, X, y=None):
    """
    loss_scoring(net, X, y=None)

    Parameters
    ----------
    net : skorch.NeuralNet
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
    loss_value : float32???? or np.ndarray
        Return type depends on net.criterion_.reduction, and will be a float if
        reduction is ``'sum'`` or ``'mean'``. If reduction is ``'none'`` then
        this function returns a ``np.ndarray`` object.
    """
    net.check_is_fitted()

    criterion = _CriterionAccumulator(net.criterion_)
    dataset = net.get_dataset(X, y)
    iterator = net.get_iterator(dataset, training=False)

    for data in iterator:
        Xi, yi = unpack_data(data)
        yp = net.evaluation_step(Xi, training=False)
        criterion(yp, yi)
    loss_value = criterion.reduce()
    return loss_value
