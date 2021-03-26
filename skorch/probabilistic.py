"""TODO

Move to experimental folder?

Assumptions being made:

- The criterion always takes likelihood and module as input arguments
- We optimize the negative criterion

"""

from itertools import chain
import pickle

import gpytorch
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
import torch
from torch.utils.data import DataLoader

from skorch.net import NeuralNet
from skorch.dataset import CVSplit
from skorch.dataset import unpack_data
from skorch.callbacks import EpochScoring
from skorch.callbacks import EpochTimer
from skorch.callbacks import PassthroughScoring
from skorch.callbacks import PrintLog
from skorch.setter import optimizer_setter
from skorch.utils import get_dim
from skorch.utils import is_dataset
from skorch.utils import to_device
from skorch.utils import to_numpy


# TODO: warn about experimental API

__all__ = ['GPRegressor', 'GPBinaryClassifier']


class GPBase(NeuralNet):
    """TODO"""
    prefixes_ = ['module', 'iterator_train', 'iterator_valid', 'optimizer',
                 'criterion', 'callbacks', 'dataset', 'likelihood']

    cuda_dependent_attributes_ = ['module_', 'optimizer_', 'criterion_', 'likelihood_']

    def __init__(
            self,
            module,
            *args,
            likelihood,
            criterion,
            train_split=False,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            train_split=train_split,
            **kwargs
        )
        self.likelihood = likelihood

    def initialize(self):
        """Initializes all components of the :class:`.NeuralNet` and
        returns self.

        TODO

        """
        self.initialize_likelihood()
        return super().initialize()

    def initialize_likelihood(self):
        """Initializes the likelihood."""
        kwargs = self.get_params_for('likelihood')
        likelihood = self.likelihood
        is_initialized = isinstance(likelihood, torch.nn.Module)

        if kwargs or not is_initialized:
            if is_initialized:
                likelihood = type(likelihood)

            if (is_initialized or self.initialized_) and self.verbose:
                msg = self._format_reinit_msg("likelihood", kwargs)
                print(msg)

            likelihood = likelihood(**kwargs)

        self.likelihood_ = likelihood
        if isinstance(self.likelihood_, torch.nn.Module):
            self.likelihood_ = to_device(self.likelihood_, self.device)
        return self

    def initialize_optimizer(self, triggered_directly=True):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly : bool (default=True)
          Only relevant when optimizer is re-initialized.
          Initialization of the optimizer can be triggered directly
          (e.g. when lr was changed) or indirectly (e.g. when the
          module was re-initialized). If and only if the former
          happens, the user should receive a message informing them
          about the parameters that caused the re-initialization.

        """
        named_parameters = chain(
            self.module_.named_parameters(), self.likelihood_.named_parameters())
        args, kwargs = self.get_params_for_optimizer(
            'optimizer', named_parameters)

        if self.initialized_ and self.verbose:
            msg = self._format_reinit_msg(
                "optimizer", kwargs, triggered_directly=triggered_directly)
            print(msg)

        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr

        self.optimizer_ = self.optimizer(*args, **kwargs)

        self._register_virtual_param(
            ['optimizer__param_groups__*__*', 'optimizer__*', 'lr'],
            optimizer_setter,
        )
        return self

    def initialize_criterion(self):
        """Initializes the criterion."""
        criterion_params = self.get_params_for('criterion')
        self.criterion_ = self.criterion(
            likelihood=self.likelihood_,
            model=self.module_,
            **criterion_params
        )
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_ = to_device(self.criterion_, self.device)
        return self

    def train_step_single(self, batch, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        Returns
        -------
        step : dict
          A dictionary ``{'loss': loss, 'y_pred': y_pred}``, where the
          float ``loss`` is the result of the loss function and
          ``y_pred`` the prediction generated by the PyTorch module.

        """
        step = super().train_step_single(batch, **fit_params)
        # TODO: add explanation
        step['y_pred'] = self.likelihood_(step['y_pred'])
        return step

    # pylint: disable=unused-argument
    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.

        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values

        y_true : torch tensor
          True target values.

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

        training : bool (default=False)
          Whether train mode should be used or not.

        """
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        if loss.dim() != 0:
            loss = loss.mean()
        return -loss

    def evaluation_step(self, batch, training=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.

        Therefore, the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.

        Parameters
        ----------
        batch
          A single batch returned by the data loader.

        training : bool (default=False)
          Whether to set the module to train mode or not.

        Returns
        -------
        y_infer
          The prediction generated by the module.

        """
        self.check_is_fitted()
        Xi, _ = unpack_data(batch)
        with torch.set_grad_enabled(training), gpytorch.settings.fast_pred_var():
            self.module_.train(training)
            y_infer = self.infer(Xi)
            if isinstance(y_infer, tuple):  # multiple outputs:
                return (self.likelihood_(y_infer[0]),) + y_infer[1:]
            return self.likelihood_(y_infer)

    def forward(self, X, training=False, device='cpu'):
        """Gather and concatenate the output from forward call with
        input data.

        The outputs from ``self.module_.forward`` are gathered on the
        compute device specified by ``device`` and then concatenated
        using PyTorch :func:`~torch.cat`. If multiple outputs are
        returned by ``self.module_.forward``, each one of them must be
        able to be concatenated this way.

        Parameters
        ----------
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

        training : bool (default=False)
          Whether to set the module to train mode or not.

        device : string (default='cpu')
          The device to store each inference result on.
          This defaults to CPU memory since there is genereally
          more memory available there. For performance reasons
          this might be changed to a specific CUDA device,
          e.g. 'cuda:0'.

        Returns
        -------
        y_infer : TODO

        """
        y_infer = list(self.forward_iter(X, training=training, device=device))
        return y_infer

    def predict_proba(self, X):
        raise NotImplementedError

    def sample(self, X, n_samples, axis=-1):
        """TODO"""
        samples = [p.rsample(torch.Size([n_samples])) for p in self.forward_iter(X)]
        return torch.cat(samples, axis=axis)

    def confidence_region(self, X, sigmas=2):
        """TODO"""
        nonlin = self._get_predict_nonlinearity()
        lower, upper = [], []
        for yi in self.forward_iter(X):
            posterior = yi[0] if isinstance(yi, tuple) else yi
            mean = posterior.mean
            std = posterior.stddev
            std = std.mul_(sigmas)
            lower.append(nonlin(mean.sub(std)))
            upper.append(nonlin(mean.add(std)))

        lower = torch.cat(lower)
        upper = torch.cat(upper)
        return lower, upper

    def __getstate__(self):
        try:
            return super().__getstate__()
        except pickle.PicklingError as exc:
            msg = "TODO"  # reference issue
            raise pickle.PicklingError(msg) from exc


class GPRegressor(GPBase, RegressorMixin):
    def __init__(
            self,
            module,
            *args,
            likelihood=gpytorch.likelihoods.GaussianLikelihood,
            criterion=gpytorch.mlls.ExactMarginalLogLikelihood,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            likelihood=likelihood,
            **kwargs
        )

    def predict(self, X, return_std=False, return_cov=False):
        """TODO

        Follows API of sklearn GaussianProcressRegressor. But should we??

        """
        if return_cov:
            raise NotImplementedError("TODO")

        nonlin = self._get_predict_nonlinearity()
        y_preds, y_stds = [], []
        for yi in self.forward_iter(X, training=False):
            posterior = yi[0] if isinstance(yi, tuple) else yi
            y_preds.append(to_numpy(nonlin(posterior.mean)))
            if not return_std:
                continue

            y_stds.append(to_numpy(nonlin(posterior.stddev)))

        y_pred = np.concatenate(y_preds, 0)
        if not return_std:
            return y_pred

        y_std = np.concatenate(y_stds, 0)
        return y_pred, y_std


class _GPClassifier(GPBase, ClassifierMixin):
    def __init__(
            self,
            module,
            *args,
            likelihood=gpytorch.likelihoods.SoftmaxLikelihood,
            criterion=gpytorch.mlls.VariationalELBO,
            train_split=CVSplit(5, stratified=True),
            classes=None,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            likelihood=likelihood,
            train_split=train_split,
            **kwargs
        )
        self.classes = classes

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            # add train accuracy because by default, there is no valid split
            ('train_acc', EpochScoring(
                'accuracy',
                name='train_acc',
                lower_is_better=False,
                on_train=True,
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]

    @property
    def classes_(self):
        if self.classes is not None:
            if not len(self.classes):
                raise AttributeError("{} has no attribute 'classes_'".format(
                    self.__class__.__name__))
            return self.classes
        return self.classes_inferred_

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            msg = ("No y-values are given (y=None). You must either supply a "
                   "Dataset as X or implement your own DataLoader for "
                   "training (and your validation) and supply it using the "
                   "``iterator_train`` and ``iterator_valid`` parameters "
                   "respectively.")
            raise ValueError(msg)
        if y is not None:
            # pylint: disable=attribute-defined-outside-init
            self.classes_inferred_ = np.unique(y)

    def predict_proba(self, X):
        """TODO"""
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yi in self.forward_iter(X, training=False):
            posterior = yi[0] if isinstance(yi, tuple) else yi
            y_probas.append(to_numpy(nonlin(posterior.mean)))

        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def predict(self, X):
        """TODO
        """
        return self.predict_proba(X).argmax(axis=1)


class GPBinaryClassifier(GPBase, ClassifierMixin):
    def __init__(
            self,
            module,
            *args,
            likelihood=gpytorch.likelihoods.BernoulliLikelihood,
            criterion=gpytorch.mlls.VariationalELBO,
            train_split=CVSplit(5, stratified=True),
            threshold=0.5,
            **kwargs
    ):
        super().__init__(
            module,
            *args,
            criterion=criterion,
            likelihood=likelihood,
            train_split=train_split,
            **kwargs
        )
        self.threshold = threshold

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('valid_loss', PassthroughScoring(
                name='valid_loss',
            )),
            # add train accuracy because by default, there is no valid split
            ('train_acc', EpochScoring(
                'accuracy',
                name='train_acc',
                lower_is_better=False,
                on_train=True,
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('print_log', PrintLog()),
        ]

    @property
    def classes_(self):
        return [0, 1]

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        super().check_data(X, y)
        if (not is_dataset(X)) and (get_dim(y) != 1):
            raise ValueError("The target data should be 1-dimensional.")

    def predict_proba(self, X):
        """TODO"""
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yi in self.forward_iter(X, training=False):
            posterior = yi[0] if isinstance(yi, tuple) else yi
            y_probas.append(to_numpy(nonlin(posterior.mean)))

        y_proba = np.concatenate(y_probas, 0).reshape(-1, 1)
        return np.hstack((1 - y_proba, y_proba))

    def predict(self, X):
        """TODO
        """
        y_proba = self.predict_proba(X)
        return (y_proba[:, 1] > self.threshold).astype('uint8')
