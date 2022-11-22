"""TODO"""

from collections.abc import Mapping
from functools import partial

import numpy as np
from sklearn.exceptions import NotFittedError
import torch

from skorch.callbacks import Callback
from skorch.exceptions import NotInitializedError
from skorch.utils import to_numpy
from skorch.utils import check_is_fitted
from sklearn.utils.validation import check_is_fitted as sk_check_is_fitted


def named_modules(net):
    """For a skorch net, yield all module names and modules

    Ignore modules that are not ``torch.nn.Module`` instances.

    """
    # pylint: disable=protected-access
    for module_name in net._modules + net._criteria:
        module = getattr(net, module_name + '_')
        if isinstance(module, torch.nn.Module):
            yield module_name, module


# pylint: disable=unused-argument,redefined-builtin
def _add_activation_hook(model, input, output, *, activations, module_name, layer_name):
    """Helper function for adding activation hooks"""
    if module_name not in activations:
        activations[module_name] = {}
    val = to_numpy(output)

    # disambiguate activations when the output is not a simple array
    if isinstance(val, np.ndarray):
        activations[module_name][layer_name] = val
    elif isinstance(val, (list, tuple)):
        for i, v in enumerate(val):
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Activations of type {type(v)} are not supported")
            activations[module_name][layer_name + f'[{i}]'] = v
    elif isinstance(val, Mapping):
        for k, v in val.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Activations of type {type(v)} are not supported")
            activations[module_name][layer_name + f'["{k}"]'] = v
    else:
        raise TypeError(f"Activations of type {type(v)} are not supported")


def add_activation_hooks(net):
    """Add forward hooks to all layers to record activations

    Ignore the top level modules like ``net.module_`` and ``net.criterion_``.

    If an output is not a simple array, it is disambiguated. E.g. if it's a
    list, the name get a suffix of ``[i]`` where ``i`` designates the index in
    the list. Similary, when the output is a dict, a ``[key]`` suffix is added,
    where ``[key]`` is the key of the corresponding value in the dictionary.

    Parameters
    ----------
    net : skorch.NeuralNet
      The neural net instance (has to be initialized).

    Returns
    -------
    activations: dict
      The dict that will be used to store the activations for each module.

    handles: list of torch.utils.hooks.RemovableHandle
      These handles are returned by torch when adding hooks. They can be used to
      remove the hooks.

    Raises
    ------
    TypeError
      When the output of a layer is not a simple array and if it cannot be
      disambiguated, a ``TypeError`` is raised. Disambiguation works for lists,
      tuples, and dicts (but not nested ones).

    """
    activations = {}
    handles = []

    for module_name, module in named_modules(net):
        for layer_name, submodule in module.named_modules():
            if submodule is module:
                # is logging activations for whole module useful?
                continue

            handle = submodule.register_forward_hook(partial(
                _add_activation_hook,
                activations=activations,
                module_name=module_name,
                layer_name=layer_name,
            ))
            handles.append(handle)

    return activations, handles


def _add_grad_hook(grad, *, gradients, module_name, param_name):
    """Helper function for adding gradient hooks"""
    if module_name not in gradients:
        gradients[module_name] = {}
    gradients[module_name][param_name] = to_numpy(grad)


def add_grad_hooks(net):
    """Add backward hooks to all parameters to record gradients

    Parameters
    ----------
    net : skorch.NeuralNet
      The neural net instance (has to be initialized).

    Returns
    -------
    gradients: dict
      The dict that will be used to store the gradients for each module.

    handles: list of torch.utils.hooks.RemovableHandle
      These handles are returned by torch when adding hooks. They can be used to
      remove the hooks.

    """
    gradients = {}
    handles = []
    for module_name, module in named_modules(net):
        for param_name, tensor in module.named_parameters():
            if not tensor.requires_grad:
                continue

            handle = tensor.register_hook(partial(
                _add_grad_hook,
                gradients=gradients,
                module_name=module_name,
                param_name=param_name,
            ))
            handles.append(handle)

    return gradients, handles


class LogActivationsGradients(Callback):
    """Helper callback that stores useful metrics

    Not to be used directly. This is only intended to be used internally by
    ``SkorchDoctor``.

    """
    def __init__(
        self,
        activation_logs,
        gradient_logs,
        param_update_logs,
        activations,
        gradients
    ):
        self.activation_logs = activation_logs
        self.gradient_logs = gradient_logs
        self.param_update_logs = param_update_logs
        self.activations = activations
        self.gradients = gradients

    # pylint: disable=arguments-differ
    def on_epoch_begin(self, net, **kwargs):
        for module_name, _ in named_modules(net):
            self.activation_logs[module_name].append([])
            self.gradient_logs[module_name].append([])
            self.param_update_logs[module_name].append([])
        return self

    def on_batch_end(self, net, batch=None, training=False, **kwargs):
        if not training:
            return self

        eps = 1e-9  # prevent divide by 0
        for module_name, module in named_modules(net):
            param_updates = {}
            for key, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                grad = self.gradients[module_name][key]
                param_updates[key] = (grad.std() / (eps + param.std())).item()

            self.param_update_logs[module_name][-1].append(param_updates)
            if module_name in self.activations:  # not all modules record activations
                self.activation_logs[module_name][-1].append(
                    self.activations[module_name].copy()
                )
                self.activations[module_name].clear()

            if module_name in self.gradients:  # not all modules record gradients
                self.gradient_logs[module_name][-1].append(
                    self.gradients[module_name].copy()
                )
                self.gradients[module_name].clear()
        return self


class SkorchDoctor:
    """SkorchDoctor helps you better understand and debug neural net training.

    By providing records activations, gradienst, and parameter updates, as well
    as useful plotting functions, this class can assist you in understanding
    your training process and how to improve it.

    This is heavily inspred by the tips from Andrej Karpathy on how to better
    understand the neural net and diagnose potential errors.

        - https://karpathy.github.io/2019/04/25/recipe/
        - https://www.youtube.com/watch?v=P6sfmUTpUmc

    To use this class, initialize your skorch net and load your data as you
    would typically do when training a net. Then, initializea ``SkorchDoctor``
    instance by passing said net, and fit the doctor with a small amount of
    data. Finally, use the records or plotting functions to help you better
    understand the training process.

    What exactly you do with this information is up to you. Some examples that
    come to mind:

        - Use the ``plot_loss`` figure to see if your model is powerful enough
          to completely overfit a small sample. If not, consider increasing its
          capacity, e.g. by stacking more layers or using more units per layer.

        - Use the distribution of activations to see if some layers produce
          extreme values and may need a different non-linearity or some form of
          normalization, e.g. batch norm or layer norm. A different weight
          initialization scheme could also help.

        - Check the relative magnitude of the parameter updates to check if your
          learning rate is too low or too high. Maybe some layers should be
          frozen, or you might want to have different learning rates for
          different parameter groups, or an adaptive optimizer like Adam.

        - If gradients are too big, consider using gradient clipping. If the
          mangitude of gradients shifts over time, you might want to use a
          learning rate scheduler.

    At the end of the day, ``SkorchDoctor`` will not tell you what you need to
    do to improve the training process, but it will greatly facilitate to
    diagnose potential problems.

    Examples
    --------
    >>> net = NeuralNet(..., max_epochs=5)  # a couple of epochs are enough
    >>> from skorch.helper import SkorchDoctor
    >>> doctor = SkorchDoctor(net)
    >>> X_sample, y_sample = X[:100], y[:100]  # a few samples are enough
    >>> doctor.fit(X_sample, y_sample)
    >>> # now use the attributes and plotting functions to better
    >>> # understand the training process
    >>> doctor.activation_logs_  # the recorded activations
    >>> doctor.gradient_logs_  # the recorded gradients
    >>> # the next steps require matplotlib to be installed
    >>> doctor.plot_loss()
    >>> doctor.plot_activations()
    >>> doctor.plot_gradients()
    >>> doctor.plot_param_updates()
    >>> doctor.plot_activations_over_time(<layer-name>)
    >>> doctor.plot_gradients_over_time(<param-name>)

    Notes
    -----
    Even if a train/valid split is used for the net, only training data is
    logged.

    Since ``SkorchDoctor`` will record a lot of values, you should expect an
    increase in memory usage and training time. However, it's sufficient to
    train with a handful of samples and only a few epochs, which helps
    offsetting those disadvantages.

    After you finished the analysis, it is recommended to re-initialize the net
    or even better start a new process. This is because the net you passed is
    modified by adding hooks, callbacks, and training it. Although there is a
    clean up step at the end, it's better to start fresh when starting the real
    model training.

    Parameters
    ----------
    net : skorch.NeuralNet
      The skorch net to be diagnosed.

    Attributes
    ----------
    num_steps_ : int
      The total number of training steps. E.g. if trained for 10 epochs with 5
      batches each, that would be 50.

    module_names_ : list of str
      All modules used by the net, typically those are ``"module"`` and
      ``"criterion"``.

    activation_logs_: dict of list of list of dict of np.ndarray
      The activations of each layer for each module. The outer dict contains one
      entry for each top level module. The values are lists of lists, one entry
      for each epoch and each batch, respectively. The entries of the final
      lists are dictionaries, with keys corresponding to the layer name and
      values to the activations of that layer, stored as a numpy array.

      This data structure seems to be a bit complicated at first but its use is
      quite straightforward. E.g. to get the activations of the layer called
      "dense0" of the "module" in epoch 0 and batch 0, use
      ``doctor.activation_logs_['module'][0][0]['dense0']``.

      If an activation is not a simple array, it is disambiguated. E.g. if it's a
      list, the name get a suffix of ``[i]`` where ``i`` designates the index in
      the list. Similary, when the output is a dict, a ``[key]`` suffix is
      added, where ``[key]`` is the key of the corresponding value in the
      dictionary.

    gradient_logs_: dict of list of list of dict of np.ndarray
      The gradients of each parameter for each module. The outer dict contains
      one entry for each top level module. The values are lists of lists, one
      entry for each epoch and each batch, respectively. The entries of the
      final lists are dictionaries, with keys corresponding to the parameter
      name and values to the gradients of that parameter, stored as a numpy
      array.

      This data structure seems to be a bit complicated at first but its use is
      quite straightforward. E.g. to get the gradient of the parameter called
      "dense0.weight" of the "module" in epoch 5 and batch 3, use
      ``doctor.gradient_logs_['module'][5][3]['dense0.weight]``.

    param_update_logs_: dict of list of list of dict of float
      The relative parameter update of each parameter for each module. The outer
      dict contains one entry for each top level module. The values are lists of
      lists, one entry for each epoch and each batch, respectively. The entries
      of the final lists are dictionaries, with keys corresponding to the
      parameter name and values to the standard deviation of the update of that
      parameter, relative to the standard deviation of that parameter, stored as
      a float.

      This data structure seems to be a bit complicated at first but its use is
      quite straightforward. E.g. to get the update of the parameter called
      "dense0.weight" of the "module" in the last epoch and last batch, use
      ``doctor.paramter_udpate_logs_['module'][-1][-1]['dense0.weight]``.

    fitted_ : bool
      Whether the instance has been fitted.

    """
    def __init__(self, net):
        self.net = net

    def _add_callback(self, net, name, callback):
        if net.callbacks is None:
            net.callbacks = [(name, callback)]
        else:
            net.callbacks.append((name, callback))
        net.initialize_callbacks()

    def initialize(self):
        """Initialize the SkorchDoctor

        This method typically does not need to be invoked explicitly, because it
        is called by ``fit``.

        """
        if not getattr(self.net, 'initialized_', False):
            self.net.initialize()
        self.fitted_ = False

        module_names = []
        self.module_names_ = [name for name, _ in named_modules(self.net)]

        self.handles_ = []

        module_names = [name for name, _ in named_modules(self.net)]
        self.activation_logs_ = {module_name: [] for module_name in module_names}
        self.gradient_logs_ = {module_name: [] for module_name in module_names}
        self.param_update_logs_ = {module_name: [] for module_name in module_names}

        activations, ahandles = add_activation_hooks(self.net)
        self.activations_ = activations
        self.handles_.extend(ahandles)

        gradients, ghandles = add_grad_hooks(self.net)
        self.gradients_ = gradients
        self.handles_.extend(ghandles)

        cb = LogActivationsGradients(
            activation_logs=self.activation_logs_,
            gradient_logs=self.gradient_logs_,
            param_update_logs=self.param_update_logs_,
            activations=self.activations_,
            gradients=self.gradients_,
        )
        self._add_callback(self.net, 'log_activations_gradients', cb)
        return self

    def _remove_callback(self, net, callback_name):
        """Helper function to remove callbacks"""
        indices = []
        for i, callback in enumerate(net.callbacks):
            if not isinstance(callback, tuple):
                continue
            name, _ = callback
            if name == callback_name:
                indices.append(i)

        for i in indices[::-1]:
            del net.callbacks[i]

        indices = []
        for i, (name, _) in enumerate(net.callbacks_):
            if name == callback_name:
                indices.append(i)

        for i in indices[::-1]:
            del net.callbacks_[i]

    def _clean_up(self):
        """Clean up method to be called after training"""
        for handle in self.handles_:
            handle.remove()

        self._remove_callback(self.net, 'log_activations_gradients')

    def flatten(self, items):
        """Flatten values in nested lists or tuples"""
        for item in items:
            if isinstance(item, (list, tuple)):
                yield from self.flatten(item)
            else:
                yield item

    def check_is_fitted(self):
        try:
            sk_check_is_fitted(self, ['fitted_'])
        except NotFittedError as exc:
            msg = (
                f"{self.__class__.__name__} is not initialized yet. Call "
                "'fit(X, y) before using this method."
            )
            raise NotInitializedError(msg) from exc

    def fit(self, X, y=None, **fit_params):
        """Initialize and fit the SkorchDoctor

        It is advised to use a low number of epochs and a small amount of data
        only, since the collection of data results in time and memory overhead.

        """
        self.initialize()
        try:
            self.net.partial_fit(X, y, **fit_params)
        finally:
            self._clean_up()

        self.num_steps_ = sum(
            1 for _ in self.flatten(list(self.activation_logs_.values())[0])
        )
        self.fitted_ = True
        return self

    def get_layer_names(self):
        """Return the names of all layers/modules

        Returns
        -------
        names : dict of list of str
          For each top level module, all layer names as a list of strings.

        """
        self.check_is_fitted()
        names = {}
        for module in self.module_names_:
            if self.activation_logs_[module][0]:
                names[module] = list(self.activation_logs_[module][0][0].keys())
            else:
                names[module] = []
        return names

    def get_param_names(self):
        """Return all learnable parameters of the net

        Returns
        -------
        names : dict of list of str
          For each top level module, all parameter names as a list of strings.

        """
        self.check_is_fitted()
        names = {}
        for module in self.module_names_:
            if self.gradient_logs_[module][0]:
                # using the reversed order because gradients are recorded from
                # last to first, but first to last is more intuitive to show
                # TODO: When dropping python 3.7, dict keys are reversible, so
                # no need to call list(keys)
                keys = list(self.gradient_logs_[module][0][0].keys())
                names[module] = list(reversed(keys))
            else:
                names[module] = []
        return names

    def predict(self, X, **kwargs):
        """Calls the ``predict`` method of the underlying net"""
        return self.net.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """Calls the ``predict_proba`` method of the underlying net"""
        return self.net.predict_proba(X, **kwargs)

    def score(self, X, y=None, **kwargs):
        """Calls the ``score`` method of the underlying net"""
        return self.net.score(X, y=y, **kwargs)

    def _get_axes(self, axes=None, figsize=None, nrows=1, squeeze=False):
        """Helper function to get empty axes for plotting"""
        if axes is not None:
            return axes

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            msg = (
                "This feature requires matplotlib to be installed; "
                "please install it first, e.g. using "
                "'python -m pip install matplotlib'"
            )
            raise ImportError(msg) from exc

        width = 12
        height = 6

        if figsize is None:
            figsize = (width, nrows * height)

        _, axes = plt.subplots(nrows, 1, figsize=figsize, squeeze=squeeze)
        return axes

    def plot_loss(self, ax=None, figsize=None, **kwargs):
        """Plot the loss over each epoch.

        Plots the training loss and, if present, the validation loss over time.

        """
        self.check_is_fitted()

        ax = self._get_axes(ax, figsize=figsize, nrows=1, squeeze=True)
        history = self.net.history
        xvec = np.arange(len(history)) + 1
        ax.plot(xvec, history[:, 'train_loss'], label='train', **kwargs)

        if 'valid_loss' in history[0]:
            ax.plot(xvec, history[:, 'valid_loss'], label='valid', **kwargs)

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        ax.set_title('loss over time')
        return ax

    def plot_activations(
            self,
            step=-1,
            match_fn=None,
            axes=None,
            histtype='step',
            lw=2,
            bins=None,
            density=True,
            figsize=None,
            **kwargs
    ):
        """Plot the distribution of activations produced by the layers

        Parameters
        ----------
        step : int (default=-1)
          Which training step to plot. By default, the last step (-1) is chosen.

        match_fn : callable or None (default=None)
          If not ``None``, this should be a callable/function that takes the
          name of a layer as input and returns a bool as output, where ``False``
          indicates that this layer should be excluded. Use this to filter only
          specific layers you want to plot.

        axes : np.ndarray of AxesSubplot or None (default=None)
          By default, a new matplotlib plot is created. If you instead want to
          plot onto an existing plot, pass it here. There should be one subplot
          for each top level module (typically 2).

        bins : np.ndarray or None (default=None)
          Bins to use for the histogram. If left as ``None``, they are inferred
          from the data.

        **kwargs
          You can override remaining plotting arguments like ``lw`` (line width)
          or ``figsize`` (figure size).

        Returns
        -------
        axes : np.andarray of AxesSubplot
          The axes of the plot.

        """
        self.check_is_fitted()

        # only use modules for which the values are not simply empty lists
        module_names = [
            key for key, val in self.activation_logs_.items() if any(l for l in val)
        ]

        axes = self._get_axes(axes, figsize=figsize, nrows=len(module_names))

        for module_name, ax in zip(module_names, axes):
            ax = ax[0]  # only 1 col
            activations = list(self.flatten(self.activation_logs_[module_name]))[step]
            if match_fn:
                activations = {k: v for k, v in activations.items() if match_fn(k)}
                if not activations:
                    msg = (
                        "No layer found matching the specification of match_fn. "
                        "Use doctor.get_layer_names() to check all layers."
                    )
                    raise ValueError(msg)

            if bins is None:
                bin_min = min(a.min() for a in activations.values())
                bin_max = max(a.max() for a in activations.values())
                bins = np.linspace(bin_min, bin_max, 50)

            for key, val in activations.items():
                if val.ndim:
                    ax.hist(
                        val.flatten(),
                        label=key,
                        histtype=histtype,
                        lw=lw,
                        bins=bins,
                        density=density,
                        **kwargs
                    )
            ax.legend()
            ax.set_title(f"distribution of activations of {module_name}")
        return axes

    def plot_gradients(
            self,
            step=-1,
            match_fn=None,
            axes=None,
            histtype='step',
            lw=2,
            bins=None,
            density=True,
            figsize=None,
            **kwargs
    ):
        """Plot the distribution of gradients of each learnable parameter

        Parameters
        ----------
        step : int (default=-1)
          Which training step to plot. By default, the last step (-1) is chosen.

        match_fn : callable or None (default=None)
          If not ``None``, this should be a callable/function that takes the
          name of a parameter as input and returns a bool as output, where
          ``False`` indicates that this layer should be excluded. Use this to
          filter only specific parameters you want to plot.

        axes : np.ndarray of AxesSubplot or None (default=None)
          By default, a new matplotlib plot is created. If you instead want to
          plot onto an existing plot, pass it here. There should be one subplot
          for each top level module (typically 2).

        bins : np.ndarray or None (default=None)
          Bins to use for the histogram. If left as ``None``, they are inferred
          from the data.

        **kwargs
          You can override remaining plotting arguments like ``lw`` (line width)
          or ``figsize`` (figure size).

        Returns
        -------
        axes : np.andarray of AxesSubplot
          The axes of the plot.

        """
        self.check_is_fitted()

        # only use modules for which the values are not simply empty
        module_names = [
            key for key, val in self.gradient_logs_.items()
            if any(d for l in val for d in l)
        ]

        axes = self._get_axes(axes, figsize=figsize, nrows=len(module_names))

        for module_name, ax in zip(module_names, axes):
            ax = ax[0]  # only 1 col
            gradients = list(self.flatten(self.gradient_logs_[module_name]))[step]
            if match_fn:
                gradients = {k: v for k, v in gradients.items() if match_fn(k)}
                if not gradients:
                    msg = (
                        "No parameter found matching the specification of match_fn. "
                        "Use doctor.get_param_names() to check all parameters."
                    )
                    raise ValueError(msg)

            if bins is None:
                bin_min = min(g.min() for g in gradients.values())
                bin_max = max(g.max() for g in gradients.values())
                bins = np.linspace(bin_min, bin_max, 50)

            for key, val in gradients.items():
                if val.ndim:  # don't show scalars
                    ax.hist(
                        val.flatten(),
                        label=key,
                        histtype=histtype,
                        lw=lw,
                        bins=bins,
                        density=density,
                        **kwargs
                    )

            ax.legend()
            ax.set_title(f"distribution of gradients for {module_name}")

        return axes

    def plot_param_updates(self, match_fn=None, axes=None, figsize=None, **kwargs):
        """Plot the distribution of relative parameter updates.

        Plots the log10 of the standard deviation of the parameter update
        relative to the parameter itself, over time. Higher values mean that the
        parameter changes quite a lot with each training step, lower values mean
        that the parameter changes little.

        Parameters
        ----------
        match_fn : callable or None (default=None)
          If not ``None``, this should be a callable/function that takes the
          name of a parameter as input and returns a bool as output, where
          ``False`` indicates that this layer should be excluded. Use this to
          filter only specific parameters you want to plot.

        axes : np.ndarray of AxesSubplot or None (default=None)
          By default, a new matplotlib plot is created. If you instead want to
          plot onto an existing plot, pass it here. There should be one subplot
          for each top level module (typically 2).

        bins : np.ndarray or None (default=None)
          Bins to use for the histogram. If left as ``None``, they are inferred
          from the data.

        **kwargs
          You can override remaining plotting arguments like ``figsize`` (figure
          size).

        Returns
        -------
        axes : np.andarray of AxesSubplot
          The axes of the plot.

        """
        self.check_is_fitted()

        # only use modules for which the values are not simply empty
        module_names = [
            key for key, val in self.gradient_logs_.items()
            if any(d for l in val for d in l)
        ]

        axes = self._get_axes(axes, figsize=figsize, nrows=len(module_names))
        eps = 1e-9  # prevent log10 of 0

        for module_name, ax in zip(module_names, axes):
            ax = ax[0]  # only 1 col
            param_updates = list(self.flatten(self.param_update_logs_[module_name]))
            keys = param_updates[0].keys()
            if match_fn:
                keys = [key for key in keys if match_fn(key)]
                if not keys:
                    msg = (
                        "No parameter found matching the specification of match_fn. "
                        "Use doctor.get_param_names() to check all parameters."
                    )
                    raise ValueError(msg)

            for key in keys:
                values = [np.log10(update[key] + eps) for update in param_updates]
                xvec = np.arange(1, len(values) + 1)
                ax.plot(xvec, values, label=key, **kwargs)

            ax.set_xlabel("step")
            ax.set_ylabel(
                f"log10 of stdev of relative parameter updates for {module_name}"
            )
            ax.set_title(module_name)
            ax.legend()

        return axes

    def plot_activations_over_time(
            self,
            layer_name,
            module_name='module',
            ax=None,
            lw=2,
            bins=None,
            figsize=None,
            color='k',
            **kwargs
    ):
        """Plot the distribution of the activation of a specific layers over
        time

        The histograms are plotted "on top of each other" with an offset.
        Therefore, the absolute magnitude on the y-axis has no meaning.

        Parameters
        ----------
        layer_name : str
          The name of the specific layer whose activations should be plotted.

        module_name : str (default='module')
          The name of the module that the layer belongs to. By default, it is
          called "module" in skorch, but it's possible to define custom module
          names, in which case the corresponding name should be chosen.

        ax : AxesSubplot or None (default=None)
          By default, a new matplotlib plot is created. If you instead want to
          plot onto an existing plot, pass it here. Only a single plot is
          created.

        bins : np.ndarray or None (default=None)
          Bins to use for the histogram. If left as ``None``, they are inferred
          from the data.

        **kwargs
          You can override remaining plotting arguments like ``figsize`` (figure
          size).

        Returns
        -------
        ax : AxesSubplot
          The ax of the plot.

        """
        self.check_is_fitted()

        ax = self._get_axes(ax, figsize=figsize, squeeze=True)

        try:
            activations = [
                act[layer_name] for act in self.flatten(self.activation_logs_[module_name])
            ]
        except KeyError as exc:
            msg = (
                f"No layer named '{layer_name}' could be found. "
                "Use doctor.get_layer_names() to check all layers."
            )
            raise ValueError(msg) from exc

        n = len(activations)

        if bins is None:
            bin_min = min(a.min() for a in activations)
            bin_max = max(a.max() for a in activations)
            bins = np.linspace(bin_min, bin_max, 100)

        y_max = -1
        yvals = []
        for activation in activations:
            yval, _ = np.histogram(activation.flatten(), bins=bins)
            y_max = max(y_max, yval.max())
            yvals.append(yval)

        y_scale = y_max / n / 2
        alpha = max(0.1, 1 / n)

        for step, yval in enumerate(reversed(yvals)):
            bottom = y_scale * (n - step - 1)
            ax.fill_between(
                bins[1:],
                yval + bottom,
                y2=bottom,
                alpha=alpha,
                color=color,
                lw=lw,
                **kwargs
            )

        ax.set_yticklabels([])
        ax.set_ylabel('step / activation')
        ax.set_title(f"distribution of activations for {layer_name} of {module_name}")
        return ax

    def plot_gradient_over_time(
            self,
            param_name,
            module_name='module',
            ax=None,
            lw=2,
            bins=None,
            figsize=None,
            color='k',
            **kwargs
    ):
        """Plot the distribution of the gradients of a specific parameter over
        time

        The histograms are plotted "on top of each other" with an offset.
        Therefore, the absolute magnitude on the y-axis has no meaning.

        Parameters
        ----------
        param_name : str
          The name of the specific parameter that should be plotted.

        module_name : str (default='module')
          The name of the module that the paramter belongs to. By default, it is
          called "module" in skorch, but it's possible to define custom module
          names, in which case the corresponding name should be chosen.

        ax : AxesSubplot or None (default=None)
          By default, a new matplotlib plot is created. If you instead want to
          plot onto an existing plot, pass it here. Only a single plot is
          created.

        bins : np.ndarray or None (default=None)
          Bins to use for the histogram. If left as ``None``, they are inferred
          from the data.

        **kwargs
          You can override remaining plotting arguments like ``figsize`` (figure
          size).

        Returns
        -------
        ax : AxesSubplot
          The ax of the plot.

        """
        self.check_is_fitted()

        ax = self._get_axes(ax, figsize=figsize, squeeze=True)

        try:
            gradients = [
                grad[param_name] for grad in self.flatten(self.gradient_logs_[module_name])
            ]
        except KeyError as exc:
            msg = (
                f"No parameter named '{param_name}' could be found. "
                "Use doctor.get_param_names() to check all parameters."
            )
            raise ValueError(msg) from exc


        n = len(gradients)

        if bins is None:
            bin_min = min(g.min() for g in gradients)
            bin_max = max(g.max() for g in gradients)
            bins = np.linspace(bin_min, bin_max, 100)

        y_max = -1
        yvals = []
        for gradient in gradients:
            yval, _ = np.histogram(gradient.flatten(), bins=bins)
            y_max = max(y_max, yval.max())
            yvals.append(yval)

        y_scale = y_max / n / 2
        alpha = max(0.1, 1 / n)

        for step, yval in enumerate(reversed(yvals)):
            bottom = y_scale * (n - step - 1)
            ax.fill_between(
                bins[1:],
                yval + bottom,
                y2=bottom,
                alpha=alpha,
                color=color,
                lw=lw,
                **kwargs)

        ax.set_yticklabels([])
        ax.set_ylabel('step / gradient')
        ax.set_title(f"distribution of gradients for {param_name} of {module_name}")
        return ax
