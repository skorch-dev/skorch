""" Basic callback definition. """

import warnings

from sklearn.base import BaseEstimator
from skorch.exceptions import SkorchWarning


__all__ = ['Callback']


class Callback:
    """Base class for callbacks.

    All custom callbacks should inherit from this class. The subclass
    may override any of the ``on_...`` methods. It is, however, not
    necessary to override all of them, since it's okay if they don't
    have any effect.

    Classes that inherit from this also gain the ``get_params`` and
    ``set_params`` method.

    """
    def initialize(self):
        """(Re-)Set the initial state of the callback. Use this
        e.g. if the callback tracks some state that should be reset
        when the model is re-initialized.

        This method should return self.

        """
        return self

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        """Called at the beginning of training."""

    def on_train_end(self, net, X=None, y=None, **kwargs):
        """Called at the end of training."""

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the end of each epoch."""

    def on_batch_begin(self, net, batch=None, training=None, **kwargs):
        """Called at the beginning of each batch."""

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        """Called at the end of each batch."""

    def on_grad_computed(
            self, net, named_parameters, X=None, y=None, training=None, **kwargs):
        """Called once per batch after gradients have been computed but before
        an update step was performed.
        """

    def _get_param_names(self):
        return (key for key in self.__dict__ if not key.endswith('_'))

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)

    def set_params(self, **params):
        BaseEstimator.set_params(self, **params)


# TODO: remove after some deprecation period, e.g. skorch 0.12
def _on_batch_overridden(callback):
    """Check if on_batch_begin or on_batch_end were overridden

    If the method does not exist at all, it's not considered overridden. This is
    mostly for callbacks that are mocked.

    """
    try:
        base_skorch_cls = next(cls for cls in callback.__class__.__mro__
                               if cls.__module__.startswith('skorch'))
    except StopIteration:
        # does not derive from skorch callback, possibly a mock
        return False

    obb = base_skorch_cls.on_batch_begin
    obe = base_skorch_cls.on_batch_end
    return (
        getattr(callback.__class__, 'on_batch_begin', obb) is not obb
        or getattr(callback.__class__, 'on_batch_end', obe) is not obe
    )


# TODO: remove after some deprecation period, e.g. skorch 0.12
def _issue_warning_if_on_batch_override(callback_list):
    """Check callbacks for overridden on_batch method and issue warning

    We introduced a breaking change by changing the signature of on_batch_begin
    and on_batch_end. To help users, we try to detect if they use any custom
    callback that overrides on of these methods and issue a warning if they do.
    The warning states how to adjust the method signature and how it can be
    filtered.

    After some transition period, the checking and the warning should be
    removed again.

    Parameters
    ----------
    callback_list : list of (str, callback) tuples
      List of initialized callbacks.

    Warns
    -----
    Issues a ``SkorchWarning`` if any of the callbacks fits the conditions.

    """
    if not callback_list:
        return

    callbacks = [callback for _, callback in callback_list]

    # first detect if there are any user defined callbacks
    user_defined_callbacks = [
        callback for callback in callbacks
        if not callback.__module__.startswith('skorch')
    ]
    if not user_defined_callbacks:
        return

    # check if any of these callbacks overrides on_batch_begin or on_batch_end
    overriding_callbacks = [
        callback for callback in user_defined_callbacks
        if _on_batch_overridden(callback)
    ]

    if not overriding_callbacks:
        return

    warning_msg = (
        "You are using an callback that overrides on_batch_begin "
        "or on_batch_end. As of skorch 0.10, the signature was changed "
        "from 'on_batch_{begin,end}(self, X, y, ...)' to "
        "'on_batch_{begin,end}(self, batch, ...)'. To recover, change "
        "the signature accordingly and add 'X, y = batch' on the first "
        "line of the method body. To suppress this warning, add:\n"
        "'import warnings; from skorch.exceptions import SkorchWarning\n"
        "warnings.filterwarnings('ignore', category=SkorchWarning)'.")

    warnings.warn(warning_msg, SkorchWarning)
