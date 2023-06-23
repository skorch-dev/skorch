"""Contains skorch-specific exceptions and warnings."""

from sklearn.exceptions import NotFittedError


class SkorchException(Exception):
    """Base skorch exception."""


class NotInitializedError(SkorchException, NotFittedError):
    """Module is not initialized, please call the ``.initialize``
    method or train the model by calling ``.fit(...)``.

    """


class SkorchAttributeError(SkorchException):
    """An attribute was set incorrectly on a skorch net."""


class SkorchWarning(UserWarning):
    """Base skorch warning."""


class DeviceWarning(SkorchWarning):
    """A problem with a device (e.g. CUDA) was detected."""


class SkorchTrainingImpossibleError(SkorchException):
    """The net cannot be used for training"""


class LowProbabilityError(SkorchException):
    """Error raised when the predictions of an LLM have low probability"""
