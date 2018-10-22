"""This module serves to elevate callbacks in submodules to the
skorch.callback namespace. Remember to define `__all__` in each
submodule.

"""

# pylint: disable=wildcard-import

from .base import *
from .logging import *
from .regularization import *
from .scoring import *
from .training import *
from .lr_scheduler import *

__all__ = ['Callback', 'EpochTimer', 'PrintLog', 'ProgressBar',
           'LRScheduler', 'WarmRestartLR', 'CyclicLR', 'GradientNormClipping',
           'BatchScoring', 'EpochScoring', 'Checkpoint', 'EarlyStopping',
           'Freezer', 'Unfreezer', 'Initializer', 'ParamMapper',
           'LoadInitState', 'TrainEndCheckpoint']
