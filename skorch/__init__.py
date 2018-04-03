"""skorch base imports"""

import pkg_resources

from .history import History
from .net import NeuralNet
from .net import NeuralNetClassifier
from .net import NeuralNetRegressor

from . import callbacks


__all__ = [
    'History',
    'NeuralNet',
    'NeuralNetClassifier',
    'NeuralNetRegressor',
    'callbacks',
]


try:
    __version__ = pkg_resources.get_distribution('skorch').version
except:  # pylint: disable=bare-except
    __version__ = 'n/a'
