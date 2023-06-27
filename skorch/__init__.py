"""skorch base imports"""

import sys
import warnings

import importlib.metadata

from .history import History
from .net import NeuralNet
from .classifier import NeuralNetClassifier
from .classifier import NeuralNetBinaryClassifier
from .regressor import NeuralNetRegressor
from . import callbacks
from ._version import Version


MIN_TORCH_VERSION = '1.1.0'


try:
    # pylint: disable=wrong-import-position
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named 'torch', and skorch depends on PyTorch "
        "(aka 'torch'). "
        "Visit https://pytorch.org/ for installation instructions.")

torch_version = torch.__version__
if Version(torch_version) < Version(MIN_TORCH_VERSION):
    msg = ('skorch depends on a newer version of PyTorch (at least {req}, not '
           '{installed}). Visit https://pytorch.org for installation details')
    raise ImportWarning(msg.format(req=MIN_TORCH_VERSION, installed=torch_version))


__all__ = [
    'History',
    'NeuralNet',
    'NeuralNetClassifier',
    'NeuralNetBinaryClassifier',
    'NeuralNetRegressor',
    'callbacks',
]

try:
    __version__ = importlib.metadata.version("skorch")
except:  # pylint: disable=bare-except
    __version__ = 'n/a'
