"""skorch base imports"""

import pkg_resources
from pkg_resources import parse_version

MIN_TORCH_VERSION = '0.4.0'

try:
    import torch
except:  # pylint: disable=bare-except
    raise Exception('skorch depends on PyTorch. Visit https://pytorch.org/ '
                    'for install instructions')

torch_version = pkg_resources.get_distribution('torch').version
if parse_version(torch_version) < parse_version(MIN_TORCH_VERSION):
    msg = ('skorch depends a newer version of PyTorch (at least {req}, not '
           '{installed}). Visit https://pytorch.org for install details')
    raise Exception(msg.format(req=MIN_TORCH_VERSION, installed=torch_version))


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
