"""Simple training script for a MLP classifier.

See accompanying README.md for more details.

"""

import pickle

import fire
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from skorch import NeuralNetClassifier
import torch
from torch import nn

from skorch.helper import parse_args


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


N_FEATURES = 20  # number of input features
N_CLASSES = 2  # number of classes


class MLPClassifier(nn.Module):
    """A simple multi-layer perceptron module.

    This can be adapted for usage in different contexts, e.g. binary
    and multi-class classification, regression, etc.

    Note: This docstring is used to create the help for the CLI.

    Parameters
    ----------
    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    """
    def __init__(
            self,
            hidden_units=10,
            num_hidden=1,
            nonlin=nn.ReLU(),
            dropout=0,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_hidden = num_hidden
        self.nonlin = nonlin
        self.dropout = dropout

        self.reset_params()

    def reset_params(self):
        """(Re)set all parameters."""
        units = [N_FEATURES]
        units += [self.hidden_units] * self.num_hidden
        units += [N_CLASSES]

        sequence = []
        for u0, u1 in zip(units, units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))

        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        return nn.Softmax(dim=-1)(self.sequential(X))


def get_data(n_samples=100):
    """Get synthetic classification data with n_samples samples."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        random_state=0,
    )
    X = X.astype(np.float32)
    return X, y


def get_model(with_pipeline=False):
    """Get a multi-layer perceptron model.

    Optionally, put it in a pipeline that scales the data.

    """
    model = NeuralNetClassifier(MLPClassifier)
    if with_pipeline:
        model = Pipeline([
            ('scale', FeatureUnion([
                ('minmax', MinMaxScaler()),
                ('normalize', Normalizer()),
            ])),
            ('select', SelectKBest(k=N_FEATURES)),  # keep input size constant
            ('net', model),
        ])
    return model


def save_model(model, output_file):
    """Save model to output_file, if given"""
    if not output_file:
        return

    with open(output_file, 'wb') as f:
        pickle.dump(model, f)
    print("Saved model to file '{}'.".format(output_file))


def net(n_samples=100, output_file=None, **kwargs):
    """Train an MLP classifier on synthetic data.

    n_samples : int (default=100)
      Number of training samples

    output_file : str (default=None)
      If not None, file name used to save the model.

    """

    model = get_model(with_pipeline=False)
    # important: wrap the model with the parsed arguments
    parsed = parse_args(kwargs)
    model = parsed(model)

    X, y = get_data(n_samples=n_samples)
    print("Training MLP classifier")
    model.fit(X, y)

    save_model(model, output_file)


def pipeline(n_samples=100, output_file=None, **kwargs):
    """Train an MLP classifier in a pipeline on synthetic data.

    The pipeline scales the input data before passing it to the net.

    Note: This docstring is used to create the help for the CLI.

    Parameters
    ----------
    n_samples : int (default=100)
      Number of training samples

    output_file : str (default=None)
      If not None, file name used to save the model.

    """

    model = get_model(with_pipeline=True)
    # important: wrap the model with the parsed arguments
    parsed = parse_args(kwargs)
    model = parsed(model)

    X, y = get_data(n_samples=n_samples)
    print("Training MLP classifier in a pipeline")
    model.fit(X, y)

    save_model(model, output_file)


if __name__ == '__main__':
    # register 2 functions, "net" and "pipeline"
    fire.Fire({'net': net, 'pipeline': pipeline})
