"""Contains toy functions and classes for quick prototyping and
testing.

"""

from functools import partial

from torch import nn


class MLPModule(nn.Module):
    """A simple multi-layer perceptron module.

    This can be adapted for usage in different contexts, e.g. binary
    and multi-class classification, regression, etc.

    Parameters
    ----------
    input_units : int (default=20)
      Number of input units.

    output_units : int (default=2)
      Number of output units.

    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    output_nonlin : torch.nn.Module instance or None (default=None)
      Non-linearity to apply after last layer, if any.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    squeeze_output : bool (default=False)
      Whether to squeeze output. Squeezing can be helpful if you wish
      your output to be 1-dimensional (e.g. for
      NeuralNetBinaryClassifier).

    """
    def __init__(
            self,
            input_units=20,
            output_units=2,
            hidden_units=10,
            num_hidden=1,
            nonlin=nn.ReLU(),
            output_nonlin=None,
            dropout=0,
            squeeze_output=False,
    ):
        super().__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.num_hidden = num_hidden
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.dropout = dropout
        self.squeeze_output = squeeze_output

        self.reset_params()

    def reset_params(self):
        """(Re)set all parameters."""
        units = [self.input_units]
        units += [self.hidden_units] * self.num_hidden
        units += [self.output_units]

        sequence = []
        for u0, u1 in zip(units, units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))

        sequence = sequence[:-2]
        if self.output_nonlin:
            sequence.append(self.output_nonlin)

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):  # pylint: disable=arguments-differ
        X = self.sequential(X)
        if self.squeeze_output:
            X = X.squeeze(-1)
        return X


def make_classifier(output_nonlin=nn.Softmax(dim=-1), **kwargs):
    """Return a multi-layer perceptron to be used with
    NeuralNetClassifier.

    Parameters
    ----------
    input_units : int (default=20)
      Number of input units.

    output_units : int (default=2)
      Number of output units.

    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    """
    return partial(MLPModule, output_nonlin=output_nonlin, **kwargs)


def make_binary_classifier(squeeze_output=True, **kwargs):
    """Return a multi-layer perceptron to be used with
    NeuralNetBinaryClassifier.

    Parameters
    ----------
    input_units : int (default=20)
      Number of input units.

    output_units : int (default=2)
      Number of output units.

    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    """
    return partial(MLPModule, squeeze_output=squeeze_output, **kwargs)


def make_regressor(output_units=1, **kwargs):
    """Return a multi-layer perceptron to be used with
    NeuralNetRegressor.

    Parameters
    ----------
    input_units : int (default=20)
      Number of input units.

    output_units : int (default=1)
      Number of output units.

    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    """
    return partial(MLPModule, output_units=output_units, **kwargs)
