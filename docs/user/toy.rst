===
Toy
===

This module contains helper functions and classes that allow you to
prototype quickly or that can be used for writing tests.

MLPModule
---------

:class:`.MLPModule` is a simple PyTorch :class:`~torch.nn.Module` that
implements a multi-layer perceptron. It allows to indicate the number
of input, hidden, and output units, as well as the non-linearity and
use of dropout. You can use this module directly in conjunction with
:class:`.NeuralNet`.

Additionally, the functions :func:`~skorch.toy.make_classifier`,
:func:`~skorch.toy.make_binary_classifier`, and
:func:`~skorch.toy.make_regressor` can be used to return a
:class:`.MLPModule` with the defaults adjusted for use in multi-class
classification, binary classification, and regression, respectively.
