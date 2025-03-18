# Optimize Multi-Layer Perceptrons using Optuna and Skorch

## Description

This example is adapted from the optuna-integration [repo](https://github.com/optuna/optuna-examples/tree/main/pytorch).In this example, we optimize the validation accuracy of hand-written digit recognition using
skorch and MNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole MNIST dataset, we here use a small subset of it.

# Installation

To install all the dependencies for the example, run:

```bash
pip install -r requirements.txt
```

You can run this example as follows, pruning can be turned on with the `--pruning`
argument.
```bash
    $ python skorch_example.py [--pruning]
```

To check all the available commands that are available, run:
```bash
    $ python skorch_example.py --help
```