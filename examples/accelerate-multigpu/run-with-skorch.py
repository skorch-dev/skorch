import numpy as np
import torch
from accelerate import Accelerator
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from torch import nn

from skorch import NeuralNetClassifier
from skorch.hf import AccelerateMixin


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(100, 2)
        self.nonlin = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.dense0(X)
        X = self.nonlin(X)
        return X

class AcceleratedNeuralNetClassifier(AccelerateMixin, NeuralNetClassifier):
    pass


class MyAccelerator(Accelerator):
    def __deepcopy__(self, memo):
        return self


def main():
    X, y = make_classification(10000, n_features=100, n_informative=50, random_state=0)
    X = X.astype(np.float32)

    accelerator = MyAccelerator()
    model = AcceleratedNeuralNetClassifier(
        MyModule,
        accelerator=accelerator,
        max_epochs=3,
        lr=0.001,
    )

    cross_validate(
        model,
        X,
        y,
        cv=2,
        scoring="average_precision",
        error_score="raise",
    )


if __name__ == '__main__':
    main()
