import numpy as np
import torch
from accelerate import Accelerator
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from torch import nn
from torch.distributed import TCPStore

from skorch import NeuralNetClassifier
from skorch.hf import AccelerateMixin
from skorch.history import DistributedHistory


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(100, 2)
        self.nonlin = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.dense0(X)
        X = self.nonlin(X)
        return X


# make use of accelerate by creating a class with the AccelerateMixin
class AcceleratedNeuralNetClassifier(AccelerateMixin, NeuralNetClassifier):
    pass


def main():
    X, y = make_classification(10000, n_features=100, n_informative=50, random_state=0)
    X = X.astype(np.float32)

    accelerator = Accelerator()

    # use history class that works in distributed setting
    # see https://skorch.readthedocs.io/en/latest/user/history.html#distributed-history
    is_master = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.local_process_index
    store = TCPStore(
        "127.0.0.1", port=8080, world_size=world_size, is_master=is_master)
    dist_history = DistributedHistory(
        store=store, rank=rank, world_size=world_size)

    model = AcceleratedNeuralNetClassifier(
        MyModule,
        criterion=nn.CrossEntropyLoss,
        accelerator=accelerator,
        max_epochs=3,
        lr=0.001,
        history=dist_history,
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
