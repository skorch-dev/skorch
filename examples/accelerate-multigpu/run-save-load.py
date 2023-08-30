"""Check that saving and loading works with accelerate.

Especially, pay attention that both the initial model, as well as the loaded
model, could be either wrapped with accelerate or not, i.e. there are 4 possible
combinations.

"""

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from torch import nn
from torch.distributed import TCPStore

from skorch import NeuralNetClassifier
from skorch.hf import AccelerateMixin
from skorch.history import DistributedHistory


PORT = 8080


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


def get_accelerate_model(accelerator):
    global PORT
    PORT += 1

    is_master = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.local_process_index
    store = TCPStore(
        "127.0.0.1", port=PORT, world_size=world_size, is_master=is_master)
    dist_history = DistributedHistory(
        store=store, rank=rank, world_size=world_size)

    return AcceleratedNeuralNetClassifier(
        MyModule,
        criterion=nn.CrossEntropyLoss,
        accelerator=accelerator,
        max_epochs=3,
        lr=0.001,
        history=dist_history,
    )


def get_vanilla_model():
    return NeuralNetClassifier(
        MyModule,
        criterion=nn.CrossEntropyLoss,
        max_epochs=3,
        lr=0.001,
    )


def main(wrap_initial_model=True, wrap_loaded_model=True):
    X, y = make_classification(10000, n_features=100, n_informative=50, random_state=0)
    X = X.astype(np.float32)

    accelerator = Accelerator()
    model = get_accelerate_model(accelerator)
    model.unwrap_after_train = True if wrap_initial_model else False
    model.fit(X, y)

    model.save_params(f_params="model_params.pt")
    y_pred = model.predict(X)
    accuracy_before = accuracy_score(y, y_pred)
    print(f"Accuracy before loading: {accuracy_before}")

    if wrap_loaded_model:
        model_loaded = get_accelerate_model(accelerator).initialize()
    else:
        model_loaded = get_vanilla_model().initialize()

    model_loaded.load_params(f_params="model_params.pt")
    y_pred = model_loaded.predict(X)
    accuracy_after = accuracy_score(y, y_pred)
    print(f"Accuracy  after loading: {accuracy_after}")

    assert accuracy_before == accuracy_after


if __name__ == '__main__':
    main(True, True)
    main(True, False)
    main(False, True)
    main(False, False)
