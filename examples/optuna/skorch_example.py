import argparse
import urllib

import numpy as np
import optuna
from optuna.integration import SkorchPruningCallback
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


SUBSET_RATIO = 0.4

mnist = fetch_openml("mnist_784", cache=False)

X = pd.DataFrame(mnist.data)
y = mnist.target.astype("int64")
indices = np.random.permutation(len(X))
N = int(len(X) * SUBSET_RATIO)
X = X.iloc[indices][:N].astype(np.float32)  
y = y[indices][:N]

X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"


class ClassifierModule(nn.Module):
    def __init__(self, n_layers: int, dropout: float, hidden_units: list[int]) -> None:
        super().__init__()

        layers = []
        input_dim = 28 * 28  # Assuming flattened MNIST input

        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_units[i]))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            input_dim = hidden_units[i]

        layers.append(nn.Linear(input_dim, 10))
        layers.append(nn.Softmax(dim=-1))  # Final softmax layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["data"]
        return self.model(x)

X_train_np = X_train.to_numpy().astype(np.float32)
X_test_np = X_test.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy()

def objective(trial: optuna.Trial) -> float:
    # Define hyperparameter search space
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden_units = [trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)]

    # Initialize model with suggested hyperparameters
    model = ClassifierModule(n_layers, dropout, hidden_units)

    net = skorch.NeuralNetClassifier(
        model,
        max_epochs=trial.suggest_int("max_epochs", 10, 50),
        lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        device=device,
        verbose=0,
        callbacks=[SkorchPruningCallback(trial, "valid_acc")],
    )

    net.fit(X_train, y_train)

    return accuracy_score(y_test_np, net.predict(X_test_np))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skorch example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=100,
        help="Number of trials to run in the study (default: 100).",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=600,
        help="Timeout in seconds for the study (default: 600).",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))