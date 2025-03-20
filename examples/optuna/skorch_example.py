import argparse
import numpy as np
import optuna
from optuna.integration import SkorchPruningCallback
import skorch
import torch
import torch.nn as nn
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_mnist_data(subset_ratio=0.4):
    mnist = fetch_openml("mnist_784", cache=False, parser="auto")  # Explicit parser for compatibility
    num_samples = int(len(mnist.data) * subset_ratio)
    return mnist.data.iloc[:num_samples].astype(np.float32), mnist.target.iloc[:num_samples].astype("int64")

class ClassifierModule(nn.Module):
    def __init__(self, n_layers: int, dropout: float, hidden_units: list[int]) -> None:
        super().__init__()

        layers = []
        input_dim = 28 * 28  # Assuming flattened MNIST input

        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_units[i]

        layers.append(nn.Linear(input_dim, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["data"]
        return self.model(x)

def convert_to_numpy(X_train, X_test, y_train, y_test):
    return (
        X_train.to_numpy().astype(np.float32),
        X_test.to_numpy().astype(np.float32),
        y_train.to_numpy(),
        y_test.to_numpy()
    )

def objective(trial: optuna.Trial, X_train, X_test, y_train, y_test) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden_units = [trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)]

    model = ClassifierModule(n_layers, dropout, hidden_units)

    X_train_np, X_test_np, y_train_np, y_test_np = convert_to_numpy(X_train, X_test, y_train, y_test)

    net = skorch.NeuralNetClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=trial.suggest_int("max_epochs", 10, 50),
        lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        device=device,
        verbose=0,
        callbacks=[SkorchPruningCallback(trial, "valid_acc")],
    )

    net.fit(X_train_np, y_train_np)

    return accuracy_score(y_test_np, net.predict(X_test_np))

def main(args):
    # Load and preprocess data
    subset_ratio = 0.4
    X, y = load_mnist_data(subset_ratio)

    indices = np.random.permutation(len(X))
    N = int(len(X) * subset_ratio)
    X, y = X.iloc[indices][:N], y.iloc[indices][:N]

    X /= 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Run optimization
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=args.n_trials, timeout=args.timeout)

    # Print results
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial value: {study.best_trial.value}")
    print("Best trial parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skorch example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. MedianPruner stops unpromising trials early.",
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
    main(args)
