import numpy as np
import torch
from accelerate import Accelerator
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from torch import nn


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(100, 2)
        self.nonlin = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.dense0(X)
        X = self.nonlin(X)
        return X


class Net(BaseEstimator):
    def __init__(self, module, accelerator):
        self.module = module
        self.accelerator = accelerator

    def fit(self, X, y, **fit_params):
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

        self.module = self.accelerator.prepare(self.module)
        optimizer = self.accelerator.prepare(optimizer)
        dataloader = self.accelerator.prepare(dataloader)

        # training
        self.module.train()
        for epoch in range(5):
            for source, targets in dataloader:
                optimizer.zero_grad()
                output = self.module(source)
                loss = nn.functional.nll_loss(output, targets)
                self.accelerator.backward(loss)
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.module.eval()
        X = torch.as_tensor(X)
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        dataloader = self.accelerator.prepare(dataloader)

        probas = []
        with torch.no_grad():
            for source, *_ in dataloader:
                output = self.module(source)
                output = self.accelerator.gather_for_metrics(output)
                output = output.cpu().detach().numpy()
                probas.append(output)

        return np.vstack(probas)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return y_proba.argmax(1)


class MyAccelerator(Accelerator):
    def __deepcopy__(self, memo):
        return self


def main():
    X, y = make_classification(10000, n_features=100, n_informative=50, random_state=0)
    X = X.astype(np.float32)

    module = MyModule()
    accelerator = MyAccelerator()
    net = Net(module, accelerator)
    # cross_validate creates a deepcopy of the accelerator attribute
    res = cross_validate(
        net, X, y, cv=2, scoring='accuracy', verbose=3, error_score='raise',
    )
    print(res)


if __name__ == '__main__':
    main()
