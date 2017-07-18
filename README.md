# Inferno

A scikit-learn compatible neural network library that wraps pytorch.

## Example

```
import numpy as np
from sklearn.datasets import make_classification
import torch
from torch import nn
import torch.nn.functional as F

from inferno.net import NeuralNetClassifier


X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
)

net.fit(X, y)
y_proba = net.predict_proba(X)

```

In an sklearn Pipeline:

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


pipe = Pipeline([
    ('scale', StandardScaler()),
    ('net', net),
])

pipe.fit(X, y)
y_proba = pipe.predict_proba(X)

```

With grid search

```
from sklearn.model_selection import GridSearchCV


params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10, 20],
    'module__num_units': [10, 20],
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

gs.fit(X, y)
print(gs.best_score_, gs.best_params_)

```
## Installation

### conda

You need a working conda installation. Get the correct miniconda for
your system from [here](https://conda.io/miniconda.html).

#### For users

```
conda env create
source activate inferno
python setup.py install
```

#### For developers

```
conda env create
source activate inferno
conda install --file requirements-dev.txt
python setup.py develop

py.test  # unit tests
pylint inferno  # static code checks
```

### pip

Same as for conda, but to install main requirements, run:

```
pip install -r requirements.txt
```
