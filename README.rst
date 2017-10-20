.. image:: ./assets/skorch.svg
   :width: 30%
======

A scikit-learn compatible neural network library that wraps pytorch.

Example
-------

To see a more elaborate example, look `here
<https://github.com/dnouri/skorch/tree/master/notebooks/README.md>`__.

.. code:: python

    import numpy as np
    from sklearn.datasets import make_classification
    import torch
    from torch import nn
    import torch.nn.functional as F

    from skorch.net import NeuralNetClassifier


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

In an sklearn Pipeline:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', net),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)

With grid search

.. code:: python

    from sklearn.model_selection import GridSearchCV


    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
        'module__num_units': [10, 20],
    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

    gs.fit(X, y)
    print(gs.best_score_, gs.best_params_)

Installation
------------

conda
~~~~~

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

For users
^^^^^^^^^

Note: pip installation will follow soon.

.. code:: shell

    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    python setup.py install

For developers
^^^^^^^^^^^^^^

.. code:: shell

    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    conda install --file requirements-dev.txt
    python setup.py develop

    py.test  # unit tests
    pylint skorch  # static code checks

pip
~~~

Same as for conda, but to install main requirements, run:

.. code:: shell

    pip install -r requirements.txt

pytorch
~~~~~~~

For installation instructions for pytorch, visit the `pytorch
website <http://pytorch.org/>`__.

In general, this should work:

.. code:: shell

    # using conda:
    conda install pytorch cuda80 -c soumith
    # using pip
    pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
