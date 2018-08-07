.. image:: https://github.com/dnouri/skorch/blob/master/assets/skorch.svg
   :width: 30%

------------

|build| |coverage| |docs| |powered|

A scikit-learn compatible neural network library that wraps PyTorch.

.. |build| image:: https://travis-ci.org/dnouri/skorch.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.org/dnouri/skorch?branch=master

.. |coverage| image:: https://github.com/dnouri/skorch/blob/master/assets/coverage.svg
    :alt: Test Coverage
    :scale: 100%

.. |docs| image:: https://readthedocs.org/projects/skorch/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://skorch.readthedocs.io/en/latest/?badge=latest

.. |powered| image:: https://github.com/dnouri/skorch/blob/master/assets/powered.svg
    :alt: Powered by
    :scale: 100%
    :target: https://github.com/ottogroup/

=========
Resources
=========

- `Documentation <https://skorch.readthedocs.io/en/latest/?badge=latest>`_
- `Source Code <https://github.com/dnouri/skorch/>`_

=======
Example
=======

To see a more elaborate example, look `here
<https://github.com/dnouri/skorch/tree/master/notebooks/README.md>`__.

.. code:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from torch import nn
    import torch.nn.functional as F

    from skorch import NeuralNetClassifier


    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

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
            X = F.softmax(self.output(X), dim=-1)
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

============
Installation
============

pip installation
================

To install with pip, run:

.. code:: bash

    pip install -U skorch

We recommend to use a virtual environment for this.

From source
===========

If you would like to use the must recent additions to skorch or
help development, you should install skorch from source.

Using conda
===========

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/dnouri/skorch.git
    cd skorch
    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    python setup.py install

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/dnouri/skorch.git
    cd skorch
    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    conda install --file requirements-dev.txt
    python setup.py develop

    py.test  # unit tests
    pylint skorch  # static code checks

Using pip
=========

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/dnouri/skorch.git
    cd skorch
    # create and activate a virtual environment
    pip install -r requirements.txt
    # install pytorch version for your system (see below)
    python setup.py install

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/dnouri/skorch.git
    cd skorch
    # create and activate a virtual environment
    pip install -r requirements.txt
    # install pytorch version for your system (see below)
    pip install -r requirements-dev.txt
    python setup.py develop

    py.test  # unit tests
    pylint skorch  # static code checks

PyTorch
=======

PyTorch is not covered by the dependencies, since the PyTorch
version you need is dependent on your system. For installation
instructions for PyTorch, visit the `PyTorch website
<http://pytorch.org/>`__.

In general, this should work (assuming CUDA 9):

.. code:: bash

    # using conda:
    conda install pytorch cuda90 -c pytorch
    # using pip
    pip install torch

=============
Communication
=============

- `GitHub issues <https://github.com/dnouri/skorch/issues>`_: bug
  reports, feature requests, install issues, RFCs, thoughts, etc.

- Slack: We run the #skorch channel on the `PyTorch Slack server
  <https://pytorch.slack.com/>`_.  If you need an invite, send an
  email to daniel.nouri@gmail.com.
