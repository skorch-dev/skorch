.. image:: https://github.com/skorch-dev/skorch/blob/master/assets/skorch_bordered.svg
   :width: 30%

------------

|build| |coverage| |docs| |huggingface| |powered|

A scikit-learn compatible neural network library that wraps PyTorch.

.. |build| image:: https://github.com/skorch-dev/skorch/workflows/tests/badge.svg
    :alt: Test Status
    :scale: 100%

.. |coverage| image:: https://github.com/skorch-dev/skorch/blob/master/assets/coverage.svg
    :alt: Test Coverage
    :scale: 100%

.. |docs| image:: https://readthedocs.org/projects/skorch/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://skorch.readthedocs.io/en/latest/?badge=latest

.. |huggingface| image:: https://github.com/skorch-dev/skorch/actions/workflows/test-hf-integration.yml/badge.svg
    :alt: Hugging Face Integration
    :scale: 100%
    :target: https://github.com/skorch-dev/skorch/actions/workflows/test-hf-integration.yml

.. |powered| image:: https://github.com/skorch-dev/skorch/blob/master/assets/powered.svg
    :alt: Powered by
    :scale: 100%
    :target: https://github.com/ottogroup/

=========
Resources
=========

- `Documentation <https://skorch.readthedocs.io/en/latest/?badge=latest>`_
- `Source Code <https://github.com/skorch-dev/skorch/>`_
- `Installation <https://github.com/skorch-dev/skorch#installation>`_

========
Examples
========

To see more elaborate examples, look `here
<https://github.com/skorch-dev/skorch/tree/master/notebooks/README.md>`__.

.. code:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from torch import nn
    from skorch import NeuralNetClassifier

    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    class MyModule(nn.Module):
        def __init__(self, num_units=10, nonlin=nn.ReLU()):
            super().__init__()

            self.dense0 = nn.Linear(20, num_units)
            self.nonlin = nonlin
            self.dropout = nn.Dropout(0.5)
            self.dense1 = nn.Linear(num_units, num_units)
            self.output = nn.Linear(num_units, 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.dropout(X)
            X = self.nonlin(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    net = NeuralNetClassifier(
        MyModule,
        max_epochs=10,
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    net.fit(X, y)
    y_proba = net.predict_proba(X)

In an `sklearn Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', net),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)

With `grid search <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_:

.. code:: python

    from sklearn.model_selection import GridSearchCV

    # deactivate skorch-internal train-valid split and verbose logging
    net.set_params(train_split=False, verbose=0)
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
        'module__num_units': [10, 20],
    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=2)

    gs.fit(X, y)
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))


skorch also provides many convenient features, among others:

- `Learning rate schedulers <https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.LRScheduler>`_ (Warm restarts, cyclic LR and many more)
- `Scoring using sklearn (and custom) scoring functions <https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.EpochScoring>`_
- `Early stopping <https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.EarlyStopping>`_
- `Checkpointing <https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.Checkpoint>`_
- `Parameter freezing/unfreezing <https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.Freezer>`_
- `Progress bar <https://skorch.readthedocs.io/en/stable/callbacks.html#skorch.callbacks.ProgressBar>`_ (for CLI as well as jupyter)
- `Automatic inference of CLI parameters <https://github.com/skorch-dev/skorch/tree/master/examples/cli>`_
- `Integration with GPyTorch for Gaussian Processes <https://skorch.readthedocs.io/en/latest/user/probabilistic.html>`_
- `Integration with Hugging Face 🤗 <https://skorch.readthedocs.io/en/stable/user/huggingface.html>`_

============
Installation
============

skorch requires Python 3.7 or higher.

conda installation
==================

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

To install skorch, you need to use the conda-forge channel:

.. code:: bash

    conda install -c conda-forge skorch

We recommend to use a `conda virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

**Note**: The conda channel is *not* managed by the skorch
maintainers. More information is available `here
<https://github.com/conda-forge/skorch-feedstock>`__.

pip installation
================

To install with pip, run:

.. code:: bash

    python -m pip install -U skorch

Again, we recommend to use a `virtual environment
<https://docs.python.org/3/tutorial/venv.html>`_ for this.

From source
===========

If you would like to use the most recent additions to skorch or
help development, you should install skorch from source.

Using conda
-----------

To install skorch from source using conda, proceed as follows:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda env create
    conda activate skorch
    python -m pip install .

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda env create
    conda activate skorch
    python -m pip install -e .

    py.test  # unit tests
    pylint skorch  # static code checks

Using pip
---------

For pip, follow these instructions instead:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    python -m pip install -r requirements.txt
    # install pytorch version for your system (see below)
    python -m pip install .

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    python -m pip install -r requirements.txt
    # install pytorch version for your system (see below)
    python -m pip install -r requirements-dev.txt
    python -m pip install -e .

    py.test  # unit tests
    pylint skorch  # static code checks

PyTorch
=======

PyTorch is not covered by the dependencies, since the PyTorch version
you need is dependent on your OS and device. For installation
instructions for PyTorch, visit the `PyTorch website
<http://pytorch.org/>`__. skorch officially supports the last four
minor PyTorch versions, which currently are:

- 1.9.1
- 1.10.2
- 1.11.0
- 1.12.0

However, that doesn't mean that older versions don't work, just that
they aren't tested. Since skorch mostly relies on the stable part of
the PyTorch API, older PyTorch versions should work fine.

In general, running this to install PyTorch should work (assuming CUDA
11.1):

.. code:: bash

    # using conda:
    conda install pytorch cudatoolkit==11.1 -c pytorch
    # using pip
    python -m pip install torch

==================
External resources
==================

- @jakubczakon: `blog post
  <https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem>`_
  "8 Creators and Core Contributors Talk About Their Model Training
  Libraries From PyTorch Ecosystem" 2020
- @BenjaminBossan: `talk 1
  <https://www.youtube.com/watch?v=Qbu_DCBjVEk>`_ "skorch: A
  scikit-learn compatible neural network library" at PyCon/PyData 2019
- @githubnemo: `poster <https://github.com/githubnemo/skorch-poster>`_
  for the PyTorch developer conference 2019
- @thomasjpfan: `talk 2 <https://www.youtube.com/watch?v=0J7FaLk0bmQ>`_
  "Skorch: A Union of Scikit learn and PyTorch" at SciPy 2019
- @thomasjpfan: `talk 3 <https://www.youtube.com/watch?v=yAXsxf2CQ8M>`_
  "Skorch - A Union of Scikit-learn and PyTorch" at PyData 2018

=============
Communication
=============

- `GitHub issues <https://github.com/skorch-dev/skorch/issues>`_: bug
  reports, feature requests, install issues, RFCs, thoughts, etc.

- Slack: We run the #skorch channel on the `PyTorch Slack server
  <https://pytorch.slack.com/>`_, for which you can `request access
  here <https://bit.ly/ptslack>`_.
