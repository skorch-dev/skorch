==========
Quickstart
==========

Training a model
----------------

Below, we define our own PyTorch :class:`~torch.nn.Module` and train
it on a toy classification dataset using skorch
:class:`.NeuralNetClassifier`:

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

        def forward(self, X, **kwargs):
            X = self.nonlin(self.dense0(X))
            X = self.dropout(X)
            X = self.nonlin(self.dense1(X))
            X = self.output(X)
            return X


    net = NeuralNetClassifier(
        MyModule,
        max_epochs=10,
        criterion=nn.CrossEntropyLoss(),
        lr=0.1,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
    )

    net.fit(X, y)
    y_proba = net.predict_proba(X)

.. note::

    In this example, instead of using the standard ``softmax`` non-linearity
    with :class:`~torch.nn.NLLLoss` as criterion, no output non-linearity is
    used and :class:`~torch.nn.CrossEntropyLoss` as ``criterion``. The reason is
    that the use of ``softmax`` can lead to numerical instability in some cases.

In an sklearn Pipeline
----------------------

Since :class:`.NeuralNetClassifier` provides an sklearn-compatible
interface, it is possible to put it into an sklearn
:class:`~sklearn.pipeline.Pipeline`:

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', net),
    ])

    pipe.fit(X, y)
    y_proba = pipe.predict_proba(X)


Grid search
-----------

Another advantage of skorch is that you can perform an sklearn
:class:`~sklearn.model_selection.GridSearchCV` or
:class:`~sklearn.model_selection.RandomizedSearchCV`:

.. code:: python

    from sklearn.model_selection import GridSearchCV

    # deactivate skorch-internal train-valid split and verbose logging
    net.set_params(train_split=False, verbose=0)
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
        'module__num_units': [10, 20],
    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

    gs.fit(X, y)
    print(gs.best_score_, gs.best_params_)


What's next?
------------

Please visit the :ref:`tutorials` page to explore additional examples on using skorch!
