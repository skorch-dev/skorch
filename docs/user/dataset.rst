=======
Dataset
=======

This module contains classes and functions related to data handling.


ValidSplit
----------

This class is responsible for performing the :class:`.NeuralNet`\'s
internal cross validation. For this, it sticks closely to the sklearn
standards. For more information on how sklearn handles cross
validation, look `here
<http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators>`_.

The first argument that :class:`.ValidSplit` takes is ``cv``. It works
analogously to the ``cv`` argument from sklearn
:class:`~sklearn.model_selection.GridSearchCV`,
:func:`~sklearn.model_selection.cross_val_score`, etc. For those not
familiar, here is a short explanation of what you may pass:

- ``None``: Use the default 3-fold cross validation.
- integer: Specifies the number of folds in a ``(Stratified)KFold``,
- float: Represents the proportion of the dataset to include in the
  validation split (e.g. ``0.2`` for 20%).
- An object to be used as a cross-validation generator.
- An iterable yielding train, validation splits.

Furthermore, :class:`.ValidSplit` takes a ``stratified`` argument that
determines whether a stratified split should be made (only makes sense
for discrete targets), and a ``random_state`` argument, which is used
in case the cross validation split has a random component.

One difference to sklearn\'s cross validation is that skorch
makes only a single split. In sklearn, you would expect that in a
5-fold cross validation, the model is trained 5 times on the different
combination of folds. This is often not desirable for neural networks,
since training takes a lot of time. Therefore, skorch only ever
makes one split.

If you would like to have all splits, you can still use skorch in
conjunction with the sklearn functions, as you would do with any
other sklearn\-compatible estimator. Just remember to set
``train_split=None``, so that the whole dataset is used for
training. Below is shown an example of making out-of-fold predictions
with skorch and sklearn:

.. code:: python

    net = NeuralNetClassifier(
        module=MyModule,
        train_split=None,
    )

    from sklearn.model_selection import cross_val_predict

    y_pred = cross_val_predict(net, X, y, cv=5)


Dataset
-------

In PyTorch, we have the concept of a
:class:`~torch.utils.data.Dataset` and a
:class:`~torch.utils.data.DataLoader`. The former is purely the
container of the data and only needs to implement ``__len__()`` and
``__getitem__(<int>)``. The latter does the heavy lifting, such as
sampling, shuffling, and distributed processing.

skorch uses the PyTorch :class:`~torch.utils.data.DataLoader`\s by default.
skorch supports PyTorch's :class:`~torch.utils.data.Dataset` when calling
:func:`~skorch.net.NeuralNet.fit` or 
:func:`~skorch.net.NeuralNet.partial_fit`. Details on how to use PyTorch's
:class:`~torch.utils.data.Dataset` with skorch, can be found in 
:ref:`faq_how_do_i_use_a_pytorch_dataset_with_skorch`.
In order to support other data formats, we provide our own
:class:`.Dataset` class that is compatible with:

- :class:`numpy.ndarray`\s
- PyTorch :class:`~torch.Tensor`\s
- scipy sparse CSR matrices
- pandas DataFrames or Series

Note that currently, sparse matrices are cast to dense arrays during
batching, given that PyTorch support for sparse matrices is still very
incomplete. If you would like to prevent that, you need to override
the ``transform`` method of :class:`~torch.utils.data.Dataset`.

In addition to the types above, you can pass dictionaries or lists of
one of those data types, e.g. a dictionary of
:class:`numpy.ndarray`\s. When you pass dictionaries, the keys of the
dictionaries are used as the argument name for the
:meth:`~torch.nn.Module.forward` method of the net's
``module``. Similarly, the column names of pandas ``DataFrame``\s are
used as argument names. The example below should illustrate how to use
this feature:

.. code:: python

    import numpy as np
    import torch
    import torch.nn.functional as F

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.dense_a = torch.nn.Linear(10, 100)
            self.dense_b = torch.nn.Linear(20, 100)
            self.output = torch.nn.Linear(200, 2)

        def forward(self, key_a, key_b):
            hid_a = F.relu(self.dense_a(key_a))
            hid_b = F.relu(self.dense_b(key_b))
            concat = torch.cat((hid_a, hid_b), dim=1)
            out = F.softmax(self.output(concat))
            return out

    net = NeuralNetClassifier(MyModule)

    X = {
        'key_a': np.random.random((1000, 10)).astype(np.float32),
        'key_b': np.random.random((1000, 20)).astype(np.float32),
    }
    y = np.random.randint(0, 2, size=1000)

    net.fit(X, y)

Note that the keys in the dictionary ``X`` exactly match the argument
names in the :meth:`~torch.nn.Module.forward` method. This way, you
can easily work with several different types of input features.

The :class:`.Dataset` from skorch makes the assumption that you always
have an ``X`` and a ``y``, where ``X`` represents the input data and
``y`` the target. However, you may leave ``y=None``, in which case
:class:`.Dataset` returns a dummy variable.

:class:`.Dataset` applies a transform final transform on the data
before passing it on to the PyTorch
:class:`~torch.utils.data.DataLoader`. By default, it replaces ``y``
by a dummy variable in case it is ``None``. If you would like to
apply your own transformation on the data, you should subclass
:class:`.Dataset` and override the
:func:`~skorch.dataset.Dataset.transform` method, then pass your
custom class to :class:`.NeuralNet` as the ``dataset`` argument.

Preventing dictionary unpacking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As noted, when ``X`` is a dictionary, it is automatically unpacked when passed
to the module's :meth:`~torch.nn.Module.forward` method. Sometimes, you may want
to prevent this, e.g. because you're using a ``module`` from another library
that expects a dict as input, or because the exact dict keys are unknown. This
can be achieved by wrapping the original ``module`` and packing the dict again:

.. code:: python

    from other_lib import ModuleExpectingDict

    class WrappedModule(ModuleExpectingDict):
        def forward(self, **kwargs):
            # catch **kwargs, pass as a dict
            return super().forward(kwargs)

Similarly, wrapping the ``module`` can be used to make any other desired changes
to the input arguments.
