===
FAQ
===

How do I apply L2 regularization?
---------------------------------

To apply L2 regularization (aka weight decay), PyTorch supplies
the ``weight_decay`` parameter, which must be supplied to the
optimizer. To pass this variable in skorch, use the
double-underscore notation for the optimizer:

.. code:: python

    net = NeuralNet(
        ...,
        optimizer__weight_decay=0.01,
    )


How can I continue training my model?
-------------------------------------

By default, when you call ``fit`` more than once, the training starts
from zero instead of from where it was left. This is in line with
sklearn\'s behavior but not always desired. If you would like to
continue training, use ``partial_fit`` instead of
``fit``. Alternatively, there is the ``warm_start`` argument, which is
``False`` by default. Set it to ``True`` instead and you should be
fine.


How do I shuffle my train batches?
----------------------------------

skorch uses the ``DataLoader`` from PyTorch under the
hood. This ``DataLoader`` takes a couple of arguments, for instance
``shuffle``. We therefore need to pass the ``shuffle`` argument to
``DataLoader``, which we achieve by using the double-underscore
notation (as known from sklearn):

.. code:: python

    net = NeuralNet(
        ...,
        iterator_train__shuffle=True,
    )

Note that we have an ``iterator_train`` for the training data and an
``iterator_valid`` for validation and test data. In general, you only
want to shuffle the train data, which is what the code above does.


How do I use sklearn GridSeachCV when my data is in a dictionary?
-----------------------------------------------------------------

skorch supports dicts as input but sklearn doesn't. To get around
that, try to wrap your dictionary into a ``SliceDict``. This is a data
container that partly behaves like a dict, partly like an ndarray. For
more details on how to do this, have a look at the coresponding
`data section
<https://nbviewer.jupyter.org/github/dnouri/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Working-with-sklearn-FunctionTransformer-and-GridSearch>`__
in the notebook.
