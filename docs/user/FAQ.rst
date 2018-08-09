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

By default, when you call :func:`~skorch.net.NeuralNet.fit` more than
once, the training starts from zero instead of from where it was left.
This is in line with sklearn\'s behavior but not always desired. If
you would like to continue training, use
:func:`~skorch.net.NeuralNet.partial_fit` instead of
:func:`~skorch.net.NeuralNet.fit`. Alternatively, there is the
``warm_start`` argument, which is ``False`` by default. Set it to
``True`` instead and you should be fine.


How do I shuffle my train batches?
----------------------------------

skorch uses :class:`~torch.utils.data.DataLoader` from PyTorch under
the hood.  This class takes a couple of arguments, for instance
``shuffle``. We therefore need to pass the ``shuffle`` argument to
:class:`~torch.utils.data.DataLoader`, which we achieve by using the
double-underscore notation (as known from sklearn):

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
that, try to wrap your dictionary into a :class:`.SliceDict`. This is
a data container that partly behaves like a dict, partly like an
ndarray. For more details on how to do this, have a look at the
coresponding `data section
<https://nbviewer.jupyter.org/github/dnouri/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Working-with-sklearn-FunctionTransformer-and-GridSearch>`__
in the notebook.


I want to use sample_weight, how can I do this?
-----------------------------------------------

Some scikit-learn models support to pass a ``sample_weight`` argument
to ``fit`` calls as part of the ``fit_params``. This allows you to
give different samples different weights in the final loss
calculation.

In general, skorch supports ``fit_params``, but unfortunately just
calling ``net.fit(X, y, sample_weight=sample_weight)`` is not enough,
because the ``fit_params`` are not split into train and valid, and are
not batched, resulting in a mismatch with the training batches.

Fortunately, skorch supports passing dictionaries as arguments, which
are actually split into train and valid and then batched. Therefore,
the best solution is to pass the ``sample_weight`` with ``X`` as a
dictionary. Below, there is example code on how to achieve this:

.. code:: python

    X, y = get_data()
    # put your X into a dict if not already a dict
    X = {'data': X}
    # add sample_weight to the X dict
    X['sample_weight'] = sample_weight

    class MyModule(nn.Module):
        ...
        def forward(self, data, sample_weight):
            # when X is a dict, its keys are passed as kwargs to forward, thus
            # our forward has to have the arguments 'data' and 'sample_weight';
            # usually, sample_weight can be ignored here
            ...

    class MyNet(NeuralNet):
        def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            # override get_loss to use the sample_weight from X
            loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
            sample_weight = X['sample_weight']
            loss_reduced = (sample_weight * loss_unreduced).mean()
            return loss_reduced

    # make sure to pass reduce=False to your criterion, since we need the loss
    # for each sample so that it can be weighted
    net = MyNet(MyModule, ..., criterion__reduce=False)
    net.fit(X, y)


I already split my data into training and validation sets, how can I use them?
------------------------------------------------------------------------------

If you have predefined training and validation datasets that are
subclasses of PyTorch :class:`~torch.utils.data.Dataset`, you can use
:func:`~skorch.helper.predefined_split` to wrap your validation dataset and
pass it to :class:`~skorch.net.NeuralNet`'s ``train_split`` parameter:

.. code:: python

    from skorch.helper import predefined_split

    net = NeuralNet(
        ...,
        train_split=predefined_split(valid_ds)
    )
    net.fit(train_ds)

If you split your data by using :func:`~sklearn.model_selection.train_test_split`,
you can create your own skorch :class:`~skorch.dataset.Dataset`, and then pass
it to :func:`~skorch.helper.predefined_split`:

.. code:: python

    from sklearn.model_selection import train_test_split
    from skorch.helper import predefined_split
    from skorch.dataset import Dataset

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    valid_ds = Dataset(X_test, y_test)

    net = NeuralNet(
        ...,
        train_split=predefined_split(valid_ds)
    )

    net.fit(X_train, y_train)
