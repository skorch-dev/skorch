.. _FAQ:

===
FAQ
===

How do I apply L2 regularization?
---------------------------------

To apply L2 regularization (aka weight decay), `pytorch` supplies the
`weight_decay` parameter, which must be supplied to the optimizer. To
pass this variable in `skorch`, use the double-underscore notation for
the optimizer:

.. code:: python

    net = NeuralNet(
        ...,
        optimizer__weight_decay=0.01,
    )
