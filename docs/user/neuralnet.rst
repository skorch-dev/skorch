.. _neuralnet:

=========
NeuralNet
=========

Using NeuralNet
---------------

The ``NeuralNet`` class and the derived classes are the main touch
point for the user. They wrap the pytorch module while provide an
interface that should be familiar for sklearn users.

Define your pytorch ``Module`` the same way as you always do. Then
pass it to ``NeuralNet``, in conjunction with a pytorch criterion. And
finally, you can call ``fit`` and ``predict``, as with an sklearn
estimator. The finished code could look something like this:

.. code:: python

    class MyModule(torch.nn.Module):
        ...

    net = NeuralNet(
        module=MyModule,
	criterion=torch.nn.NLLLoss,
    )
    net.fit(X, y)
    y_pred = net.predict(X_valid)

Let's see what skorch did for us here:

- wraps the pytorch module in an sklearn interface
- abstracts away the fit loop
- takes care of batching the data

You therefore have a lot less boilerplate code, letting you focus on
what matters. At the same time, skorch is very flexible and can be
extended with ease, getting out of your way when needed.

Most important arguments and methods
------------------------------------

A complete explanation of all arguments and methods of ``NeuralNet``
are found in the API documentation. Here we focus on the main ones.

module
^^^^^^

This is where you pass your pytorch module. Ideally, it should not be
instantiated. Instead, the init arguments for your module should be
passed to ``NeuralNet`` with the ``module__`` prefix. E.g., if your
module takes the arguments ``num_units`` and ``dropout``, the code
would look like this:

.. code:: python

    class MyModule(torch.nn.Module):
        def __init__(self, num_units, dropout):
	    ...

    net = NeuralNet(
        module=MyModule,
	module__num_units=100,
	module__dropout=0.5,
	criterion=torch.nn.NLLLoss,
    )

It is, however, also possible to pass an instantiated module, e.g. a
``torch.nn.Sequential`` instance.

Note that ``skorch`` does not automatically apply any nonlinearities
to the outputs. That means that if you have a classification task, you
should make sure that the final output nonlinearity is a
softmax. Otherwise, when you call ``predict_proba``, you won't get
actual probabilities.

criterion
^^^^^^^^^

This should be a ``pytorch`` (-compatible) criterion. When you use the
``NeuralNetClassifier``, the criterion is set to ``torch.nn.NLLLoss``
by default, for ``NeuralNetRegressor``, it is ``torch.nn.MSELoss``.

saving and loading
^^^^^^^^^^^^^^^^^^


Special arguments
-----------------

Under the hood
--------------

Explain initialize and get methods.
Also explain the parameters ending on '_'

Subclassing NeuralNet
---------------------

NeuralNetClassifier
-------------------

NeuralNetRegressor
------------------
