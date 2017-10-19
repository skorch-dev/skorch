.. _neuralnet:

=========
NeuralNet
=========

Using NeuralNet
---------------

:class:`NeuralNet <skorch.net.NeuralNet>` and the derived classes are
the main touch point for the user. They wrap the ``torch.nn.Module``
while providing an interface that should be familiar for sklearn
users.

Define your ``torch.nn.Module`` the same way as you always do. Then
pass it to ``NeuralNet``, in conjunction with a PyTorch criterion.
Finally, you can call :func:`fit <skorch.net.NeuralNet.fit>` and
:func:`predict <skorch.net.NeuralNet.predict>`, as with an sklearn
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

- wraps the PyTorch module in an sklearn interface
- converts the numpy ``array``\s to ``torch.Tensor``\s
- abstracts away the fit loop
- takes care of batching the data

You therefore have a lot less boilerplate code, letting you focus on
what matters. At the same time, skorch is very flexible and can be
extended with ease, getting out of your way as much as possible.

Initialization
^^^^^^^^^^^^^^

In general, when you instantiate the ``NeuralNet`` instance, only the
given arguments are stored. They are stored exactly as you pass them
to ``NeuralNet``. For instance, the ``module`` will remain
uninstantiated. This is to make sure that the arguments you pass are
not touched afterwards, which makes it possible to clone the
``NeuralNet`` instance, for instance.

Only when the ``fit`` or ``initialize`` method are called, are the
different attributes of the net, such as the ``module``,
initialized. An initialzed attribute's name always ends on an
underscore; e.g., the initialized ``module`` is called
``module_``. (This is the same nomenclature as sklearn uses.)
Thefore, you always know which attributes you set and which ones were
created by ``NeuralNet``.

The only exception is the :ref:`history <history>` attribute, which is not
set by the user.

Most important arguments and methods
------------------------------------

A complete explanation of all arguments and methods of ``NeuralNet``
are found in the skorch API documentation. Here we focus on the
main ones.

module
^^^^^^

This is where you pass your PyTorch module. Ideally, it should not be
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

Note that skorch does not automatically apply any nonlinearities
to the outputs. That means that if you have a classification task, you
should make sure that the final output nonlinearity is a
softmax. Otherwise, when you call ``predict_proba``, you won't get
actual probabilities.

criterion
^^^^^^^^^

This should be a PyTorch (-compatible) criterion. When you use the
``NeuralNetClassifier``, the criterion is set to ``torch.nn.NLLLoss``
by default, for ``NeuralNetRegressor``, it is ``torch.nn.MSELoss``.

After initializing the ``NeuralNet``, the initialized criterion will
stored in the ``criterion_`` attribute.

optimizer
^^^^^^^^^

This should be a PyTorch optimizer,
e.g. ``torch.optim.SGD``. After initializing the ``NeuralNet``, the
initialized optimizer will stored in the ``optimizer_`` attribute.

lr
^^^

The learning rate. This argument exists for convenience, since it
could also be set by ``optimizer__lr`` instead. However, it is used so
often that we provided this shortcut. If you set both ``lr`` and
``optimizer__lr``, the latter have precedence.

max_epochs
^^^^^^^^^^

The maximum number of epochs to train with each ``fit`` call. When you
call ``fit``, the net will train for this many epochs, except if you
interrupt training before the end (e.g. by using an early stopping
callback or interrupt manually with ctrl+c).

If you want to change the number of epochs to train, you can either
set a different value for ``max_epochs``, or you call ``fit_loop``
instead of ``fit`` and pass the desired number of epochs explicitely:

.. code:: python

    net.fit_loop(X, y, epochs=20)


batch_size
^^^^^^^^^^

This argument controls the batch size for ``iterator_train`` and
``iterator_valid`` at the same time. ``batch_size=128`` is thus a
convenient shortcut for explicitely typing
``iterator_train__batch_size=128`` and
``iterator_valid__batch_size=128``. If you set all three arguments,
the latter two will have precedence.

train_split
^^^^^^^^^^^

This determines the ``NeuralNet``\'s internal train/validation
split. By default, 20% of the incoming data is reserved for
validation. If you set this value to ``None``, all the data is used
for training.

For more details, please look at :ref:`dataset <dataset>`.

callbacks
^^^^^^^^^

For more details on the callback classes, please look at
:ref:`callbacks <callbacks>`.

By default, ``NeuralNet`` and its subclasses start with a couple of
useful callbacks. Those are defined in the ``get_default_callbacks``
method and include, for instance, callbacks for measuring and printing
model performance.

In addition to the default callbacks, you may provide your own
callbacks. There are a couple of ways to pass callbacks to the
``NeuralNet`` instance. The easiest way is to pass a list of all your
callbacks to the ``callbacks`` argument:

.. code:: python

    net = NeuralNet(
        module=MyModule,
	callbacks=[
	    MyCallback1(...),
	    MyCallback2(...),
	],
    )

Inside the ``NeuralNet`` instance, each callback will receive a
separate name. Since we provide no name in the example above, the
class name will taken, which will lead to a name collision in case of
two or more callbacks of the same class. This is why it is better to
initialize the callbacks with a list of tuples of *name* and *callback
instance*, like this:

.. code:: python

    net = NeuralNet(
        module=MyModule,
	callbacks=[
	    ('cb1', MyCallback1(...)),
	    ('cb2', MyCallback2(...)),
	],
    )

This approach of passing a list of *name*, *instance* tuples should be
familiar to users of sklearn\ ``Pipeline``\s and
``FeatureUnion``\s.

An additonal advantage of this way of passing callbacks is that it
allows to pass arguments to the callbacks by name (using the
double-underscore notation):

.. code:: python

    net.set_params(cb1__foo=123, cb2__bar=456)

Use this, for instance, when trying out different callback parameters
in a grid search.

*Note*: The user-defined callbacks are always called *after* the
default callbacks. This is so that user-defined callbacks can make use
of the things provided by the default callbacks. The only exception is
the default callback ``PrintLog``, which is always called last.

cold_start
^^^^^^^^^^

This argument determines whether each ``fit`` call leads to a
re-initialization of the ``NeuralNet`` or not. By default, when
calling ``fit``, the parameters of the net are initialized, so your
previous training progress is lost (consistent with the sklearn
``fit`` calls). In contrast, with ``cold_start=False``, each ``fit``
call will continue from the most recent state.

use_cuda
^^^^^^^^

As the name suggests, this determines whether CUDA should be used or
not. If ``True``, the incoming data will be transferred to CUDA before
being passed to the ``pytorch module``.

initialize()
^^^^^^^^^^^^

As mentioned earlier, upon instantiating the ``NeuralNet`` instance,
the net's components are not yet initialized. That means, e.g., that
the weights and biases of the layers are not yet set. This only
happens after the ``initialize`` call. However, when you call ``fit``
and the net is not yet initialized, ``initialize`` is called
automatically. You thus rarely need to call ``initialize`` manually.

The ``initialize`` methods itself calls a couple of other
initialization methods that are specific to each component. E.g.,
``initialize_module`` is responsible for initializing the ``pytorch
module``. Therefore, if you have special needs for initializing the
module, it is enough to override ``initialize_module``, you don't need
to override the whole ``initialize`` method.

fit(X, y)
^^^^^^^^^

This is one of the main methods you will use. It contains everything
required to train the model, be it batching of the data, triggering
the callbacks, or handling the internal validation set.

In general, we assume there to be an ``X`` and a ``y``. If you have
more input data than just one array, it is possible for ``X`` to be a
list or dictionary of data (see :ref:`dataset <dataset>`). And if your
task does not have an actual ``y``, you may pass ``y=None``.

In addition to ``fit``, there is also the ``partial_fit`` method,
known from some sklearn estimators. ``partial_fit`` allows you to
continue training from your current status, even if you set
``cold_start=True``. A further use case for ``partial_fit`` is when
your data does not fit into memory and you thus need to have several
training steps.

*Tip* :
skorch gracefully cathes the ``KeyboardInterrupt``
exception. Therefore, during a training run, you can send a
``KeyboardInterrupt`` signal without the Python process exiting
(typically, ``KeyboardInterrupt`` can be triggered by *ctrl+c* or, in
a Jupyter notebook, by clicking *Kernel* -> *Interrupt*). This way, when
your model has reached a good score before ``max_epochs`` have been
reached, you can dynamically stop training.

predict(X) and predict_proba(X)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods perform an inference step on the input data and return
``numpy array``\s. By default, ``predict_proba`` will return whatever
it is that the ``module``\'s ``forward`` method returns, cast to a
``numpy array``.

If casting the ``forward``\-output to ``numpy`` is impossible, you
will get an error. In that case, you should consider returning a torch
tensor from your ``forward`` method, as this tensor can be converted
to a ``numpy`` array. Alternatively, consider using the
``forward_iter`` method to generate outputs from the ``module``, or
directly call ``net.module_(X)``.

The ``predict`` method tries to return the class labels by applying
the argmax over the last axis of the result of
``predict_proba``. Obviously, this only makes sense if
``predict_proba`` returns class probabilities. If this is not true,
you should just use ``predict_proba``. (Since it deals with regression
tasks, ``NeuralNetRegressor``\'s ``predict`` method just returns the
result from ``predict_proba``, without applying argmax.)

saving and loading
^^^^^^^^^^^^^^^^^^

skorch provides two ways to persist your model. First it is
possible to store the model using Python's ``pickle`` function. This
saves the whole model, including hyperparameters. This is useful when
you don't want to initialize your model before loading its parameters,
or when your ``NeuralNet`` is part of an ``sklearn Pipeline``:

.. code:: python

    net = NeuralNet(
        module=MyModule,
	criterion=torch.nn.NLLLoss,
    )

    model = Pipeline([
        ('my-features', get_features()),
	('net', net),
    ])
    model.fit(X, y)

    # saving
    with open('some-file.pkl', 'wb') as f:
        pickle.dump(model, f)

    # loading
    with open('some-file.pkl', 'rb') as f:
        model = pickle.load(f)

The disadvantage of pickling is that if your underlying code changes,
unpickling might raise errors. Also, some Python code (e.g. lambda
functions) cannot be pickled.

For this reason, we provide a second method for persisting your
model. To use it, call the ``save_params`` and ``load_params`` method
on ``NeuralNet``. Under the hood, this saves the ``module``\'s
``state_dict``, i.e. only the weights and biases of the
``module``. This is more robust to changes in the code but requires
you to initialize a ``NeuralNet`` to load the parameters again:

.. code:: python

    net = NeuralNet(
        module=MyModule,
	criterion=torch.nn.NLLLoss,
    )

    model = Pipeline([
        ('my-features', get_features()),
	('net', net),
    ])
    model.fit(X, y)

    net.save_params('some-file.pkl')

    new_net = NeuralNet(
        module=MyModule,
	criterion=torch.nn.NLLLoss,
    )
    new_net.initialize()  # This is important!
    new_net.load_params('some-file.pkl')


Special arguments
-----------------

In addition to the arguments explicitely listed for ``NeuralNet``,
there are some arguments with special prefixes, as shown below:

.. code:: python

    class MyModule(torch.nn.Module):
        def __init__(self, num_units, dropout):
	    ...

    net = NeuralNet(
        module=MyModule,
	module__num_units=100,
	module__dropout=0.5,
	criterion=torch.nn.NLLLoss,
	criterion__weight=weight,
	optimizer=torch.optim.SGD,
	optimizer__momentum=0.9,
    )

Those arguments are used to initialize your ``Module``, ``criterion``,
etc. They are not fixed because we cannot know them in advance; in
fact, you can define any parameter for your ``Module`` or other
components.

All special prefixes are stored in the ``prefixes_`` class attribute
of ``NeuralNet``. Currently, they are:

- ``module``
- ``iterator_train``
- ``iterator_valid``
- ``optimizer``
- ``criterion``
- ``callbacks``

Subclassing NeuralNet
---------------------

Apart from the ``NeuralNet`` base class, we provide
``NeuralNetClassifier`` and ``NeuralNetRegressor`` for typical
classification and regressions tasks. They should work as drop-in
replacements for sklearn classifiers and regressors.

The ``NeuralNet`` class is a little less opinionated about the
incoming data, e.g. it does not determine a loss function by
default. Therefore, if you want to write your own subclass for a
special use case, you would typically subclass from ``NeuralNet``.

skorch aims at making subclassing as easy as possible, so that
doesn't stand in your way. For instance, all components (``module``,
``optimizer``, etc.) have their own initialization method
(``initialize_module``, ``initialize_optimizer``, etc.). That way, if
you want to modify the initialization of a component, you can easily
do so.

Additonally, ``NeuralNet`` has a couple of ``get_*`` methods for when
a component is retrieved repeatedly. E.g., ``get_loss`` is called when
the loss is determined. Below we show an example of overriding
``get_loss`` to add L1 regularization to our total loss:

.. code:: python

    class RegularizedNet(NeuralNet):
        def __init__(self, *args, lambda1=0.01, **kwargs):
            super().__init__(*args, **kwargs)
            self.lambda1 = lambda1

        def get_loss(self, y_pred, y_true, X=None, train=False):
            loss = super().get_loss(y_pred, y_true, X=X, train=train)
            loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
            return loss

*Note*: This example also reguralizes the biases, which you typically
 don't need to do.
