===========
Performance
===========

Since skorch provides extra functionality on top of a pure PyTorch training
code, it is expected that it will add an overhead to the total runtime. For
typical workloads, this overhead should be unnoticeable.

In a few situations, skorch's extra functionality may add significant overhead.
This is especially the case when the amount of data and the neural net are
relatively small. The reason is that typically, most time is spent on the
``forward``, ``backward``, and parameter update calls. When those are really
fast, the skorch overhead will get noticed.

There are, however, a few things that can be done to reduce the skorch overhead.
We will focus on accelerating the training process, where the overhead should be
largest. Below, some mitigations are described, including the potential
downsides.

First make sure that there is no significant slowdown
------------------------------------------------------

Neural nets are notoriously slow to train. Therefore, if your training takes a
lot of time, that doesn't automatically mean that the skorch overhead is at
fault. Maybe the training would take the same time without skorch. If you have
some measurements about training the same model without skorch, first make sure
that this points to skorch being the culprit before trying to optimize using the
mitigations described below. If it turns out skorch is not the culprit, look
into optimizing the performance of PyTorch code in general.

Many people use skorch for hyper-parameter search. Remember that this implies
fitting the model repeatedly, thus a long run time is expected. E.g. if you run
a grid search on two hyper-parameters, each with 10 variants, and 5 splits,
there will actually be 10 x 10 x 5 fit calls, so expect the process to take
approximately 500 times as long as a single model fit. Increase the verbosity on
the grid search to get a better idea on the progress (e.g. ``GridSerachCV(...,
verbose=3)``).

Turning off verbosity
---------------------

By default, skorch produces a print log of the training progress. This is useful
for checking the training progress, monitor overfitting, etc. If you don't need
these diagnostics, you can turn them off via the ``verbose`` parameter. This
way, printing is deactivated, saving time on i/o. You can still access the
diagnostics through the ``history`` attribute after training has finished.

.. code:: python

   net = NeuralNet(..., verbose=0)  # turn off verbosity
   net.fit(X, y)
   train_loss = net.history[..., 'train_loss']  # access history as usual

Disabling callbacks all together
--------------------------------

If you don't need any callbacks at all, turning them off can be potential time
saver. Callbacks present the most significant "extra" that skorch provides over
pure PyTorch, hence they might add a lot of overhead for small workloads. By
turning them off, you lose their functionality, though. It's up to you to
determine if that's a worthwhile trade-off or not. For instance, in contrast to
just turning down verbosity, you will no longer have access to useful
diagnostics in the ``history`` attribute.

.. code:: python

   # skorch version 0.10 or later:
   net = NeuralNet(..., callbacks='disable')
   net.fit(X, y)
   print(net.history)  # no longer contains useful diagnostics

   # skorch version 0.9 or earlier
   net = NeuralNet(...)
   net.initialize()
   net.callbacks_ = []  # manually remove all callbacks
   net.fit(X, y)

Instead of turning off all callbacks, you can also turn off specific callbacks,
including default callbacks. This way, you can decide which ones to keep and
which ones to get rid of. Typically, callbacks that calculate some kind of
metric tend to be slow.

.. code:: python

   # deactivate callbacks that determine train and valid loss after each epoch
   net = NeuralNet(..., callbacks__train_loss=None, callbacks__valid_loss=None)
   net.fit(X, y)
   print(net.history)  # no longer contains 'train_loss' and 'valid_loss' entries

Prepare the Dataset
-------------------

skorch can deal with a number of different input data types. This is very
convenient, as it removes the necessity for the user to deal with them, but it
also adds a small overhead. Therefore, if you can prepare your data so that it's
already contained in an appropriate :class:`torch.utils.data.Dataset`, this
check can be skipped.

.. code:: python

   X, y = ...  # let's assume that X and y are numpy arrays
   net = NeuralNet(...)

   # normal way: let skorch figure out how to create the Dataset
   net.fit(X, y)

   # faster way: prepare Dataset yourself
   from torch.utils.data import TensorDataset
   Xt = torch.from_numpy(X)
   yt = torch.from_numpy(y)
   tensor_ds = TensorDataset(Xt, yt)
   net.fit(tensor_ds, None)

Still too slow
--------------

You find your skorch code still to be slow despite trying all of these tips, and
you made sure that the slowdown is indeed caused by skorch. What can you do now?
In this case, please search our `issue tracker
<https://github.com/skorch-dev/skorch/issues>`_ for solutions or open a new
issue. Provide as much context as possible and, if available, a minimal code
example. We will try to help you figure out what the problem is.
