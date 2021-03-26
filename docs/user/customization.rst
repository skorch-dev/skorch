=============
Customization
=============

Customizing NeuralNet
---------------------

:class:`.NeuralNet` and its subclasses like
:class:`.NeuralNetClassifier` are already very flexible as they are
and should cover many use cases by adjusting the provided
parameters. However, this may not always be sufficient for your use
cases. If you thus find yourself wanting to customize
:class:`.NeuralNet`, please follow these guidelines.

Initialization
^^^^^^^^^^^^^^

The method :func:`~skorch.net.NeuralNet.initialize` is responsible for
initializing all the components needed by the net, e.g. the module and
the optimizer. For this, it calls specific initialization methods,
such as :func:`~skorch.net.NeuralNet.initialize_module` and
:func:`~skorch.net.NeuralNet.initialize_optimizer`. If you'd like to
customize the initialization behavior, you should override the
corresponding methods. Following sklearn conventions, the created
components should be set as an attribute with a trailing underscore as
the name, e.g. ``module_`` for the initialized module. Finally, the
method should return ``self``.

Methods starting with get_*
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The net provides a few ``get_*`` methods, most notably
:func:`~skorch.net.NeuralNet.get_loss`,
:func:`~skorch.net.NeuralNet.get_dataset`, and
:func:`~skorch.net.NeuralNet.get_iterator`. The intent of these
methods should be pretty self-explanatory, and if you are still not
quite sure, consult their documentations. In general, these methods
are fairly safe to override as long as you make sure to conform to the
same signature as the original.

Training and validation
^^^^^^^^^^^^^^^^^^^^^^^

If you would like to customize training and validation, there are
several possibilities. Below are the methods that you most likely want
to customize:

The method :func:`~skorch.net.NeuralNet.train_step_single` performs a
single training step. It accepts the current batch of data as input
(as well as the ``fit_params``) and should return a dictionary
containing the ``loss`` and the prediction ``y_pred``. E.g. you should
override this if your dataset returns some non-standard data that
needs custom handling, and/or if your module has to be called in a
very specific way. If you want to, you can still make use of
:func:`~skorch.net.NeuralNet.infer` and
:func:`~skorch.net.NeuralNet.get_loss` but it's not strictly
necessary. Don't call the optimizer in this method, this is handled by
the next method.

The method :func:`~skorch.net.NeuralNet.train_step` defines the
complete training procedure performed for each batch. It accepts the
same arguments as :func:`~skorch.net.NeuralNet.train_step_single` but
it differs in that it defines the training closure passed to the
optimizer, which for instance could be called more than once (e.g. in
case of L-BFGS). You might override this if you deal with non-standard
training procedures, as e.g. gradient accumulation.

The method :func:`~skorch.net.NeuralNet.validation_step` is
responsible for calculating the prediction and loss on the validation
data (remember that skorch uses an internal validation set for
reporting, early stopping, etc.). Similar to
:func:`~skorch.net.NeuralNet.train_step_single`, it receives the batch
and ``fit_params`` as input and should return a dictionary containing
``loss`` and ``y_pred``. Most likely, when you need to customize
:func:`~skorch.net.NeuralNet.train_step_single`, you'll need to
customize :func:`~skorch.net.NeuralNet.validation_step` accordingly.

Finally, the method :func:`~skorch.net.NeuralNet.evaluation_step` is
called to you use the net for inference, e.g. when calling
:func:`~skorch.net.NeuralNet.forward` or
:func:`~skorch.net.NeuralNet.predict`. You may want to modify this if,
e.g., you want your model to behave differently during training and
during prediction.

You should also be aware that some methods are better left
untouched. E.g., in most cases, the following methods should *not* be
overridden:

* :func:`~skorch.net.NeuralNet.fit`
* :func:`~skorch.net.NeuralNet.partial_fit`
* :func:`~skorch.net.NeuralNet.fit_loop`
* :func:`~skorch.net.NeuralNet.run_single_epoch`

The reason why these methods should stay untouched is because they
perform some book keeping, like making sure that callbacks are handled
or writing logs to the ``history``. If you do need to override these,
make sure that you perform the same book keeping as the original
methods.
