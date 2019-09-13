=========
Callbacks
=========

Callbacks provide a flexible way to customize the behavior of your
:class:`.NeuralNet` training without the need to write subclasses.

You will often find callbacks writing to or reading from the
:ref:`history <history>` attribute. Therefore, if you would like to
log the net's behavior or do something based on the past behavior,
consider using ``net.history``.

This page will not explain all existing callbacks. For that, please
look at :mod:`skorch.callbacks`.

Callback base class
-------------------

The base class for each callback is :class:`.Callback`. If you would
like to write your own callbacks, you should inherit from this class.
A guide and practical example on how to write your own callbacks is
shown in this `notebook
<https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Writing-a-custom-callback>`_.
In general, remember this:


* They should inherit from the base class.
* They should implement at least one of the :code:`on_` methods
  provided by the parent class (see below).
* As argument, the methods first get the :class:`.NeuralNet` instance,
  and, where appropriate, the local data (e.g. the data from the
  current batch). The method should also have :code:`**kwargs` in the
  signature for potentially unused arguments.

Callback methods to override
----------------------------

The following methods could potentially be overriden when implementing
your own callbacks.

initialize()
^^^^^^^^^^^^

If you have attributes that should be reset when the model is
re-initialized, those attributes should be set in this method.

on_train_begin(net, X, y)
^^^^^^^^^^^^^^^^^^^^^^^^^

Called once at the start of the training process (e.g. when calling
fit).

on_train_end(net, X, y)
^^^^^^^^^^^^^^^^^^^^^^^

Called once at the end of the training process.

on_epoch_begin(net, dataset_train, dataset_valid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Called once at the start of the epoch, i.e. possibly several times per
fit call. Gets training and validation data as additional input.

on_epoch_end(net, dataset_train, dataset_valid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Called once at the end of the epoch, i.e. possibly several times per
fit call. Gets training and validation data as additional input.

on_batch_begin(net, Xi, yi, training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Called once before each batch of data is processed, i.e. possibly
several times per epoch. Gets batch data as additional input.
Also includes a bool indicating if this is a training batch or not.


on_batch_end(net, Xi, yi, training, loss, y_pred)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Called once after each batch of data is processed, i.e. possibly
several times per epoch. Gets batch data as additional input.

on_grad_computed(net, named_parameters, Xi, yi)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Called once per batch after gradients have been computed but before an
update step was performed. Gets the module parameters as additional
input as well as the batch data. Useful if you want to tinker with
gradients.


Setting callback parameters
---------------------------

You can set specific callback parameters using the ususal `set_params`
interface on the network by using the `callbacks__` prefix and the
callback's name. For example to change the scoring order of the train
loss you can write this:

.. code:: python

    net = NeuralNet()
    net.set_params(callbacks__train_loss__lower_is_better=False)

Changes will be applied on initialization and callbacks that
are changed using `set_params` will be re-initialized.

The name you use to address the callback can be chosen during
initialization of the network and defaults to the class name.
If there is a conflict, the conflicting names will be made unique
by appending a count suffix starting at 1, e.g.
``EpochScoring_1``, ``EpochScoring_2``, etc.


Deactivating callbacks
-----------------------

If you would like to (temporarily) deactivate a callback, you can do
so by setting its parameter to None. E.g., if you have a callback
called 'my_callback', you can deactivate it like this:

.. code:: python

    net = NeuralNet(
        module=MyModule,
            callbacks=[('my_callback', MyCallback())],
    )
    # now deactivate 'my_callback':
    net.set_params(callbacks__my_callback=None)

This also works with default callbacks.

Deactivating callbacks can be especially useful when you do a
parameter search (say with sklearn
:class:`~sklearn.model_selection.GridSearchCV`). If, for instance, you
use a callback for learning rate scheduling (e.g. via
:class:`.LRScheduler`) and want to test its usefulness, you can
compare the performance once with and once without the callback.


Scoring
-------

skorch provides two scoring callbacks by default,
:class:`.EpochScoring` and :class:`.BatchScoring`. They work basically
in the same way, except that :class:`.EpochScoring` calculates scores
after each epoch and :class:`.BatchScoring` after each batch. Use the
former if averaging of batch-wise scores is imprecise (say for AUC
score) and the latter if you are very tight for memory.

In general, the scoring callbacks are useful when the default scores
determined by the :class:`.NeuralNet` are not enough. They allow you
to easily add new metrics to be logged during training. For an example
of how to add a new score to your model, look `at this notebook
<https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Basic_Usage.ipynb#Callbacks>`_.

The first argument to both callbacks is ``name`` and should be a
string. This determines the column name of the score shown by the
:class:`.PrintLog` after each epoch.

Next comes the ``scoring`` parameter. For eager sklearn users, this
should be familiar, since it works exactly the same as in sklearn
:class:`~sklearn.model_selection.GridSearchCV`,
:class:`~sklearn.model_selection.RandomizedSearchCV`,
:func:`~sklearn.model_selection.cross_val_score`, etc. For those who
are unfamiliar, here is a short explanation:

- If you pass a string, sklearn makes a look-up for a score with
  that name. Examples would be ``'f1'`` and ``'roc_auc'``.
- If you pass ``None``, the model's ``score`` method is used. By
  default, :class:`.NeuralNet` and its subclasses don't provide a
  ``score`` method, but you can easily implement your own. If you do,
  it should take ``X`` and ``y`` (the target) as input and return a
  scalar as output.
- Finally, you can pass a function/callable. In that case, this
  function should have the signature ``func(net, X, y)`` and return a
  scalar.

More on sklearn\'s model evaluation can be found `in this notebook
<http://scikit-learn.org/stable/modules/model_evaluation.html>`_.

The ``lower_is_better`` parameter determines whether lower scores
should be considered as better (e.g. log loss) or worse
(e.g. accuracy). This information is used to write a ``<name>_best``
value to the net's ``history``. E.g., if your score is f1 score and is
called ``'f1'``, you should set ``lower_is_better=False``. The
``history`` will then contain an entry for ``'f1'``, which is the
score itself, and an entry for ``'f1_best'``, which says whether this
is the as of yet best f1 score.

``on_train`` is used to indicate whether training or validation data
should be used to determine the score. By default, it is set to
validation.

Finally, you may have to provide your own ``target_extractor``. This
should be a function or callable that is applied to the target before
it is passed to the scoring function. The main reason why we need this
is that sometimes, the target is not of a form expected by sklearn and
we need to process it before passing it on.


Checkpoint
----------

The :class:`.Checkpoint` callback creates a checkpoint of your model
after each epoch that met certain criteria. By default, the condition
is that the validation loss has improved, however you may change this
by specifying the ``monitor`` parameter. It can take three types of
arguments:

- ``None``: The model is saved after each epoch;
- string: The model checks whether the last entry in the model
  ``history`` for that key is truthy. This is useful in conjunction
  with scores determined by a scoring callback. They write a
  ``<score>_best`` entry to the ``history``, which can be used for
  checkpointing. By default, the :class:`.Checkpoint` callback looks
  at ``'valid_loss_best'``;
- function or callable: In that case, the function should take the
  :class:`.NeuralNet` instance as sole input and return a bool as
  output.

To specify where and how your model is saved, change the arguments
starting with ``f_``:

- ``f_params``: to save model parameters
- ``f_optimizer``: to save optimizer state
- ``f_history``: to save training history
- ``f_pickle``: to pickle the entire model object.

Please refer to :ref:`saving and loading` for more information about
restoring your network from a checkpoint.


Learning rate schedulers
------------------------

The :class:`.LRScheduler` callback allows the use of the various
learning rate schedulers defined in :mod:`torch.optim.lr_scheduler`,
such as :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`, which
allows dynamic learning rate reducing based on a given value to
monitor, or :class:`~torch.optim.lr_scheduler.CyclicLR`, which cycles
the learning rate between two boundaries with a constant frequency.

Here's a network that uses a callback to set a cyclic learning rate:

.. code:: python

    from skorch.callbacks import LRScheduler
    from torch.optim.lr_scheduler import CyclicLR
          
    net = NeuralNet(
        module=MyModule,
        callbacks=[
            ('lr_scheduler',
             LRScheduler(policy=CyclicLR,
                         base_lr=0.001,
                         max_lr=0.01)),
        ],
    )

As with other callbacks, you can use `set_params` to set parameters,
and thus search learning rate scheduler parameters using
:class:`~sklearn.model_selection.GridSearchCV` or similar.  An
example:

.. code:: python

    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(
        net,
        param_grid={'callbacks__lr_scheduler__max_lr': [0.01, 0.1, 1.0]},
    )
