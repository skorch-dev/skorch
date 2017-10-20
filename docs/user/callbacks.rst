=========
Callbacks
=========

Callbacks provide a flexible way to customize the behavior of your
:class:`NeuralNet <skorch.net.NeuralNet>` training without the need to
write subclasses.

You will often find callbacks writing to or reading from the
:ref:`history <history>` attribute. Therefore, if you would like to
log the net's behavior or do something based on the past behavior,
consider using ``net.history``.

This page will not explain all existing callbacks. For that, please
look at :mod:`skorch.callbacks`.

Callback base class
-------------------

The base class for each callback is :class:`skorch.callbacks.Callback`. If
you would like to write your own callbacks, you should inherit from
this class. A guide and practical example on how to write your own
callbacks is shown in this `notebook
<https://nbviewer.jupyter.org/github/dnouri/skorch/blob/master/notebooks/Advanced_Usage.ipynb#Writing-a-custom-callback>`_.

Scoring
-------

This is a useful callback for when the default scores determined by
the ``NeuralNet`` are not enough. It allows you to easily add new
metrics to be logged during training. For an example of how to add a
new score to your model, look `at this notebook
<https://nbviewer.jupyter.org/github/dnouri/skorch/blob/master/notebooks/Basic_Usage.ipynb#Callbacks>`_.

The first argument to :class:`Scoring <skorch.callbacks.Scoring>` is
``name`` and should be a string. This determines the column name of
the score shown by the :class:`PrintLog <skorch.callbacks.PrintLog>`
after each epoch.

Next comes the ``scoring`` parameter. For eager sklearn users,
this should be familiar, since it works exactly the same as in
sklearn\'s ``GridSearchCV``, ``RandomizedSearchCV``,
``cross_val_score``, etc. For those who are unfamiliar, here is a
short explanation:

- If you pass a string, sklearn makes a look-up for a score with
  that name. Examples would be ``'f1'`` and ``'roc_auc'``.
- If you pass ``None``, the model's ``score`` method is used. By
  default, ``NeuralNet`` and its subclasses don't provide a ``score``
  method, but you can easily implement your own. If you do, it should
  take ``X`` and ``y`` (the target) as input and return a scalar as
  output.
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

Finally, you may have to provide your own ``target_extractor`` or
``pred_extractor``. This should be functions or callables that are
applied to the target or prediction before they are passed to the
scoring function. The main reason why we need this is that the
prediction you get from the PyTorch module is typically a
``torch.Tensor``, whereas the scoring functions from sklearn
expect ``numpy.ndarray``\s. This is why, by default, predictions are
cast to ``numpy.ndarray``\s.


Checkpoint
----------

Creates a checkpoint of your model parameters after each epoch if your
valid loss improved.

To change where your model is saved, change the ``target``
argument. To change under what circumstances your model is saved,
change the ``monitor`` argument. The latter can take 3 types of
arguments:

- ``None``: The model is saved after each epoch
- string: The model checks whether the last entry in the model
  ``history`` for that key is truthy. This is useful in conjunction
  with scores determined by a ``Scoring`` callback. They write a
  ``<score>_best`` entry to the ``history``, which can be used for
  checkpointing. By default, the ``Checkpoint`` callback looks at
  ``'valid_loss_best'``.
- function or callable: In that case, the function should take the
  ``NeuralNet`` instance as sole input and return a bool as output.
