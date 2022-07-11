======
Helper
======

This module provides helper functions and classes for the user. They
make working with skorch easier but are not used by skorch itself.


SliceDict
---------

A :class:`.SliceDict` is a wrapper for Python dictionaries that makes
them behave a little bit like :class:`numpy.ndarray`\s. That way, you
can slice your dictionary across values, ``len()`` will show the
length of the arrays and not the number of keys, and you get a
``shape`` attribute.  This is useful because if your data is in a
``dict``, you would normally not be able to use sklearn
:class:`~sklearn.model_selection.GridSearchCV` and similar things;
with :class:`.SliceDict`, this works.


.. _slicedataset:

SliceDataset
------------

A :class:`.SliceDataset` is a wrapper for
PyTorch :class:`~torch.utils.data.Dataset`\s that makes them behave a little
bit like :class:`numpy.ndarray`\s. That way, you can slice your
dataset with lists and arrays, and you get a ``shape`` attribute.
These properties are useful because if your data is in a dataset, you
would normally not be able to use sklearn
:class:`~sklearn.model_selection.GridSearchCV` and similar things;
with :class:`.SliceDataset`, this works.

Note that :class:`.SliceDataset` can only ever return one of the
values returned by the dataset. Typically, this will be either the X
or the y value. Therefore, if you want to wrap both X and y, you
should create two instances of :class:`.SliceDataset`, one for X (with
argument ``idx=0``, the default) and one for y (with argument
``idx=1``):

.. code:: python

    ds = MyCustomDataset()
    X_sl = SliceDataset(ds, idx=0)  # idx=0 is the default
    y_sl = SliceDataset(ds, idx=1)
    gs.fit(X_sl, y_sl)


AccelerateMixin
---------------

This mixin class can be used to add support for huggingface accelerate_ to
skorch. E.g., this allows you to use mixed precision training (AMP), multi-GPU
training, or training with a TPU. For the time being, this feature should be
considered experimental.

To use this feature, create a new subclass of the neural net class you want to
use and inherit from the mixin class. E.g., if you want to use a
:class:`.NeuralNet`, it would look like this:

.. code:: python

    from skorch import NeuralNet
    from skorch.helper import AccelerateMixin

    class AcceleratedNet(AccelerateMixin, NeuralNet):
        """NeuralNet with accelerate support"""

The same would work for :class:`.NeuralNetClassifier`,
:class:`.NeuralNetRegressor`, etc. Then pass an instance of Accelerator_ with
the desired parameters and you're good to go:

.. code:: python

    from accelerate import Accelerator

    accelerator = Accelerator(...)
    net = AcceleratedNet(
        MyModule,
        accelerator=accelerator,
    )
    net.fit(X, y)

accelerate_ recommends to leave the device handling to the Accelerator_, which
is why ``device`` defautls to ``None`` (thus telling skorch not to change the
device).

To install accelerate_, run the following command inside your Python environment:

.. code:: bash

      python -m pip install accelerate

.. note::

    Under the hood, accelerate uses :class:`~torch.cuda.amp.GradScaler`,
    which does not support passing the training step as a closure.
    Therefore, if your optimizer requires that (e.g.
    :class:`torch.optim.LBFGS`), you cannot use accelerate.

Command line interface helpers
------------------------------

Often you want to wrap up your experiments by writing a small script
that allows others to reproduce your work. With the help of skorch and
the fire_ library, it becomes very easy to write command line
interfaces without boilerplate. All arguments pertaining to skorch or
its PyTorch module are immediately available as command line
arguments, without the need to write a custom parser. If docstrings in
the numpydoc_ specification are available, there is also an
comprehensive help for the user. Overall, this allows you to make your
work reproducible without the usual hassle.

There is an example_ in the skorch repository that shows how to use
the CLI tools. Below is a snippet that shows the output created by the
help function without writing a single line of argument parsing:

.. code:: bash

    $ python examples/cli/train.py pipeline --help

    <SelectKBest> options:
       --select__score_func : callable
         Function taking two arrays X and y, and returning a pair of arrays
         (scores, pvalues) or a single array with scores.
         Default is f_classif (see below "See also"). The default function only
         works with classification tasks.
       --select__k : int or "all", optional, default=10
         Number of top features to select.
         The "all" option bypasses selection, for use in a parameter search.

    ...

    <NeuralNetClassifier> options:
       --net__module : torch module (class or instance)
         A PyTorch :class:`~torch.nn.Module`. In general, the
         uninstantiated class should be passed, although instantiated
         modules will also work.
       --net__criterion : torch criterion (class, default=torch.nn.NLLLoss)
         Negative log likelihood loss. Note that the module should return
         probabilities, the log is applied during ``get_loss``.
       --net__optimizer : torch optim (class, default=torch.optim.SGD)
         The uninitialized optimizer (update rule) used to optimize the
         module
       --net__lr : float (default=0.01)
         Learning rate passed to the optimizer. You may use ``lr`` instead
         of using ``optimizer__lr``, which would result in the same outcome.
       --net__max_epochs : int (default=10)
         The number of epochs to train for each ``fit`` call. Note that you
         may keyboard-interrupt training at any time.
       --net__batch_size : int (default=128)
         ...
       --net__verbose : int (default=1)
         Control the verbosity level.
       --net__device : str, torch.device (default='cpu')
         The compute device to be used. If set to 'cuda', data in torch
         tensors will be pushed to cuda tensors before being sent to the
         module.

    <MLPClassifier> options:
       --net__module__hidden_units : int (default=10)
         Number of units in hidden layers.
       --net__module__num_hidden : int (default=1)
         Number of hidden layers.
       --net__module__nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
         Non-linearity to apply after hidden layers.
       --net__module__dropout : float (default=0)
         Dropout rate. Dropout is applied between layers.

Installation
^^^^^^^^^^^^

To use this functionality, you need some further libraries that are not
part of skorch, namely fire_ and numpydoc_. You can install them
thusly:


.. code:: bash

    python -m pip install fire numpydoc

Usage
^^^^^

When you write your own script, only the following bits need to be
added:

.. code:: python

    import fire
    from skorch.helper import parse_args

    # your model definition and data fetching code below
    ...

    def main(**kwargs):
        X, y = get_data()
        my_model = get_model()

        # important: wrap the model with the parsed arguments
        parsed = parse_args(kwargs)
        my_model = parsed(my_model)

        my_model.fit(X, y)


    if __name__ == '__main__':
        fire.Fire(main)


This even works if your neural net is part of an sklearn pipeline, in
which case the help extends to all other estimators of your pipeline.

In case you would like to change some defaults for the net (e.g. using
a ``batch_size`` of 256 instead of 128), this is also possible. You
should have a dictionary containing your new defaults and pass it as
an additional argument to ``parse_args``:

.. code:: python

    my_defaults = {'batch_size': 128, 'module__hidden_units': 30}

    def main(**kwargs):
        ...
        parsed = parse_args(kwargs, defaults=my_defaults)
        my_model = parsed(my_model)


This will update the displayed help to your new defaults, as well as
set the parameters on the net or pipeline for you. However, the
arguments passed via the commandline have precedence. Thus, if you
additionally pass ``--batch_size 512`` to the script, batch size will
be 512.

Restrictions
^^^^^^^^^^^^

Almost all arguments should work out of the box. Therefore, you get
command line arguments for the number of epochs, learning rate, batch
size, etc. for free. Moreover, you can access the module parameters
with the double-underscore notation as usual with skorch
(e.g. ``--module__num_units 100``). This should cover almost all
common cases.

Parsing command line arguments that are non-primitive Python objects
is more difficult, though. skorch's custom parsing should support
normal Python types and simple custom objects, e.g. this works:
``--module__nonlin 'torch.nn.RReLU(0.1, upper=0.4)'``. More complex
parsing might not work. E.g., it is currently not possible to add new
callbacks through the command line (but you can modify existing ones
as usual).


.. _accelerate: https://github.com/huggingface/accelerate
.. _Accelerator: https://huggingface.co/docs/accelerate/accelerator.html
.. _fire: https://github.com/google/python-fire
.. _numpydoc: https://github.com/numpy/numpydoc
.. _example: https://github.com/skorch-dev/skorch/tree/master/examples/cli
