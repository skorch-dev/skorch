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

Installation
^^^^^^^^^^^^

To use this functionaliy, you need some further libraries that are not
part of skorch, namely fire_ and numpydoc_. You can install them
thusly:


.. code:: bash

    pip install fire numpydoc

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

Restrictions
^^^^^^^^^^^^

Almost all arguments should work out of the box. Therefore, you get
command line arguments for the number of epochs, learning rate, batch
size, etc. for free. Morevoer, you can access the module paremeters
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


.. _fire: https://github.com/google/python-fire
.. _numpydoc: https://github.com/numpy/numpydoc
