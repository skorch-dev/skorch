skorch documentation
====================

A scikit-learn compatible neural network library that wraps PyTorch.


Introduction
------------

The goal of skorch is to make it possible to use PyTorch_ with
sklearn_. This is achieved by providing a wrapper around
PyTorch that has an sklearn interface.

skorch does not re-invent the wheel, instead getting as much out
of your way as possible. If you are familiar with sklearn and
PyTorch, you don't have to learn any new concepts, and the syntax
should be well known. (If you're not familiar with those libraries, it
is worth getting familiarized.)

Additionally, skorch abstracts away the training loop, making a
lot of boilerplate code obsolete. A simple ``net.fit(X, y)`` is
enough. Out of the box, skorch works with many types of data, be
it PyTorch Tensors, NumPy arrays, Python dicts, and so
on. However, if you have other data, extending skorch is easy to
allow for that.

Overall, skorch aims at being as flexible as PyTorch while
having a clean interface as sklearn.

If you use skorch, please use this BibTeX entry:

.. code:: bibtex

   @manual{skorch,
     author       = {Marian Tietz and Thomas J. Fan and Daniel Nouri and Benjamin Bossan and {skorch Developers}},
     title        = {skorch: A scikit-learn compatible neural network library that wraps PyTorch},
     month        = jul,
     year         = 2017,
     url          = {https://skorch.readthedocs.io/en/stable/}
   }


User's Guide
------------
.. toctree::
   :maxdepth: 2

   user/installation
   user/quickstart
   user/tutorials
   user/neuralnet
   user/callbacks
   user/dataset
   user/save_load
   user/probabilistic
   user/history
   user/toy
   user/helper
   user/REST
   user/parallelism
   user/customization
   user/performance
   user/FAQ


API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  skorch API <skorch>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _pytorch: https://pytorch.org/
.. _sklearn: https://scikit-learn.org/
