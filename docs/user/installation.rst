============
Installation
============

.. contents::
   :local:


pip installation
~~~~~~~~~~~~~~~~

To install with pip, run:

.. code:: bash

    python -m pip install -U skorch

We recommend to use a virtual environment for this.

From source
~~~~~~~~~~~

If you would like to use the most recent additions to skorch or
help development, you should install skorch from source.

Using conda
^^^^^^^^^^^

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda create -n skorch-env python=3.10
    conda activate skorch-env
    # install pytorch version for your system (see below)
    python -m pip install .


If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda create -n skorch-env python==3.10
    conda activate skorch-env
    # install pytorch version for your system (see below)
    python -m pip install '.[test,docs,dev,extended]'

    py.test  # unit tests
    pylint skorch  # static code checks

You may adjust the Python version to any of the supported Python versions, i.e.
Python 3.9 or higher.

Using pip
^^^^^^^^^

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    # install pytorch version for your system (see below)
    python -m pip install .

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    # install pytorch version for your system (see below)
    python -m pip install -e '.[test,docs,dev,extended]'

    py.test  # unit tests
    pylint skorch  # static code checks

PyTorch
~~~~~~~

PyTorch is not covered by the dependencies, since the PyTorch version
you need is dependent on your OS and device. For installation
instructions for PyTorch, visit the `PyTorch website
<http://pytorch.org/>`__. skorch officially supports the last four
minor PyTorch versions, which currently are:

- 2.5.1
- 2.6.0
- 2.7.1
- 2.8.0

However, that doesn't mean that older versions don't work, just that
they aren't tested. Since skorch mostly relies on the stable part of
the PyTorch API, older PyTorch versions should work fine.

In general, running this to install PyTorch should work:

.. code:: bash

    python -m pip install torch
