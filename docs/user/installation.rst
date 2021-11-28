============
Installation
============

.. contents::
   :local:


pip installation
~~~~~~~~~~~~~~~~

To install with pip, run:

.. code:: bash

    pip install -U skorch

We recommend to use a virtual environment for this.

From source
~~~~~~~~~~~

If you would like to use the must recent additions to skorch or
help development, you should install skorch from source.

Using conda
^^^^^^^^^^^

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda env create
    source activate skorch
    pip install .

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda env create
    source activate skorch
    pip install -e .

    py.test  # unit tests
    pylint skorch  # static code checks

Using pip
^^^^^^^^^

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    pip install -r requirements.txt
    # install pytorch version for your system (see below)
    pip install .

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    pip install -r requirements.txt
    # install pytorch version for your system (see below)
    pip install -r requirements-dev.txt
    pip install -e .

    py.test  # unit tests
    pylint skorch  # static code checks

PyTorch
~~~~~~~

PyTorch is not covered by the dependencies, since the PyTorch version
you need is dependent on your OS and device. For installation
instructions for PyTorch, visit the `PyTorch website
<http://pytorch.org/>`__. skorch officially supports the last four
minor PyTorch versions, which currently are:

- 1.7.1
- 1.8.1
- 1.9.1
- 1.10.0

However, that doesn't mean that older versions don't work, just that
they aren't tested. Since skorch mostly relies on the stable part of
the PyTorch API, older PyTorch versions should work fine.

In general, running this to install PyTorch should work (assuming CUDA
11.1):

.. code:: bash

    # using conda:
    conda install pytorch cudatoolkit==11.1 -c pytorch
    # using pip
    pip install torch
