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
    # install pytorch version for your system (see below)
    python setup.py install

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    conda install -c conda-forge --file requirements-dev.txt
    python setup.py develop

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
    python setup.py install

If you want to help developing, run:

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    pip install -r requirements.txt
    # install pytorch version for your system (see below)
    pip install -r requirements-dev.txt
    python setup.py develop

    py.test  # unit tests
    pylint skorch  # static code checks

pytorch
~~~~~~~

PyTorch is not covered by the dependencies, since the PyTorch
version you need is dependent on your system. For installation
instructions for PyTorch, visit the `pytorch website
<http://pytorch.org/>`__.

In general, this should work:

.. code:: bash

    # using conda:
    conda install pytorch cuda80 -c soumith
    # using pip
    pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
