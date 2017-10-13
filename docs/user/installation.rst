.. _installation:

============
Installation
============

.. contents::
   :local:


conda
~~~~~

You need a working conda installation. Get the correct miniconda for
your system from `here <https://conda.io/miniconda.html>`__.

For users
^^^^^^^^^

Note: pip installation will follow soon.

.. code:: bash

    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    python setup.py install

For developers
^^^^^^^^^^^^^^

.. code:: bash

    conda env create
    source activate skorch
    # install pytorch version for your system (see below)
    conda install --file requirements-dev.txt
    python setup.py develop

    py.test  # unit tests
    pylint skorch  # static code checks

pip
~~~

Same as for conda, but to install main requirements, run:

.. code:: bash

    pip install -r requirements.txt

pytorch
~~~~~~~

`pytorch` is not covered by the dependencies, since the pytorch
version you need is dependent on your system. For installation
instructions for pytorch, visit the `pytorch website
<http://pytorch.org/>`__.

In general, this should work:

.. code:: bash

    # using conda:
    conda install pytorch cuda80 -c soumith
    # using pip
    pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
