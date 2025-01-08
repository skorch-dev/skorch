.. image:: https://github.com/skorch-dev/skorch/blob/master/assets/skorch_bordered.svg
   :width: 30%

===========================
How to Contribute to Skorch
===========================

We ❤️ contributions from the open-source community! We follow the Contributor Convenant Code of Conduct that ensures a positive
experience for contributors and maintainers. To know more, please see `here <https://www.contributor-covenant.org/version/2/1/code_of_conduct/>`


To begin contributing, you need to clone the repository from source and install dev dependencies as shown below:


conda installation
==================

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    # create and activate a virtual environment
    python -m pip install -r requirements.txt
    # install pytorch version for your system (see below)
    python -m pip install -r requirements-dev.txt
    python -m pip install -e .

    py.test  # unit tests
    pylint skorch  # static code checks

pip installation
================

.. code:: bash

    git clone https://github.com/skorch-dev/skorch.git
    cd skorch
    conda create -n skorch-env python=3.10
    conda activate skorch-env
    conda install -c pytorch pytorch
    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python -m pip install -e .

    py.test  # unit tests
    pylint skorch  # static code checks

It is recommend to run the unit tests and static code checks before submitting a PR 
to ensure the predefined coding standards are followed.

===========
Maintainers
===========

You can tag the following maintainers to request a review for your PR:

