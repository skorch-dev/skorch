===========
Parallelism
===========

Skorch supports distributing work among a cluster of workers via
`dask.distributed <http://distributed.readthedocs.io>`_.  In this
section we'll describe how to use Dask to efficiently distribute a
grid search or a randomized search on hyperparamerers across multiple
GPUs and potentially multiple hosts.

Let's assume that you have two GPUs that you want to run a hyper
parameter search on.

The key here is using the CUDA environment variable
`CUDA_VISIBLE_DEVICES
<https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/>`_
to limit which devices are visible to our CUDA application.  We'll set
up Dask workers that, using this environment variable, each see one
GPU only.  On the PyTorch side, we'll have to make sure to set the
device to ``cuda`` when we initialize the :class:`.NeuralNet` class.

Let's run through the steps.  First, install Dask and dask.distrubted::

  pip install dask distributed

Next, assuming you have two GPUs on your machine, let's start up a
Dask scheduler and two Dask workers.  Make sure the Dask workers are
started up in the right environment, that is, with access to all
packages required to do the work::

  dask-scheduler
  CUDA_VISIBLE_DEVICES=0 dask-worker 127.0.0.1:8786 --nthreads 1
  CUDA_VISIBLE_DEVICES=1 dask-worker 127.0.0.1:8786 --nthreads 1

In your code, use joblib's :func:`~joblib.parallel_backend` context
manager to activate the Dask backend when you run grid searches and
the like.  Also instantiate a :class:`dask.distributed.Client` to
point to the Dask scheduler that you want to use.  Let's see how this
could look like:

.. code:: python

    from dask.distributed import Client
    from joblib import parallel_backend

    client = Client('127.0.0.1:8786')

    X, y = load_my_data()
    net = get_that_net()

    gs = GridSearchCV(
        net,
        param_grid={'lr': [0.01, 0.03]},
        scoring='accuracy',
        )
    with parallel_backend('dask'):
        gs.fit(X, y)
    print(gs.cv_results_)

You can also use `Palladium <http://palladium.readthedocs.io>`_ to do
the job.  An example is included in the source in the
``examples/rnn_classifier`` folder.  Change in there and run the
following command, after having set up your Dask workers::

  PALLADIUM_CONFIG=palladium-config.py,dask-config.py pld-grid-search
