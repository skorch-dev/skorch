=======
History
=======

A :class:`.NeuralNet` object logs training progress internally using a
:class:`.History` object, stored in the ``history`` attribute.  Among
other use cases, ``history`` is used to print the training progress
after each epoch:

.. code::

    net.fit(X, y)

    # prints
      epoch    train_loss    valid_acc    valid_loss     dur
    -------  ------------  -----------  ------------  ------
          1        0.7111       0.5100        0.6894  0.1345
          2        0.6928       0.5500        0.6803  0.0608
          3        0.6833       0.5650        0.6741  0.0620
          4        0.6763       0.5850        0.6674  0.0594

All this information (and more) is stored in and can be accessed
through ``net.history``. It is thus best practice to make use of
``history`` for storing training-related data.

In general, :class:`.History` works like a list of dictionaries, where
each item in the list corresponds to one epoch, and each key of the
dictionary to one column. Thus, if you would like to access the
``'train_loss'`` of the last epoch, you can call
``net.history[-1]['train_loss']``. To make the ``history`` more
accessible, though, it is possible to just pass the indices separated
by a comma: ``net.history[-1, 'train_loss']``.

Moreover, :class:`.History` stores the results from each individual
batch under the ``batches`` key during each epoch. So to get the train
loss of the 3rd batch of the 7th epoch, use ``net.history[7,
'batches', 3, 'train_loss']``.

Here are some examples showing how to index ``history``:

.. code:: python

    # history of a fitted neural net
    history = net.history
    # get current epoch, a dict
    history[-1]
    # get train losses from all epochs, a list of floats
    history[:, 'train_loss']
    # get train and valid losses from all epochs, a list of tuples
    history[:, ('train_loss', 'valid_loss')]
    # get current batches, a list of dicts
    history[-1, 'batches']
    # get latest batch, a dict
    history[-1, 'batches', -1]
    # get train losses from current batch, a list of floats
    history[-1, 'batches', :, 'train_loss']
    # get train and valid losses from current batch, a list of tuples
    history[-1, 'batches', :, ('train_loss', 'valid_loss')]

As :class:`.History` essentially is a list of dictionaries, you can
also write to it as if it were a list of dictionaries. Here too,
skorch provides some convenience functions to make life easier. First
there is :func:`~skorch.history.History.new_epoch`, which will add a
new epoch dictionary to the end of the list. Also, there is
:func:`~skorch.history.History.new_batch` for adding new batches to
the current epoch.

To add a new item to the current epoch, use ``history.record('foo',
123)``. This will set the value ``123`` for the key ``foo`` of the
current epoch. To write a value to the current batch, use
``history.record_batch('bar', 456)``. Below are some more examples:

.. code:: python

    # history of a fitted neural net
    history = net.history
    # add new epoch row
    history.new_epoch()
    # add an entry to current epoch
    history.record('my-score', 123)
    # add a batch row to the current epoch
    history.new_batch()
    # add an entry to the current batch
    history.record_batch('my-batch-score', 456)
    # overwrite entry of current batch
    history.record_batch('my-batch-score', 789)

Distributed history
-------------------

.. _dist-history:

When training a net in a distributed setting, e.g. when using
:class:`torch.nn.parallel.DistributedDataParallel`, directly or indirectly with
the help of :class:`.AccelerateMixin`, the default history class should not be
used. This is because each process will have its own history instance with no
syncing happening between processes. Therefore, the information in the histories
can diverge. When steering the training process through the histories, the
resulting differences can cause trouble. When using early stopping, for
instance, one process could receive the signal to stop but not the other.

To avoid this, use the :class:`.DistributedHistory` class provided by skorch. It
will take care of syncing the distributed batch information across processes,
which will prevent the issue just described.

This class needs to be initialized with a `distributed store provided by PyTorch
<https://pytorch.org/docs/stable/distributed.html#distributed-key-value-store>`_.
We have only tested :class:`torch.distributed.TCPStore` so far, so if unsure,
use that one, though :class:`torch.distributed.FileStore` should also work. The
:class:`.DistributedHistory` also needs to be initialized with its rank and the
world size (number of processes) so that it has all the required information to
perform the syncing. When using ``accelerate``, that information can be
retrieved from the ``Accelerator`` instance.

A typical training script without ``accelerate`` may contain a function like
this:

.. code:: python

    from torch.distributed import TCPStore
    from torch.nn.parallel import DistributedDataParallel

    def train(rank, world_size, is_master):
        store = TCPStore(
            "127.0.0.1", port=1234, world_size=world_size)
        dist_history = DistributedHistory(
            store=store, rank=rank, world_size=world_size)
        net = NeuralNetClassifier(..., history=dist_history)
        net.fit(X, y)

When using :class:`.AccelerateMixin`, it could look like this instead:

.. code:: python

    from accelerate import Accelerator
    from skorch.hf import AccelerateMixin

    accelerator = Accelerator(...)

    def train(accelerator):
        is_master = accelerator.is_main_process
        world_size = accelerator.num_processes
        rank = accelerator.local_process_index
        store = TCPStore(
            "127.0.0.1", port=1234, world_size=world_size, is_master=is_master)
        dist_history = DistributedHistory(
            store=store, rank=rank, world_size=world_size)
        net = AcceleratedNet(..., history=dist_history)
        net.fit(X, y)

When using ``accelerate`` in a non-distributed setting (e.g. to take advantage
of mixed precision training), it is not necessary to use
:class:`.DistributedHistory`, the normal history class will do.
