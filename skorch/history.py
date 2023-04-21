"""Contains history class and helper functions."""

import json
import pickle

from skorch.utils import open_file_like


# pylint: disable=invalid-name
class _none:
    """Special placeholder since ``None`` is a valid value."""


def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)


def _getitem_list_list(items, keys, tuple_=False):
    """Ugly but efficient extraction of multiple values from a list of
    items.
    """
    filtered = []
    for item in items:
        row = []
        for key in keys:
            try:
                row.append(item[key])
            except KeyError:
                break
        else:  # no break
            if row:
                filtered.append(tuple(row) if tuple_ else row)
    if items and not filtered:
        return _none
    return filtered


def _getitem_list_tuple(items, keys):
    return _getitem_list_list(items, keys, tuple_=True)


def _getitem_list_str(items, key):
    filtered = []
    for item in items:
        try:
            filtered.append(item[key])
        except KeyError:
            continue
    if items and not filtered:
        return _none
    return filtered


def _getitem_dict_list(item, keys):
    return [item.get(key, _none) for key in keys]


def _getitem_dict_tuple(item, keys):
    return tuple(item.get(key, _none) for key in keys)


def _getitem_dict_str(item, key):
    return item.get(key, _none)


def _get_getitem_method(items, key):
    """Return method to extract values from items.

    For the given type of items and type of keys, find the correct
    method to extract the values. By calling this only once per items,
    we can save a lot of type checking, which can be slow if there are
    a lot of epochs and a lot of items. However, we now make the
    assumption that the type of items doesn't change (we know that the
    key doesn't change). This should always be true, except if
    something really weird happens.

    We are multi-dispatching based on the following possibilities:

    * history[0, 'foo', :10]: get a list of items
    * history[0, 'foo', 0]: get a dict
    * history[0, 'foo', :, 'bar']: key is a str
    * history[0, 'foo', :, ('bar', 'baz')]: key is list/tuple of str

    """
    if isinstance(items, list):
        if isinstance(key, list):
            return _getitem_list_list
        if isinstance(key, tuple):
            return _getitem_list_tuple
        if isinstance(key, str):
            return _getitem_list_str
        raise TypeError("History access with given types not supported")

    if isinstance(items, dict):
        if isinstance(key, list):
            return _getitem_dict_list
        if isinstance(key, tuple):
            return _getitem_dict_tuple
        if isinstance(key, str):
            return _getitem_dict_str
    raise TypeError("History access with given types not supported")


def _unpack_index(i):
    """Unpack index and return exactly four elements.

    If index is more shallow than 4, return None for trailing
    dimensions. If index is deeper than 4, raise a KeyError.

    """
    if len(i) > 4:
        raise KeyError(
            "Tried to index history with {} indices but only "
            "4 indices are possible.".format(len(i)))

    # fill trailing indices with None
    i_e, k_e, i_b, k_b = i + tuple([None] * (4 - len(i)))

    return i_e, k_e, i_b, k_b


class History(list):
    """History contains the information about the training history of
    a :class:`.NeuralNet`, facilitating some of the more common tasks
    that are occur during training.

    When you want to log certain information during training (say, a
    particular score or the norm of the gradients), you should write
    them to the net's history object.

    It is basically a list of dicts for each epoch, that, again,
    contains a list of dicts for each batch. For convenience, it has
    enhanced slicing notation and some methods to write new items.

    To access items from history, you may pass a tuple of up to four
    items:

      1. Slices along the epochs.
      2. Selects columns from history epochs, may be a single one or a
         tuple of column names.
      3. Slices along the batches.
      4. Selects columns from history batchs, may be a single one or a
         tuple of column names.

    You may use a combination of the four items.

    If you select columns that are not present in all epochs/batches,
    only those epochs/batches are chosen that contain said columns. If
    this set is empty, a ``KeyError`` is raised.

    Examples
    --------
    >>> # ACCESSING ITEMS
    >>> # history of a fitted neural net
    >>> history = net.history
    >>> # get current epoch, a dict
    >>> history[-1]
    >>> # get train losses from all epochs, a list of floats
    >>> history[:, 'train_loss']
    >>> # get train and valid losses from all epochs, a list of tuples
    >>> history[:, ('train_loss', 'valid_loss')]
    >>> # get current batches, a list of dicts
    >>> history[-1, 'batches']
    >>> # get latest batch, a dict
    >>> history[-1, 'batches', -1]
    >>> # get train losses from current batch, a list of floats
    >>> history[-1, 'batches', :, 'train_loss']
    >>> # get train and valid losses from current batch, a list of tuples
    >>> history[-1, 'batches', :, ('train_loss', 'valid_loss')]

    >>> # WRITING ITEMS
    >>> # add new epoch row
    >>> history.new_epoch()
    >>> # add an entry to current epoch
    >>> history.record('my-score', 123)
    >>> # add a batch row to the current epoch
    >>> history.new_batch()
    >>> # add an entry to the current batch
    >>> history.record_batch('my-batch-score', 456)
    >>> # overwrite entry of current batch
    >>> history.record_batch('my-batch-score', 789)

    """

    def new_epoch(self):
        """Register a new epoch row."""
        self.append({'batches': []})

    def new_batch(self):
        """Register a new batch row for the current epoch."""
        # pylint: disable=invalid-sequence-index
        self[-1]['batches'].append({})

    def record(self, attr, value):
        """Add a new value to the given column for the current
        epoch.

        """
        msg = "Call new_epoch before recording for the first time."
        if not self:
            raise ValueError(msg)
        self[-1][attr] = value

    def record_batch(self, attr, value):
        """Add a new value to the given column for the current
        batch.

        """
        # pylint: disable=invalid-sequence-index
        self[-1]['batches'][-1][attr] = value

    def to_list(self):
        """Return history object as a list."""
        return list(self)

    @classmethod
    def from_file(cls, f):
        """Load the history of a ``NeuralNet`` from a json file.

        Parameters
        ----------
        f : file-like object or str

        """

        with open_file_like(f, 'r') as fp:
            return cls(json.load(fp))

    def to_file(self, f):
        """Saves the history as a json file. In order to use this feature,
        the history must only contain JSON encodable Python data structures.
        Numpy and PyTorch types should not be in the history.

        Parameters
        ----------
        f : file-like object or str

        """
        with open_file_like(f, 'w') as fp:
            json.dump(self.to_list(), fp)

    def __getitem__(self, i):
        # This implementation resolves indexing backwards,
        # i.e. starting from the batches, then progressing to the
        # epochs.
        if isinstance(i, (int, slice)):
            i = (i,)

        # i_e: index epoch, k_e: key epoch
        # i_b: index batch, k_b: key batch
        i_e, k_e, i_b, k_b = _unpack_index(i)
        keyerror_msg = "Key {!r} was not found in history."

        if i_b is not None and k_e != 'batches':
            raise KeyError("History indexing beyond the 2nd level is "
                           "only possible if key 'batches' is used, "
                           "found key {!r}.".format(k_e))

        items = self.to_list()

        # extract the epochs
        # handles: history[i_e]
        if i_e is not None:
            items = items[i_e]
            if isinstance(i_e, int):
                items = [items]

        # extract indices of batches
        # handles: history[..., k_e, i_b]
        if i_b is not None:
            items = [row[k_e][i_b] for row in items]

        # extract keys of epochs or batches
        # handles: history[..., k_e]
        # handles: history[..., ..., ..., k_b]
        if k_e is not None and (i_b is None or k_b is not None):
            key = k_e if k_b is None else k_b

            if items:
                extract = _get_getitem_method(items[0], key)
                items = [extract(item, key) for item in items]

                # filter out epochs with missing keys
                items = list(filter(_not_none, items))

            if not items and not (k_e == 'batches' and i_b is None):
                # none of the epochs matched
                raise KeyError(keyerror_msg.format(key))

            if (
                    isinstance(i_b, slice)
                    and k_b is not None
                    and not any(batches for batches in items)
            ):
                # none of the batches matched
                raise KeyError(keyerror_msg.format(key))

        if isinstance(i_e, int):
            items, = items

        return items


class DistributedHistory:
    """History for use in training using multiple processes

    When using skorch with :class:`.AccelerateMixin` for multi GPU training, use
    this class instead of the default :class:`.History` class.

    When using PyTorch :class:`torch.nn.parallel.DistributedDataParallel`, the
    whole training process is forked and batches are processed in parallel. That
    means that the standard :class:`.History` does not see all the batches that
    are being processed, which results in the different processes having
    histories that are out of sync. This is bad because the history is used as a
    reference to influence the training, e.g. to control early stopping.

    This class solves the problem by using a distributed store from PyTorch,
    e.g. :class:`torch.distributed.TCPStore`, to synchronize the batch
    information across processes. This ensures that the information stored in
    the individual history copies is identical for ``history[:, 'batches']``.
    When it comes to the epoch-level information, it can still diverge between
    processes (e.g. the recorded duration of the epoch).

    To use this class, instantiate it and pass it as the ``history`` argument to
    the net.

    Click here for more information on `PyTorch distributed key-value stores
    <https://pytorch.org/docs/stable/distributed.html#distributed-key-value-store>`_.

    Notes
    -----
    If using this class results in the processes hanging or timing out, double
    check that the ``rank`` and ``world_size`` arguments are set correctly.
    Otherwise, the history instances will be waiting for records that are
    actually never written.

    If the speed of the processes is very uneven, there can also be timeouts. To
    increase the waiting time, pass a corresponding ``timeout`` argument to the
    ``store`` instance.

    Objects stored with this history make a json roundtrip. Therefore, if you
    store objects that don't survive a json roundtrip (say, numpy arrays), don't
    use this class.

    The PyTorch ``Store`` classes cannot be pickled. Therefore, if a net using
    this history class is pickled, the ``store`` attribute is discarded so that
    the pickling does not fail. This means, however, that an unpickled net
    cannot be used for further training without manually setting the ``store``
    attribute on the ``history``.

    Examples
    --------
    >>> # general
    >>> from skorch import NeuralNetClassifier
    >>> from torch.distributed import TCPStore
    >>> from torch.nn.parallel import DistributedDataParallel
    >>> def train(rank, world_size, is_master):
    ...     store = TCPStore(
    ...         "127.0.0.1", port=1234, world_size=world_size)
    ...     dist_history = DistributedHistory(
    ...         store=store, rank=rank, world_size=world_size)
    ...     net = NeuralNetClassifier(..., history=dist_history)
    ...     net.fit(X, y)
    >>> # with accelerate
    >>> from accelerate import Accelerator
    >>> from skorch.hf import AccelerateMixin
    >>> accelerator = Accelerator(...)
    >>> def train(accelerator):
    ...     is_master = accelerator.is_main_process
    ...     world_size = accelerator.num_processes
    ...     rank = accelerator.local_process_index
    ...     store = TCPStore(
    ...         "127.0.0.1", port=1234, world_size=world_size, is_master=is_master)
    ...     dist_history = DistributedHistory(
    ...         store=store, rank=rank, world_size=world_size)
    ...     net = AcceleratedNet(..., history=dist_history)
    ...     net.fit(X, y)

    Parameters
    ----------
    store : torch.distributed.Store
      The torch distributed ``Store`` instance,
      :class:`torch.distributed.TCPStore` has been tested to work.

    rank : int
      The rank of this particular process among all processes. Each process
      should have a unique rank between 0 and ``world_size`` - 1. If using
      ``accelerate``, the rank can be determined as
      ``accelerator.local_process_index``.

    world_size : int
      The number of processes in the training. When using ``accelerate``, the
      world size can be determined as ``accelerator.num_processes``.

    Attributes
    ----------
    history : skorch.history.History
      The actual skorch ``History`` object can be accessed using the
      :class:`.History` attribute. You should call ``net.history.sync()`` to
      ensure that all data is synced into the history before reading from it.

    """
    def __init__(self, *args, store, rank, world_size):
        self.store = store
        self.rank = rank
        self.world_size = world_size

        self._history = History(*args)
        self._cur_batch = 0
        self._sync_queue = []

    @property
    def history(self):
        self.sync()
        return self._history

    def to_list(self):
        return self.history.to_list()

    @classmethod
    def from_file(cls, f):
        raise NotImplementedError

    def to_file(self, f):
        return self.history.to_file(f)

    def __len__(self):
        return len(self.history)

    def __delitem__(self, i):
        raise NotImplementedError

    def clear(self):
        self.history.clear()

    def new_epoch(self):
        self.sync()
        self._history.new_epoch()
        self._cur_batch = 0

    def new_batch(self):
        self.sync()
        self._history.new_batch()
        self._cur_batch += 1

    def record(self, attr, value):
        self._history.record(attr, value)

    def record_batch(self, attr, value):
        """Add a new value to the given column for the current
        batch.

        Instead of writing to the history directly, write to the distributed
        store. Then, once the history is being read from, the values from the
        store are synchronized.

        This class "remembers" which values were written to the store by
        creating a key that uniquely identifies the values. Choosing a correct
        key is crucial here, since it must not only be unique (say, a uuid), but
        also sufficient to replay the history so that they can be recorded
        correctly.

        When the structure of the key is changed, it must be changed accordingly
        inside of the sync method.

        """
        rank = self.rank
        batch = self._cur_batch
        epoch = len(self._history)
        key = [epoch, batch, rank, attr]
        self.store.set(json.dumps(key), json.dumps(value))

        # the queue not only stores the keys for this process, but the keys for
        # all processes
        for rank in range(self.world_size):
            key = [epoch, batch, rank, attr]
            self._sync_queue.append(key)

    def __getitem__(self, i):
        self.sync()
        return self._history[i]

    def sync(self):
        """Collect batch records across all ranks from store and write them to
        the history

        Syncing is not a single atomic operation, if something breaks the flow,
        we can end up with an inconsistent state.

        """
        if not self._sync_queue:
            return

        # this ensures that that the values are visited in a stable order
        # grouped by batches: order by batch count and within the batches in the
        # order of the rank and within the ranks in the alphabetical order of
        # the attributes
        keys_sorted = sorted(self._sync_queue)
        keys_json = [json.dumps(key) for key in keys_sorted]
        self.store.wait(keys_json)

        batch_prev = None
        rank_prev = None

        for (_, batch, rank, attr), key_json in zip(keys_sorted, keys_json):
            if batch_prev is None:
                batch_prev = batch
            if rank_prev is None:
                rank_prev = rank

            # each time the batch counter or rank changes, it means that the
            # next batch needs to be started
            if (batch != batch_prev) or (rank != rank_prev):
                self._history.new_batch()
                batch_prev = batch
                rank_prev = rank

            val = json.loads(self.store.get(key_json))
            self._history.record_batch(attr, val)

        self._sync_queue.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        try:
            # Unfortunately, TCPStore and FileStore are not pickleable. We still
            # try, in case the user provides another type that can be pickled.
            pickle.dumps(state['store'])
        except (TypeError, pickle.PicklingError):
            # TCPStore and FileStore raise TypeError, others could be
            # PicklingError
            state['store'] = None
        return state
