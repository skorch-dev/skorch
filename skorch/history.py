"""Contains history class and helper functions."""

import json
import warnings

from skorch.utils import open_file_like


# pylint: disable=invalid-name
class _none:
    """Special placeholder since ``None`` is a valid value."""


def _not_none(items):
    """Whether the item is a placeholder or contains a placeholder."""
    if not isinstance(items, (tuple, list)):
        items = (items,)
    return all(item is not _none for item in items)


def _filter_none(items):
    """Filter special placeholder value, preserves sequence type."""
    type_ = list if isinstance(items, list) else tuple
    return type_(filter(_not_none, items))


def _getitem(item, i):
    """Extract value or values from dicts.

    Covers the case of a single key or multiple keys. If not found,
    return placeholders instead.

    """
    if not isinstance(i, (tuple, list)):
        return item.get(i, _none)
    type_ = list if isinstance(item, list) else tuple
    return type_(item.get(j, _none) for j in i)


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

    # handle special case of
    # history[j, 'batches', somekey]
    # which should really be
    # history[j, 'batches', :, somekey]
    if i_b is not None and not isinstance(i_b, (int, slice)):
        if k_b is not None:
            raise KeyError("The last argument '{}' is invalid; it must be a "
                           "string or tuple of strings.".format(k_b))
        warnings.warn(
            "Argument 3 to history slicing must be of type int or slice, e.g. "
            "history[:, 'batches', 'train_loss'] should be "
            "history[:, 'batches', :, 'train_loss'].",
            DeprecationWarning,
        )
        i_b, k_b = slice(None), i_b

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
        keyerror_msg = "Key '{}' was not found in history."

        if i_b is not None and k_e != 'batches':
            raise KeyError("History indexing beyond the 2nd level is "
                           "only possible if key 'batches' is used, "
                           "found key '{}'.".format(k_e))

        items = self.to_list()

        # extract indices of batches
        # handles: history[..., k_e, i_b]
        if i_b is not None:
            items = [row[k_e][i_b] for row in items]

        # extract keys of batches
        # handles: history[..., k_e, i_b][k_b]
        if k_b is not None:
            items = [
                _filter_none([_getitem(b, k_b) for b in batches])
                if isinstance(batches, (list, tuple))
                else _getitem(batches, k_b)
                for batches in items
            ]
            # get rid of empty batches
            items = [b for b in items if b not in (_none, [], ())]
            if not _filter_none(items):
                # all rows contained _none or were empty
                raise KeyError(keyerror_msg.format(k_b))

        # extract epoch-level values, but only if not already done
        # handles: history[..., k_e]
        if (k_e is not None) and (i_b is None):
            items = [_getitem(batches, k_e)
                     for batches in items]
            if not _filter_none(items):
                raise KeyError(keyerror_msg.format(k_e))

        # extract the epochs
        # handles: history[i_b, ..., ..., ...]
        if i_e is not None:
            items = items[i_e]
            if isinstance(i_e, slice):
                items = _filter_none(items)
            if items is _none:
                raise KeyError(keyerror_msg.format(k_e))

        return items
