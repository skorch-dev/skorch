======
Helper
======

This module provides helper functions and classes for the user. They
make working with skorch easier but are not used by skorch itself.

SliceDict
---------

A :class:`.SliceDict` is a wrapper for Python dictionaries that makes
them behave a little bit like :class:`numpy.ndarray`\s. That way, you
can slice your dictionary across values, ``len()`` will show the
length of the arrays and not the number of keys, and you get a
``shape`` attribute.  This is useful because if your data is in a
``dict``, you would normally not be able to use sklearn
:class:`~sklearn.model_selection.GridSearchCV` and similar things;
with :class:`.SliceDict`, this works.
