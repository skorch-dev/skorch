======
Helper
======

This module provides helper functions and classes for the user. They
make working with skorch easier but are not used by skorch itself.

SliceDict
---------

A :class:`SliceDict <skorch.helper.SliceDict>` is a wrapper for Python
dictionaries that makes them behave a little bit like numpy
arrays. That way, you can slice your dictionary across values, `len`
will show the length of the arrays and not the number of keys, and you
get a `shape` attribute. This is useful because if your data is in a
dict, you would normally not be able to use sklearn `GridSearchCV` and
similar things; with SliceDict, this works.
