"""Contains shared fixtures, hooks, etc."""


pandas_installed = False
try:
    # pylint: disable=unused-import
    import pandas
    pandas_installed = True
except ImportError:
    pass
