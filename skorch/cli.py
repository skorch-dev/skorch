"""Helper functions for quick command line interfaces with skorch and
fire.

"""

from functools import partial
from importlib import import_module
from itertools import chain
import re
import shlex
import sys

from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


__all__ = ['parse_args']


# matches: bar(), foo.bar(), foo.bar(baz)
P_PARAMS = re.compile(r"(?P<name>^[a-zA-Z][a-zA-Z0-9_\.]*)(?P<params>\(.*\)$)")

P_DEFAULTS = re.compile(
    # standard, matches: int (default=123)
    r"(.+\s\(default\s?\=\s?(?P<default>.+)\)$)|"
    # no parens, matches: int, default=123
    r"(.+\sdefault\s?\=\s?(?P<default_np>.+)$)|"
    # no equal, matches: int, default 123
    r"(.+default\s(?P<default_ne>.+))|"
    # 'by-default', matches: str (l2 by default)
    r"[^\(]+\((?P<default_bd>[^\"\']+)(\sby\sdefault\)?)|"
    # 'by-default-double-tick', matches: "l1" or "l2" ("l2" by default)
    r"[^\(]+\(\"(?P<default_bd_dt>.+)\"\sby\sdefault\)?|"
    # 'by-default-single-tick', matches: 'l1' or 'l2' ('l2' by default)
    r"[^\(]+\(\'(?P<default_bd_st>.+)\'\sby\sdefault\)?"
)


def _param_split(params):
    return (p.strip(' ,') for p in shlex.split(params))


def _get_span(s, pattern):
    """Return the span of the first group that matches the pattern."""
    i, j = -1, -1

    match = pattern.match(s)
    if not match:
        return i, j

    for group_name in pattern.groupindex:
        i, j = match.span(group_name)
        if (i, j) != (-1, -1):
            return i, j

    return i, j


def _substitute_default(s, new_value):
    """Replaces the default value in a parameter docstring by a new value.

    The docstring must conform to the numpydoc style and have the form
    "something (keyname=<value-to-replace>)"

    If no matching pattern is found or ``new_value`` is None, return
    the input untouched.

    Examples
    --------
    >>> _replace_default('int (default=128)', 256)
    'int (default=256)'
    >>> _replace_default('nonlin (default = ReLU())', nn.Hardtanh(1, 2))
    'nonlin (default = Hardtanh(min_val=1, max_val=2))'

    """
    if new_value is None:
        return s

    # BB: ideally, I would like to replace the 'default*' group
    # directly but I haven't found a way to do this
    i, j = _get_span(s, pattern=P_DEFAULTS)
    if (i, j) == (-1, -1):
        return s
    return '{}{}{}'.format(s[:i], new_value, s[j:])


def _parse_args_kwargs(params):
    from fire.parser import DefaultParseValue

    args = ()
    kwargs = {}
    for param in _param_split(params):
        if '=' not in param:
            args += (DefaultParseValue(param),)
        else:
            k, v = param.split('=')
            kwargs[k.strip()] = DefaultParseValue(v)
    return args, kwargs


def _resolve_dotted_name(dotted_name):
    """Returns objects from strings

    Deals e.g. with 'torch.nn.Softmax(dim=-1)'.

    Modified from palladium:

    https://github.com/ottogroup/palladium/blob/8a066a9a7690557d9b1b6ed54b7d1a1502ba59e3/palladium/util.py

    with added support for instantiated objects.

    """
    if not isinstance(dotted_name, str):
        return dotted_name

    if '.' not in dotted_name:
        return dotted_name

    args = None
    params = None
    match = P_PARAMS.match(dotted_name)
    if match:
        dotted_name = match.group('name')
        params = match.group('params')

    module, name = dotted_name.rsplit('.', 1)
    attr = import_module(module)
    attr = getattr(attr, name)

    if params:
        args, kwargs = _parse_args_kwargs(params[1:-1])
        attr = attr(*args, **kwargs)

    return attr


def parse_net_kwargs(kwargs):
    """Parse arguments for the estimator.

    Resolves dotted names and instantiated classes.

    Examples
    --------
    >>> kwargs = {'lr': 0.1, 'module__nonlin': 'torch.nn.Hardtanh(-2, max_val=3)'}
    >>> parse_net_kwargs(kwargs)
    {'lr': 0.1, 'module__nonlin': Hardtanh(min_val=-2, max_val=3)}

    """
    if not kwargs:
        return kwargs

    resolved = {}
    for k, v in kwargs.items():
        resolved[k] = _resolve_dotted_name(v)

    return resolved


def _yield_preproc_steps(model):
    if not isinstance(model, Pipeline):
        return

    preproc_pipe = Pipeline(model.steps[:-1])
    for key, val in preproc_pipe.get_params().items():
        if isinstance(val, BaseEstimator):
            if not isinstance(val, (Pipeline, FeatureUnion)):
                yield key, val


def _yield_estimators(model):
    """Yield estimator and its prefix from the model.

    First, pipeline preprocessing steps are yielded (if there are
    any). Next the neural net is yielded. Finally, the module is
    yielded.

    """
    yield from _yield_preproc_steps(model)

    net_prefixes = []
    module_prefixes = []

    if isinstance(model, Pipeline):
        name = model.steps[-1][0]
        net_prefixes.append(name)
        module_prefixes.append(name)
        net = model.steps[-1][1]
    else:
        net = model

    yield '__'.join(net_prefixes), net

    module = getattr(net, 'module', None)
    if not module:
        # There is no module attribute, we're dealing with a normal
        # scikit-learn estimator, so no need to show further help.
        return

    module_prefixes.append('module')
    yield '__'.join(module_prefixes), module


def _extract_estimator_cls(estimator):
    if isinstance(estimator, partial):
        # is partialled
        return estimator.func
    if not isinstance(estimator, type):
        # is instance
        return estimator.__class__
    return estimator


def _yield_printable_params(param, prefix, defaults):
    name, default, descr = param
    name = name if not prefix else '__'.join((prefix, name))
    default = _substitute_default(default, defaults.get(name))

    printable = '--{} : {}'.format(name, default)
    yield printable

    for line in descr:
        yield line


def _get_help_for_params(params, prefix='--', defaults=None, indent=2):
    defaults = defaults or {}
    for param in params:
        first, *rest = tuple(_yield_printable_params(
            param, prefix=prefix, defaults=defaults))
        yield " " * indent + first
        for line in rest:
            yield " " * 2 * indent + line


def _get_help_for_estimator(prefix, estimator, defaults=None):
    """Yield help lines for the given estimator and prefix."""
    from numpydoc.docscrape import ClassDoc

    defaults = defaults or {}
    estimator = _extract_estimator_cls(estimator)
    yield "<{}> options:".format(estimator.__name__)

    doc = ClassDoc(estimator)
    yield from _get_help_for_params(
        doc['Parameters'],
        prefix=prefix,
        defaults=defaults,
    )
    yield ''  # add a newline line between estimators


def print_help(model, defaults=None):
    """Print help for the command line arguments of the given model.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
      The basic model, e.g. a ``NeuralNet`` or sklearn ``Pipeline``.

    defautls : dict or None (default=None)
      Optionally, change the default values to use custom
      defaults. Commandline arguments have precedence over defaults.

    """
    defaults = defaults or {}

    print("This is the help for the model-specific parameters.")
    print("To invoke help for the remaining options, run:")
    print("python {} -- --help".format(sys.argv[0]))
    print()

    lines = (_get_help_for_estimator(prefix, estimator, defaults=defaults) for
             prefix, estimator in _yield_estimators(model))
    print('\n'.join(chain(*lines)))


def parse_args(kwargs, defaults=None):
    """Apply command line arguments or show help.

    Use this in conjunction with the fire library to quickly build
    command line interfaces for your scripts.

    This function returns another function that must be called with
    the estimator (e.g. ``NeuralNet``) to apply the parsed command
    line arguments. If the --help option is found, show the
    estimator-specific help instead.

    Examples
    --------
    Content of my_script.py:

    >>> def main(**kwargs):
    >>>     X, y = get_data()
    >>>     my_model = get_model()
    >>>     parsed = parse_args(kwargs)
    >>>     my_model = parsed(my_model)
    >>>     my_model.fit(X, y)
    >>>
    >>> if __name__ == '__main__':
    >>>     fire.Fire(main)


    Note
    ----
    The function you pass to `fire.Fire` shouldn't have any positional
    arguments, otherwise the displayed help will not correctly work;
    this is a quirk of fire.

    Parameters
    ----------
    kwargs : dict
      The arguments as parsed by fire.

    defaults : dict or None (default=None)
      Optionally, change the default values to use custom
      defaults. Commandline arguments have precedence over defaults.

    Returns
    -------
    print_help_and_exit : callable
      If --help is in the arguments, print help and exit.

    set_params : callable
      If --help is not in the options, apply command line arguments to
      the estimator and return it.

    """
    try:
        import fire  # pylint: disable=unused-import
    except ImportError:
        raise ImportError(
            "Using skorch cli helpers requires the fire library,"
            " you can install it with pip: python -m pip install fire."
        )
    try:
        import numpydoc.docscrape  # pylint: disable=unused-import
    except ImportError:
        raise ImportError(
            "Using skorch cli helpers requires the numpydoc library,"
            " you can install it with pip: python -m pip install numpydoc."
        )

    defaults = defaults or {}

    def print_help_and_exit(estimator):
        print_help(estimator, defaults=defaults)
        sys.exit()

    def set_params(estimator):
        estimator.set_params(**defaults)
        return estimator.set_params(**parse_net_kwargs(kwargs))

    if kwargs.get('help'):
        return print_help_and_exit
    return set_params
