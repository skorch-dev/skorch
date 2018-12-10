"""Setter functions for virtual params such as ``optimizer__lr``."""
import re


def _extract_optimizer_param_name_and_group(optimizer_name, param):
    """Extract param group and param name from the given parameter name.
    Raises an error if the param name doesn't match one of
    - ``optimizer__param_groups__<group>__<name>``
    - ``optimizer__<name>``
    In the second case group defaults to 'all'.
    The second case explicitly forbids ``optimizer__foo__bar``
    since we do not know how to deal with unknown sub-params.
    """
    pat_1 = '__param_groups__(?P<group>[0-9])__(?P<name>.+)'
    pat_2 = '__(?!.*__.*)(?P<name>.+)'
    pat_1 = optimizer_name + pat_1
    pat_2 = optimizer_name + pat_2

    match_1 = re.compile(pat_1).fullmatch(param)
    match_2 = re.compile(pat_2).fullmatch(param)
    match = match_1 or match_2

    if not match:
        raise AttributeError('Invalid parameter "{}" for optimizer "{}"'.format(
            param,
            optimizer_name,
        ))

    groups = match.groupdict()
    param_group = groups.get('group', 'all')
    param_name = groups['name']
    return param_group, param_name


def _set_optimizer_param(optimizer, param_group, param_name, value):
    """Set a parameter on an all or a specific parameter group of an
    optimizer instance. To select all param groups, use ``param_group='all'``.
    """
    if param_group == 'all':
        groups = optimizer.param_groups
    else:
        groups = [optimizer.param_groups[int(param_group)]]

    for group in groups:
        group[param_name] = value


def optimizer_setter(
        net, param, value, optimizer_attr='optimizer_', optimizer_name='optimizer'
    ):
    """Handle setting of optimizer parameters such as learning rate and
    parameter group specific parameters such as momentum.

    The parameters ``optimizer_attr`` and ``optimizer_name`` can be specified
    if there exists more than one optimizer (e.g., in seq2seq models).
    """
    if param == 'lr':
        param_group = 'all'
        param_name = 'lr'
        net.lr = value
    else:
        param_group, param_name = _extract_optimizer_param_name_and_group(
            optimizer_name, param)

    _set_optimizer_param(
        optimizer=getattr(net, optimizer_attr),
        param_group=param_group,
        param_name=param_name,
        value=value
    )
