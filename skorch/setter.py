"""Setter functions for virtual params such as ``optimizer__lr``."""
import re

# param names can be arbitrarily long, keep the verbose message bounded
MAX_PARAM_GROUP_MSG_LEN = 200


def format_param_group_msg(group_config, param_names):
    """Message for which module params a param group config applies to."""
    if not param_names:
        msg = (
            "Setting param group {} for parameters that are not among the "
            "module's learnable parameters (this may be unintended).".format(
                group_config,
            )
        )
    else:
        msg = "Setting param group {} for {}.".format(
            group_config,
            ', '.join(param_names),
        )
    if len(msg) > MAX_PARAM_GROUP_MSG_LEN:
        msg = msg[:MAX_PARAM_GROUP_MSG_LEN - 3] + '...'
    return msg


def _param_names_for_tensors(net, tensors):
    """Map optimizer tensors back to module parameter names when possible."""
    tensor_ids = {id(t) for t in tensors}
    names = []
    get_params = getattr(net, 'get_all_learnable_params', None)
    if get_params is None:
        return names
    for name, p in get_params():
        if id(p) in tensor_ids:
            names.append(name)
    return names


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

    return groups


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

    groups = _set_optimizer_param(
        optimizer=getattr(net, optimizer_attr),
        param_group=param_group,
        param_name=param_name,
        value=value
    )

    # only report for a specific param group; a global set (e.g. optimizer__lr)
    # touches every param and is not what #291 asks to surface
    if getattr(net, 'verbose', 0) and param_group != 'all':
        tensors = []
        for group in groups:
            tensors.extend(group.get('params', []))
        param_names = _param_names_for_tensors(net, tensors)
        print(format_param_group_msg({param_name: value}, param_names))
