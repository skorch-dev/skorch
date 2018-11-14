"""Setter functions for virtual params such as ``optimizer__lr``."""
import re

from skorch.utils import set_optimizer_param


def optimizer_setter(
        net, param, value, optimizer_attr='optimizer_', optimizer_name='optimizer'
    ):
    """Handle setting of optimizer parameters such as learning rate and
    parameter group specific parameters such as momentum.
    """
    if param == 'lr':
        param_group = 'all'
        param_name = 'lr'
        net.lr = value
    else:
        pat_1 = '__param_groups__(?P<group>[0-9])__(?P<name>.+)'
        pat_2 = '__(?P<name>.+)'
        pat_1 = optimizer_name + pat_1
        pat_2 = optimizer_name + pat_2

        match_1 = re.compile(pat_1).fullmatch(param)
        match_2 = re.compile(pat_2).fullmatch(param)
        match = match_1 or match_2

        if not match or '__' in match.groupdict().get('name', ''):
            raise AttributeError('Invalid parameter "{}" for optimizer {}'.format(
                param,
                optimizer_name,
            ))

        groups = match.groupdict()
        param_group = groups.get('group', 'all')
        param_name = groups['name']

    set_optimizer_param(
        optimizer=getattr(net, optimizer_attr),
        param_group=param_group,
        param_name=param_name,
        value=value
    )
