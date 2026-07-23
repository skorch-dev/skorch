"""Tests for virtual parameter setters"""
from unittest.mock import Mock

import pytest


class TestOptimizerSetter:

    @pytest.fixture
    def net_dummy(self):
        from skorch import NeuralNet
        net = Mock(spec=NeuralNet)
        net.lr = 0.01
        return net

    @pytest.fixture
    def optimizer_dummy(self):
        from torch.optim import Optimizer
        optim = Mock(spec=Optimizer)
        optim.param_groups = [
            {'lr': 0.01, 'momentum': 0.9},
            {'lr': 0.02, 'momentum': 0.9}
        ]
        return optim

    @pytest.fixture(scope='function')
    def net_optim_dummy(self, net_dummy, optimizer_dummy):
        net_dummy.optimizer_ = optimizer_dummy
        return net_dummy

    @pytest.fixture
    def setter(self):
        from skorch.setter import optimizer_setter
        return optimizer_setter

    def test_lr_attribute_is_updated(self, setter, net_optim_dummy):
        new_lr = net_optim_dummy.lr + 1
        setter(net_optim_dummy, 'lr', new_lr)

        assert net_optim_dummy.lr == new_lr

    def test_wrong_name_raises(self, setter, net_optim_dummy):
        # should be 'param_groups' instead
        param = 'optimizer__param_group__0__lr'
        value = 0.1
        with pytest.raises(AttributeError) as e:
            setter(net_optim_dummy, param, value)

        assert e.value.args[0] == (
            'Invalid parameter "{param}" for optimizer "optimizer"'
            .format(param=param)
        )

    @pytest.mark.parametrize('group', [0, 1])
    @pytest.mark.parametrize('sub_param, value', [
        ('momentum', 0.1),
        ('lr', 0.3),
    ])
    def test_only_specific_param_group_updated(self, setter, net_optim_dummy,
                                               group, sub_param, value):
        pgroups = net_optim_dummy.optimizer_.param_groups
        param = 'optimizer__param_groups__{}__{}'.format(group, sub_param)

        updated_group_pre = [g for i, g in enumerate(pgroups) if i == group]
        static_groups_pre = [g for i, g, in enumerate(pgroups) if i != group]
        assert len(updated_group_pre) == 1

        setter(net_optim_dummy, param, value)

        updated_group_new = [g for i, g in enumerate(pgroups) if i == group]
        static_groups_new = [g for i, g, in enumerate(pgroups) if i != group]

        assert updated_group_new[0][sub_param] == value
        assert all(old[sub_param] == new[sub_param] for old, new in zip(
            static_groups_pre, static_groups_new))

    def test_set_params_verbose_prints_param_group(self, setter, capsys):
        from skorch import NeuralNetClassifier
        from skorch.toy import make_classifier

        net = NeuralNetClassifier(make_classifier(), verbose=1, max_epochs=1)
        net.initialize()
        setter(net, 'optimizer__param_groups__0__lr', 0.03)
        out = capsys.readouterr().out
        assert "Setting param group {'lr': 0.03} for" in out
        assert 'sequential.0.weight' in out

    def test_set_params_verbose_silent_for_global_param(self, setter, capsys):
        from skorch import NeuralNetClassifier
        from skorch.toy import make_classifier

        net = NeuralNetClassifier(make_classifier(), verbose=1, max_epochs=1)
        net.initialize()
        setter(net, 'optimizer__lr', 0.03)
        out = capsys.readouterr().out
        assert 'Setting param group' not in out

    def test_format_param_group_msg_no_known_params(self):
        from skorch.setter import format_param_group_msg

        msg = format_param_group_msg({'lr': 0.1}, [])
        assert "{'lr': 0.1}" in msg
        assert 'this may be unintended' in msg

    def test_format_param_group_msg_truncated(self):
        from skorch.setter import MAX_PARAM_GROUP_MSG_LEN
        from skorch.setter import format_param_group_msg

        names = ['sequential.{}.weight'.format(i) for i in range(100)]
        msg = format_param_group_msg({'lr': 0.1}, names)
        assert len(msg) == MAX_PARAM_GROUP_MSG_LEN
        assert msg.endswith('...')
