"""Test for cli.py"""

from math import cos
import os
import subprocess
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import RReLU


fire_installed = True
try:
    import fire
except ImportError:
    fire_installed = False


@pytest.mark.skipif(not fire_installed, reason='fire libarary not installed')
class TestCli:
    @pytest.fixture
    def resolve_dotted_name(self):
        from skorch.cli import _resolve_dotted_name
        return _resolve_dotted_name

    @pytest.mark.parametrize('name, expected', [
        (0, 0),
        (1.23, 1.23),
        ('foo', 'foo'),
        ('math.cos', cos),
        ('torch.nn', nn),
        ('torch.nn.ReLU', nn.ReLU),
    ])
    def test_resolve_dotted_name(self, resolve_dotted_name, name, expected):
        result = resolve_dotted_name(name)
        assert result == expected

    def test_resolve_dotted_name_instantiated(self, resolve_dotted_name):
        result = resolve_dotted_name('torch.nn.RReLU(0.123, upper=0.456)')
        assert isinstance(result, RReLU)
        assert np.isclose(result.lower, 0.123)
        assert np.isclose(result.upper, 0.456)

    @pytest.fixture
    def parse_net_kwargs(self):
        from skorch.cli import parse_net_kwargs
        return parse_net_kwargs

    def test_parse_net_kwargs(self, parse_net_kwargs):
        kwargs = {
            'lr': 0.05,
            'max_epochs': 5,
            'module__num_units': 10,
            'module__nonlin': 'torch.nn.RReLU(0.123, upper=0.456)',
        }
        parsed_kwargs = parse_net_kwargs(kwargs)

        assert len(parsed_kwargs) == 4
        assert np.isclose(parsed_kwargs['lr'], 0.05)
        assert parsed_kwargs['max_epochs'] == 5
        assert parsed_kwargs['module__num_units'] == 10
        assert isinstance(parsed_kwargs['module__nonlin'], RReLU)
        assert np.isclose(parsed_kwargs['module__nonlin'].lower, 0.123)
        assert np.isclose(parsed_kwargs['module__nonlin'].upper, 0.456)

    @pytest.fixture
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def net(self, net_cls, classifier_module):
        return net_cls(classifier_module)

    @pytest.fixture
    def pipe(self, net):
        return Pipeline([
            ('features', FeatureUnion([
                ('scale', MinMaxScaler()),
            ])),
            ('net', net),
        ])

    @pytest.fixture
    def yield_estimators(self):
        from skorch.cli import _yield_estimators
        return _yield_estimators

    def test_yield_estimators_net(self, yield_estimators, net):
        result = list(yield_estimators(net))

        assert result[0][0] == ''
        assert result[0][1] is net
        assert result[1][0] == 'module'
        assert result[1][1] is net.module

    def test_yield_estimators_pipe(self, yield_estimators, pipe):
        result = list(yield_estimators(pipe))
        scaler = pipe.named_steps['features'].transformer_list[0][1]
        net = pipe.named_steps['net']
        module = net.module

        assert result[0][0] == 'features__scale'
        assert result[0][1] is scaler
        assert result[1][0] == 'net'
        assert result[1][1] is net
        assert result[2][0] == 'net__module'
        assert result[2][1] is module

    @pytest.fixture
    def print_help(self):
        from skorch.cli import print_help
        return print_help

    def test_print_help_net(self, print_help, net, capsys):
        print_help(net)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<NeuralNetClassifier> options',
            '--module : torch module (class or instance)',
            '<MLPModule> options',
            '--module__hidden_units : int (default=10)'
        ]
        for snippet in expected_snippets:
            assert snippet in out

    def test_print_help_pipeline(self, print_help, pipe, capsys):
        print_help(pipe)
        out = capsys.readouterr()[0]

        expected_snippets = [
            '-- --help',
            '<MinMaxScaler> options',
            '--features__scale__feature_range',
            '<NeuralNetClassifier> options',
            '--net__module : torch module (class or instance)',
            '<MLPModule> options',
            '--net__module__hidden_units : int (default=10)'
        ]
        for snippet in expected_snippets:
            assert snippet in out

    @pytest.fixture
    def parse_args(self):
        from skorch.cli import parse_args
        return parse_args

    @pytest.fixture
    def estimator(self, net_cls):
        mock = Mock(net_cls)
        return mock

    def test_parse_args_help(self, parse_args, estimator):
        with patch('skorch.cli.sys.exit') as exit:
            with patch('skorch.cli.print_help') as help:
                parsed = parse_args({'help': True, 'foo': 'bar'})
                parsed(estimator)

        assert estimator.set_params.call_count == 0
        assert help.call_count == 1
        assert exit.call_count == 1

    def test_parse_args_run(self, parse_args, estimator):
        with patch('skorch.cli.sys.exit') as exit:
            with patch('skorch.cli.print_help') as help:
                parsed = parse_args({'foo': 'bar', 'baz': 'math.cos'})
                parsed(estimator)

        assert estimator.set_params.call_count == 1
        assert estimator.set_params.call_args_list[0][1]['foo'] == 'bar'
        assert estimator.set_params.call_args_list[0][1]['baz'] == cos
        assert help.call_count == 0
        assert exit.call_count == 0
