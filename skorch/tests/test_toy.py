"""Tests for toy.py."""

import numpy as np
import pytest
import torch
from torch import nn


class TestMLPModule:
    @pytest.fixture
    def module_cls(self):
        from skorch.toy import MLPModule
        return MLPModule

    def test_one_hidden(self, module_cls):
        module = module_cls()
        parameters = list(module.named_parameters())

        # 2 linear * (weight + bias)
        assert len(parameters) == 4

        # 2 linear, 1 relu, 1 dropout
        assert len(module.sequential) == 4
        assert isinstance(module.sequential[0], nn.Linear)
        assert isinstance(module.sequential[1], nn.ReLU)
        assert isinstance(module.sequential[2], nn.Dropout)
        assert isinstance(module.sequential[3], nn.Linear)

    def test_two_hidden(self, module_cls):
        module = module_cls(num_hidden=2)
        parameters = list(module.named_parameters())

        # 3 linear * (weight + bias)
        assert len(parameters) == 6

        # 3 linear, 2 relu, 2 dropout
        assert len(module.sequential) == 7
        assert isinstance(module.sequential[0], nn.Linear)
        assert isinstance(module.sequential[1], nn.ReLU)
        assert isinstance(module.sequential[2], nn.Dropout)
        assert isinstance(module.sequential[3], nn.Linear)
        assert isinstance(module.sequential[4], nn.ReLU)
        assert isinstance(module.sequential[5], nn.Dropout)
        assert isinstance(module.sequential[6], nn.Linear)

    @pytest.mark.parametrize('num_hidden', [0, 1, 2, 5, 10])
    def test_many_hidden(self, module_cls, num_hidden):
        module = module_cls(num_hidden=num_hidden)
        parameters = list(module.named_parameters())

        assert len(parameters) == 2 * (num_hidden + 1)
        assert len(module.sequential) == (3 * num_hidden) + 1

    def test_output_nonlin(self, module_cls):
        module = module_cls(output_nonlin=nn.Sigmoid())

        # 2 linear, 1 relu, 1 dropout, 1 sigmoid
        assert len(module.sequential) == 5
        assert isinstance(module.sequential[0], nn.Linear)
        assert isinstance(module.sequential[1], nn.ReLU)
        assert isinstance(module.sequential[2], nn.Dropout)
        assert isinstance(module.sequential[3], nn.Linear)
        assert isinstance(module.sequential[4], nn.Sigmoid)

    def test_output_squeezed(self, module_cls):
        X = torch.zeros((5, 20)).float()

        module = module_cls(output_units=1)
        y = module(X)
        assert y.dim() == 2

        module = module_cls(squeeze_output=True, output_units=1)
        y = module(X)
        assert y.dim() == 1

    def test_dropout(self, module_cls):
        module = module_cls(dropout=0.567)
        assert np.isclose(module.sequential[2].p, 0.567)

    def test_make_classifier(self):
        from skorch.toy import make_classifier
        module = make_classifier()()
        assert isinstance(module.sequential[-1], nn.Softmax)

    def test_make_binary_classifier(self):
        from skorch.toy import make_binary_classifier
        module = make_binary_classifier()()
        assert isinstance(module.sequential[-1], nn.Linear)
        assert module.squeeze_output is True

    def test_make_regressor(self):
        from skorch.toy import make_regressor
        module = make_regressor()()
        assert module.sequential[-1].out_features == 1
