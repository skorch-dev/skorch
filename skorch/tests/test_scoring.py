import pytest
import numpy as np
import torch
from sklearn.base import clone
from torch import nn

# from skorch.scoring import _CriterionAccumulator, loss_scoring


class Test_ScoreAccumulator:
    @pytest.fixture(scope="module", params=["rgr", "clf"])
    def data_type(self, request):
        return request.param

    @pytest.fixture(scope="module")
    def output_data(self, data_type):
        if data_type == "clf":
            pred = torch.softmax(torch.randn(50, 2), dim=1)
            target = torch.randint(2, size=(50,))
            return pred, target
        if data_type == "rgr":
            target = torch.arange(15) + torch.randn(15)
            pred = target + torch.randn(target.size(0))
            return pred, target
        raise ValueError("Unrecognized data type")

    @pytest.fixture(scope="module", params=["mean", "sum", "none"])
    def reduction(self, request):
        return request.param

    @pytest.fixture(scope="module")
    def criterion_fn(self, data_type):
        if data_type == "clf":
            return nn.CrossEntropyLoss
        if data_type == "rgr":
            return nn.MSELoss
        raise ValueError(f"data_type not recognized: {data_type}.")

    @pytest.fixture(scope="module")
    def accumulator(self, criterion_fn, reduction):
        from skorch.scoring import _CriterionAccumulator

        criterion = criterion_fn(reduction=reduction)
        return _CriterionAccumulator(criterion)

    def test_accumulate(self, accumulator, output_data):
        pred, target = output_data
        accumulator(pred, target)

    def test_reduce(self, accumulator, output_data):
        pred, target = output_data
        accumulator(pred, target)
        accumulator(pred, target)
        output = accumulator.reduce()
        if accumulator._get_reduction() == "none":
            assert isinstance(output, np.ndarray)
        else:
            assert np.isscalar(output)


class TestLossScoring:
    @pytest.fixture(scope="module")
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope="module")
    def net_cls(self):
        from skorch import NeuralNetClassifier

        return NeuralNetClassifier

    @pytest.fixture(scope="module")
    def module_cls(self, classifier_module):
        return classifier_module

    @pytest.fixture(scope="module")
    def net(self, net_cls, module_cls):
        return net_cls(module_cls, lr=0.1)

    @pytest.fixture(scope="module")
    def net_fit(self, net, data):
        X, y = data
        return net.fit(X, y)

    @pytest.fixture(scope="module")
    def loss_scoring_fn(self):
        from skorch.scoring import loss_scoring

        return loss_scoring

    @pytest.fixture(scope="module")
    def scored_net_cls(self, net_cls, loss_scoring_fn):
        class ScoredNet(net_cls):
            def score(self, X, y=None):
                return loss_scoring_fn(self, X, y)

        return ScoredNet

    @pytest.fixture(scope="module")
    def scored_net(self, scored_net_cls, module_cls):
        return scored_net_cls(module_cls, lr=0.1)

    @pytest.fixture(scope="module")
    def scored_net_fit(self, scored_net, data):
        X, y = data
        return scored_net.fit(X, y)

    def test_score_unfit_net_raises(self, loss_scoring_fn, net, data):
        from skorch.exceptions import NotInitializedError

        X, y = data
        with pytest.raises(NotInitializedError):
            loss_scoring_fn(net, X, y)

    def test_score_unfit_scored_net_raises(self, scored_net, data):
        from skorch.exceptions import NotInitializedError

        X, y = data
        with pytest.raises(NotInitializedError):
            scored_net.score(X, y)

    def test_scored_net_output_type(self, scored_net_fit, data):
        X, y = data
        score_value = scored_net_fit.score(X, y)
        assert np.isscalar(score_value)

    def test_score_on_net_fit(self, loss_scoring_fn, net_fit, data):
        X, y = data
        score_value = loss_scoring_fn(net_fit, X, y)
        assert np.isscalar(score_value)
