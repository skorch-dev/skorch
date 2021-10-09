"""Tests for scoring.py"""

import pytest
import numpy as np
import torch


class TestLossScoring:
    @pytest.fixture(scope="module")
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope="module", params=["mean", "sum"])
    def reduction(self, request):
        return request.param

    @pytest.fixture(scope="module")
    def net_cls(self):
        from skorch import NeuralNetClassifier

        return NeuralNetClassifier

    @pytest.fixture(scope="module")
    def module_cls(self, classifier_module):
        return classifier_module

    @pytest.fixture(scope="module")
    def net(self, net_cls, module_cls, reduction):
        return net_cls(module_cls, lr=0.1, criterion__reduction=reduction)

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
    def scored_net(self, scored_net_cls, module_cls, reduction):
        return scored_net_cls(
            module_cls, lr=0.01, criterion__reduction=reduction
        )

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

    def test_nonnull_sample_weight_raises(self, loss_scoring_fn, net_fit, data):
        X, y = data
        with pytest.raises(NotImplementedError):
            loss_scoring_fn(
                net_fit, X, y, sample_weight=np.random.rand(X.shape[0])
            )

    def test_scored_net_output_type(self, scored_net_fit, data):
        X, y = data
        score_value = scored_net_fit.score(X, y)
        assert np.isscalar(score_value)

    def test_score_on_net_fit(self, loss_scoring_fn, net_fit, data):
        X, y = data
        score_value = loss_scoring_fn(net_fit, X, y)
        assert np.isscalar(score_value)

    def test_scored_net_matches_criterion_value(self, scored_net_fit, data):
        X, y = data
        y_val_proba = torch.as_tensor(scored_net_fit.predict_proba(X))
        loss_value = scored_net_fit.get_loss(y_val_proba, y)
        score_value = scored_net_fit.score(X, y)
        assert np.allclose(score_value, loss_value.item())

    def test_scored_net_with_reduction_none(
            self, scored_net_cls, module_cls, reduction, data
    ):
        X, y = data
        net = scored_net_cls(
            module_cls, lr=0.01, criterion__reduction=reduction
        ).fit(X, y)
        net.set_params(criterion__reduction="sum")
        loss_value = net.score(X, y)
        net.set_params(criterion__reduction="none")
        output = net.score(X, y)
        assert output.shape[0] == X.shape[0]
        assert np.allclose(output.sum(), loss_value)

    def test_score_unknown_reduction_raises(
            self, loss_scoring_fn, net_fit, data
    ):
        X, y = data
        net_fit.set_params(criterion__reduction="unk")
        with pytest.raises(ValueError, match="for reduction but got"):
            loss_scoring_fn(net_fit, X, y)
