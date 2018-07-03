"""Tests for early stopping callback"""

import pytest


class TestEarlyStopping:

    @pytest.fixture
    def early_stopping_cls(self):
        from skorch.callbacks import EarlyStopping
        return EarlyStopping

    @pytest.fixture
    def epoch_scoring_cls(self):
        from skorch.callbacks import EpochScoring
        return EpochScoring

    @pytest.fixture
    def net_clf_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def broken_classifier_module(self, classifier_module):
        """Return a classifier that does not improve over time."""
        class BrokenClassifier(classifier_module):
            def forward(self, x):
                return super().forward(x) * 0 + 0.5
        return BrokenClassifier

    def test_typical_use_case_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_typical_use_case_stopping(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

    def test_custom_scoring_nonstop(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_cb = epoch_scoring_cls(
            'roc_auc', lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == max_epochs

    def test_custom_scoring_stop(
            self, net_clf_cls, broken_classifier_module, classifier_data,
            early_stopping_cls, epoch_scoring_cls,
    ):
        lower_is_better = False
        scoring_name = 'valid_roc_auc'
        patience = 5
        max_epochs = 8
        scoring_cb = epoch_scoring_cls(
            'roc_auc', lower_is_better, name=scoring_name)
        early_stopping_cb = early_stopping_cls(
            patience=patience, lower_is_better=lower_is_better,
            monitor=scoring_name)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                scoring_cb,
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) < max_epochs

    def test_stopping_big_absolute_threshold(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=patience,
                                               threshold_mode='abs',
                                               threshold=0.1)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        assert len(net.history) == patience + 1 < max_epochs

    def test_wrong_threshold_mode(
            self, net_clf_cls, classifier_module, classifier_data,
            early_stopping_cls):
        patience = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(
            patience=patience, threshold_mode='incorrect')
        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )

        with pytest.raises(ValueError) as exc:
            net.fit(*classifier_data)

        expected_msg = "Invalid threshold mode: 'incorrect'"
        assert exc.value.args[0] == expected_msg
