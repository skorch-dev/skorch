"""Tests for early stopping callback"""

import numpy as np
import pytest


class TestEarlyStopping:

    @pytest.fixture
    def early_stopping_cls(self):
        from skorch.callbacks import EarlyStopping
        return EarlyStopping

    @pytest.fixture
    def net_clf_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def broken_classifier_module(self, classifier_module):
        """Return a classifier that does not imporve over time."""
        class BrokenClassifier(classifier_module):
            def forward(self, x):
                return super().forward(x) * 0 + 0.5
        return BrokenClassifier

    def test_typical_use_case_nonstop(
            self, net_clf_cls, classifier_module, classifier_data, early_stopping_cls,
    ):
        stop_threshold = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=stop_threshold)

        net = net_clf_cls(
            classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        ok_run = (
            len(net.history) == max_epochs
        )
        assert ok_run

    def test_typical_use_case_stopping(
            self, net_clf_cls, broken_classifier_module, classifier_data, early_stopping_cls,
    ):
        stop_threshold = 5
        max_epochs = 8
        early_stopping_cb = early_stopping_cls(patience=stop_threshold)

        net = net_clf_cls(
            broken_classifier_module,
            callbacks=[
                early_stopping_cb,
            ],
            max_epochs=max_epochs,
        )
        net.fit(*classifier_data)

        bad_run = (
            len(net.history) != max_epochs
        )

        assert bad_run



