"""Tests for callbacks/base.py"""

import warnings

import pytest


# TODO: remove after some deprecation period, e.g. skorch 0.12
class TestIssueWarningIfOnBatchOverride:
    @pytest.fixture
    def net(self, classifier_module):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier(classifier_module, max_epochs=1)

    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope='module')
    def callback_cls(self):
        from skorch.callbacks import Callback
        return Callback

    @pytest.fixture(scope='module')
    def skorch_warning(self):
        from skorch.exceptions import SkorchWarning
        return SkorchWarning

    def test_no_warning_with_default_callbacks(self, net, data, recwarn):
        from skorch.callbacks import EpochScoring
        net.set_params(callbacks=[('f1', EpochScoring('f1'))])
        net.fit(*data)

        assert not recwarn.list

    def test_no_warning_if_on_batch_not_overridden(
            self, net, data, callback_cls, recwarn):
        class MyCallback(callback_cls):
            def on_epoch_end(self, *args, **kwargs):
                pass

        net.set_params(callbacks=[('cb', MyCallback())])
        net.fit(*data)

        assert not recwarn.list

    def test_warning_if_on_batch_begin_overridden(
            self, net, data, callback_cls, skorch_warning):
        class MyCallback(callback_cls):
            def on_batch_begin(self, *args, **kwargs):
                pass

        net.set_params(callbacks=[('cb', MyCallback())])
        with pytest.warns(skorch_warning):
            net.fit(*data)

    def test_warning_if_on_batch_end_overridden(
            self, net, data, callback_cls, skorch_warning):
        class MyCallback(callback_cls):
            def on_batch_end(self, *args, **kwargs):
                pass

        net.set_params(callbacks=[('cb', MyCallback())])
        with pytest.warns(skorch_warning):
            net.fit(*data)

    def test_warning_if_on_batch_begin_and_end_overridden(
            self, net, data, callback_cls, skorch_warning):
        class MyCallback(callback_cls):
            def on_batch_begin(self, *args, **kwargs):
                pass
            def on_batch_end(self, *args, **kwargs):
                pass

        net.set_params(callbacks=[('cb', MyCallback())])
        with pytest.warns(skorch_warning):
            net.fit(*data)

    def test_no_warning_if_not_derived_from_base_and_no_override(
            self, net, data, recwarn):
        from skorch.callbacks import EpochScoring

        class MyCallback(EpochScoring):
            pass

        net.set_params(callbacks=[('f1', MyCallback('f1'))])
        net.fit(*data)

        assert not recwarn.list

    def test_warning_if_not_derived_from_base_and_override(
            self, net, data, skorch_warning):
        from skorch.callbacks import EpochScoring

        class MyCallback(EpochScoring):
            def on_batch_begin(self, *args, **kwargs):
                pass

        net.set_params(callbacks=[('f1', MyCallback('f1'))])

        with pytest.warns(skorch_warning):
            net.fit(*data)

    def test_no_warning_if_filtered(
            self, net, data, callback_cls, skorch_warning, recwarn):
        warnings.filterwarnings('ignore', category=skorch_warning)

        class MyCallback(callback_cls):
            def on_batch_begin(self, *args, **kwargs):
                pass

        net.set_params(callbacks=[('cb', MyCallback())])
        net.fit(*data)

        assert not recwarn.list
