"""Tests for skorch._doctor.py"""

import itertools

import numpy as np
import pytest
import torch
from torch import nn


class TestSkorchDoctorSimple:
    @pytest.fixture(scope='module')
    def module_cls(self):
        class MyModule(nn.Module):
            """Module with predictable parameters"""
            def __init__(self):
                super().__init__()

                self.lin0 = nn.Linear(20, 20)
                nn.init.eye_(self.lin0.weight)
                nn.init.zeros_(self.lin0.bias)

                self.lin1 = nn.Linear(20, 2)
                nn.init.zeros_(self.lin1.weight)
                nn.init.ones_(self.lin1.bias)

                self.softmax = nn.Softmax(dim=-1)

            def forward(self, X):
                X = self.lin0(X)
                X = self.lin1(X)
                return self.softmax(X)

        return MyModule

    @pytest.fixture(scope='module')
    def custom_split(self):
        class Split():
            """Deterministically split train/valid into 80%/20%"""
            def __call__(self, dataset, y=None, groups=None):
                n = int(len(dataset) * 0.8)
                dataset_train = torch.utils.data.Subset(dataset, np.arange(n))
                dataset_valid = torch.utils.data.Subset(
                    dataset, np.arange(n, len(dataset)))
                return dataset_train, dataset_valid

        return Split()

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def doctor_cls(self):
        from skorch.helper import SkorchDoctor
        return SkorchDoctor

    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        X, y = classifier_data
        # a small amount of data is enough
        return X[:50], y[:50]

    @pytest.fixture(scope='module')
    def doctor(self, module_cls, net_cls, doctor_cls, data, custom_split):
        net = net_cls(module_cls, max_epochs=3, batch_size=32, train_split=custom_split)
        doctor = doctor_cls(net)
        doctor.fit(*data)
        return doctor

    def test_activation_logs_general_content(self, doctor):
        logs = doctor.activation_logs_
        assert len(logs) == 2
        assert set(logs.keys()) == {'module', 'criterion'}

        # nothing logged for criterion, 3 epochs
        assert logs['criterion'] == [[], [], []]

        logs_module = logs['module']
        # 3 epochs, 2 batches per epoch
        assert len(logs_module) == 3
        assert [len(batch) for batch in logs_module] == [2, 2, 2]

        # each batch has layers lin0, lin1, softmax
        for epoch in logs_module:
            for batch in epoch:
                assert set(batch.keys()) == {'lin0', 'lin1', 'softmax'}

    def test_activation_logs_values(self, doctor, data):
        logs_module = doctor.activation_logs_['module']

        for key in ('lin0', 'lin1', 'softmax'):
            # 80% of 50 samples is 40, batch size 32 => 32 + 8 samples per batch
            batch_sizes = [[len(b[key]) for b in batch] for batch in logs_module]
            assert batch_sizes == [[32, 8], [32, 8], [32, 8]]

        X, _ = data
        # for the very first batch, before any update, we actually know the values
        batch = logs_module[0][0]
        lin0_0 = batch['lin0']
        # since it is the identity function, batches should equal the data
        np.testing.assert_array_almost_equal(lin0_0, X[:32])

        lin1_0 = batch['lin1']
        # since weights are 0 and bias is 1, all values should be 1
        np.testing.assert_array_almost_equal(lin1_0, 1.0)

        softmax_0 = batch['softmax']
        # since all inputs are equal, probabilities should be uniform
        np.testing.assert_array_almost_equal(softmax_0, 0.5)

    def test_activation_logs_not_all_identical(self, doctor):
        # make sure that values are not just all identical, using a large
        # tolerance to exclude small deviations
        logs_flat = list(doctor.flatten(doctor.activation_logs_['module']))

        logs_lin0 = [log['lin0'] for log in logs_flat]
        for act0, act1 in itertools.combinations(logs_lin0, r=2):
            if act0.shape == act1.shape:
                assert not np.allclose(act0, act1, rtol=1e-3)

        logs_lin1 = [log['lin1'] for log in logs_flat]
        for act0, act1 in itertools.combinations(logs_lin1, r=2):
            if act0.shape == act1.shape:
                assert not np.allclose(act0, act1, rtol=1e-3)

        softmax = [log['softmax'] for log in logs_flat]
        for act0, act1 in itertools.combinations(softmax, r=2):
            if act0.shape == act1.shape:
                assert not np.allclose(act0, act1, rtol=1e-3)

    def test_gradient_logs_general_content(self, doctor):
        logs = doctor.gradient_logs_
        assert len(logs) == 2
        assert set(logs.keys()) == {'module', 'criterion'}

        # nothing logged for criterion, 3 epochs
        assert logs['criterion'] == [[], [], []]

        logs_module = logs['module']
        # 3 epochs, 2 batches per epoch
        assert len(logs_module) == 3
        assert [len(batch) for batch in logs_module] == [2, 2, 2]

        # each batch has weights and biases for lin0 & lin1
        for epoch in logs_module:
            for batch in epoch:
                expected = {'lin0.weight', 'lin0.bias', 'lin1.weight', 'lin1.bias'}
                assert set(batch.keys()) == expected

    def test_gradient_logs_values(self, doctor):
        logs_module = doctor.gradient_logs_['module']

        expected_shapes = {
            'lin0.weight': (20, 20),
            'lin0.bias': (20,),
            'lin1.weight': (2, 20),
            'lin1.bias': (2,),
        }
        for key in ('lin0.weight', 'lin0.bias', 'lin1.weight', 'lin1.bias'):
            grad_shapes = [[b[key].shape for b in batch] for batch in logs_module]
            expected_shape = expected_shapes[key]
            # 2 batches, 3 epochs
            assert grad_shapes == [[expected_shape] * 2] * 3

        # There is not really much we know about the gradient values, we just
        # rely on the gradient hooks doing what they're supposed to do. The only
        # gradients we actually know are for the first layer in the first batch:
        # They have to be zero because in the second layer, we have weights of 0.

        batch0 = logs_module[0][0]
        grad_weight = batch0['lin0.weight']
        grad_bias = batch0['lin0.bias']
        assert np.allclose(grad_weight, 0.0)
        assert np.allclose(grad_bias, 0.0)

    def test_gradient_logs_not_all_identical(self, doctor):
        # make sure that values are not just all identical, using a large
        # tolerance to exclude small deviations
        logs_flat = list(doctor.flatten(doctor.gradient_logs_['module']))

        logs_lin0_weight = [log['lin0.weight'] for log in logs_flat]
        for grad0, grad1 in itertools.combinations(logs_lin0_weight, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

        logs_lin0_bias = [log['lin0.bias'] for log in logs_flat]
        for grad0, grad1 in itertools.combinations(logs_lin0_bias, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

        logs_lin1_weight = [log['lin1.weight'] for log in logs_flat]
        for grad0, grad1 in itertools.combinations(logs_lin1_weight, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

        logs_lin1_bias = [log['lin1.bias'] for log in logs_flat]
        for grad0, grad1 in itertools.combinations(logs_lin1_bias, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

    def test_hooks_cleaned_up_after_fit(self, doctor, data):
        # make sure that the hooks are cleaned up by checking that no more logs
        # are written when continuing to fit the net
        num_activation_logs_before = len(list(
            doctor.flatten(doctor.activation_logs_['module'])))
        num_gradient_logs_before = len(list(
            doctor.flatten(doctor.gradient_logs_['module'])))

        net = doctor.net
        net.partial_fit(*data)

        num_activation_logs_after = len(list(
            doctor.flatten(doctor.activation_logs_['module'])))
        num_gradient_logs_after = len(list(
            doctor.flatten(doctor.gradient_logs_['module'])))

        assert num_activation_logs_before == num_activation_logs_after
        assert num_gradient_logs_before == num_gradient_logs_after

    def test_callbacks_cleaned_up_after_fit(self, doctor, net_cls, module_cls):
        # make sure that the callbacks are the same before and after, this is
        # important because SkorchDoctor will temporarily add a callback
        net_without_doctor = net_cls(module_cls).initialize()
        callbacks_without_doctor = net_without_doctor.callbacks_
        callbacks_with_doctor = doctor.net.callbacks_

        assert len(callbacks_without_doctor) == len(callbacks_with_doctor)

        for (name0, cb0), (name1, cb1) in zip(
                callbacks_without_doctor, callbacks_with_doctor
        ):
            assert name0 == name1
            # pylint: disable=unidiomatic-typecheck
            assert type(cb0) == type(cb1)

    def test_get_layer_names(self, doctor):
        layer_names = doctor.get_layer_names()
        expected = {
            'criterion': [],
            'module': ['lin0', 'lin1', 'softmax']
        }
        assert layer_names == expected

    def test_get_parameter_names(self, doctor):
        param_names = doctor.get_param_names()
        expected = {
            'criterion': [],
            'module': ['lin0.weight', 'lin0.bias', 'lin1.weight', 'lin1.bias'],
        }
        assert param_names == expected

    def test_predict(self, doctor, data):
        X, _ = data
        y_pred_doctor = doctor.predict(X)
        y_pred_net = doctor.net.predict(X)
        np.testing.assert_allclose(y_pred_doctor, y_pred_net)

    def test_predict_proba(self, doctor, data):
        X, _ = data
        y_proba_doctor = doctor.predict_proba(X)
        y_proba_net = doctor.net.predict_proba(X)
        np.testing.assert_allclose(y_proba_doctor, y_proba_net)

    def test_score(self, doctor, data):
        X, y = data
        score_doctor = doctor.score(X, y)
        score_net = doctor.net.score(X, y)
        np.testing.assert_allclose(score_doctor, score_net)

    ############
    # PLOTTING #
    ############

    # Just do very basic plotting tests, not exact content, just that it works

    @pytest.fixture(scope='module')
    def plt(self):
        """Skip matplotlib tests if not installed when using this fixture"""
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use("agg")

        import matplotlib.pyplot as plt
        # not sure why closing is important but sklearn does it:
        # https://github.com/scikit-learn/scikit-learn/blob/964189df31dd2aa037c5bc58c96f88193f61253b/sklearn/conftest.py#L193
        plt.close("all")
        yield plt
        plt.close("all")

    # pylint: disable=unused-argument
    def test_plot_not_fitted_raises(self, plt, doctor_cls, net_cls, module_cls):
        # testing only one of the plotting functions, but all support it
        from skorch.exceptions import NotInitializedError

        doctor = doctor_cls(net_cls(module_cls))
        msg = (
            r"SkorchDoctor is not initialized yet. Call 'fit\(X, y\) before using this "
            "method."
        )
        with pytest.raises(NotInitializedError, match=msg):
            doctor.plot_loss()

    def test_plot_loss_default(self, plt, doctor):
        ax = doctor.plot_loss()
        assert isinstance(ax, plt.Subplot)

    def test_plot_loss_non_default(self, plt, doctor):
        _, ax = plt.subplots()
        ax_after = doctor.plot_loss(ax=ax, figsize=(1, 2), lw=5)
        assert isinstance(ax, plt.Subplot)
        assert ax_after is ax

    def test_plot_activations_default(self, plt, doctor):
        axes = doctor.plot_activations()
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 1)
        assert isinstance(axes[0, 0], plt.Subplot)

    def test_plot_activations_non_default(self, plt, doctor):
        _, axes = plt.subplots(1, 1, squeeze=False)
        axes_after = doctor.plot_activations(
            axes=axes,
            step=0,
            match_fn=lambda name: 'lin' in name,
            histtype='bar',
            lw=3,
            bins=np.arange(10),
            density=False,
            figsize=(1, 2),
            align='left',
        )
        assert axes_after is axes

    # pylint: disable=unused-argument
    def test_plot_activations_no_match(self, plt, doctor):
        msg = (
            r"No layer found matching the specification of match_fn. "
            r"Use doctor.get_layer_names\(\) to check all layers."
        )

        with pytest.raises(ValueError, match=msg):
            doctor.plot_activations(match_fn=lambda name: 'foo' in name)

    def test_plot_gradients_default(self, plt, doctor):
        axes = doctor.plot_gradients()
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 1)
        assert isinstance(axes[0, 0], plt.Subplot)

    def test_plot_gradients_non_default(self, plt, doctor):
        _, axes = plt.subplots(1, 1, squeeze=False)
        axes_after = doctor.plot_gradients(
            axes=axes,
            step=0,
            match_fn=lambda name: 'lin' in name,
            histtype='bar',
            lw=3,
            bins=np.arange(10),
            density=False,
            figsize=(1, 2),
            align='left',
        )
        assert axes_after is axes

    # pylint: disable=unused-argument
    def test_plot_gradients_no_match(self, plt, doctor):
        msg = (
            r"No parameter found matching the specification of match_fn. "
            r"Use doctor.get_param_names\(\) to check all parameters."
        )

        with pytest.raises(ValueError, match=msg):
            doctor.plot_gradients(match_fn=lambda name: 'foo' in name)

    def test_plot_param_updates_default(self, plt, doctor):
        axes = doctor.plot_param_updates()
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (1, 1)
        assert isinstance(axes[0, 0], plt.Subplot)

    def test_plot_param_updates_non_default(self, plt, doctor):
        _, axes = plt.subplots(1, 1, squeeze=False)
        axes_after = doctor.plot_param_updates(
            axes=axes,
            match_fn=lambda name: 'lin' in name,
            lw=3,
            figsize=(1, 2),
        )
        assert isinstance(axes, np.ndarray)
        assert axes_after is axes

    # pylint: disable=unused-argument
    def test_plot_param_updates_no_match(self, plt, doctor):
        msg = (
            r"No parameter found matching the specification of match_fn. "
            r"Use doctor.get_param_names\(\) to check all parameters."
        )

        with pytest.raises(ValueError, match=msg):
            doctor.plot_param_updates(match_fn=lambda name: 'foo' in name)

    def test_plot_activations_over_time_default(self, plt, doctor):
        ax = doctor.plot_activations_over_time(layer_name='lin0')
        assert isinstance(ax, plt.Subplot)

    def test_plot_activations_over_time_non_default(self, plt, doctor):
        _, ax = plt.subplots()
        ax_after = doctor.plot_activations_over_time(
            layer_name='softmax',
            ax=ax,
            lw=3,
            bins=np.arange(10),
            color='r',
            figsize=(1, 2),
            interpolate=True,
        )
        assert ax_after is ax

    # pylint: disable=unused-argument
    def test_plot_activations_over_time_no_match(self, plt, doctor):
        msg = (
            r"No layer named 'foo' could be found. "
            r"Use doctor.get_layer_names\(\) to check all layers."
        )

        with pytest.raises(ValueError, match=msg):
            doctor.plot_activations_over_time(layer_name='foo')

    def test_plot_gradient_over_time_default(self, plt, doctor):
        ax = doctor.plot_gradient_over_time(param_name='lin1.weight')
        assert isinstance(ax, plt.Subplot)

    def test_plot_gradient_over_time_non_default(self, plt, doctor):
        _, ax = plt.subplots()
        ax_after = doctor.plot_gradient_over_time(
            param_name='lin0.bias',
            ax=ax,
            lw=3,
            bins=np.arange(10),
            color='r',
            figsize=(1, 2),
            interpolate=True,
        )
        assert ax_after is ax

    # pylint: disable=unused-argument
    def test_plot_gradient_over_time_no_match(self, plt, doctor):
        msg = (
            r"No parameter named 'foo' could be found. "
            r"Use doctor.get_param_names\(\) to check all parameters."
        )

        with pytest.raises(ValueError, match=msg):
            doctor.plot_gradient_over_time(param_name='foo')


class TestSkorchDoctorComplexArchitecture:
    @pytest.fixture(scope='module')
    def module0_cls(self):
        """Module that returns a tuple"""
        class MyModule(nn.Module):
            """Module that returns a tuple, lin0 no grad"""
            def __init__(self):
                super().__init__()

                self.lin0 = nn.Linear(20, 20)
                self.lin0.requires_grad_(False)
                self.lin1 = nn.Linear(20, 2)

            def forward(self, X):
                X0 = self.lin0(X)
                X1 = self.lin1(X)
                return X0, X1

        return MyModule

    @pytest.fixture(scope='module')
    def module1_cls(self):
        """Module without learnable params that returns a dict"""
        class MyModule(nn.Module):
            """Module that returns a dict"""
            def __init__(self):
                super().__init__()
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, X):
                softmax = self.softmax(X)
                return {'logits': X, 'softmax': softmax}

        return MyModule

    @pytest.fixture(scope='module')
    def criterion_cls(self):
        class MyCriterion(nn.NLLLoss):
            """Criterion that has learnable parameters"""
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(2, 2)

            # pylint: disable=arguments-differ
            def forward(self, y_proba, y):
                y_proba = self.lin0(y_proba)
                return nn.functional.nll_loss(y_proba, y)

        return MyCriterion

    @pytest.fixture(scope='module')
    def net_cls(self, module0_cls, module1_cls):
        """Customize net to work with complex modules"""
        from skorch import NeuralNetClassifier
        from skorch.utils import to_tensor

        class MyNet(NeuralNetClassifier):
            """Customize net to work with complex modules"""
            def initialize_module(self):
                kwargs = self.get_params_for('module0')
                module = self.initialized_instance(module0_cls, kwargs)
                # pylint: disable=attribute-defined-outside-init
                self.module0_ = module

                kwargs = self.get_params_for('module1')
                module = self.initialized_instance(module1_cls, kwargs)
                # pylint: disable=attribute-defined-outside-init
                self.module1_ = module

                return self

            def initialize_criterion(self):
                # use non-standard name 'criterion0'
                kwargs = self.get_params_for('criterion0')
                criterion = self.initialized_instance(self.criterion, kwargs)
                # pylint: disable=attribute-defined-outside-init
                self.criterion0_ = criterion
                return self

            def infer(self, x, **fit_params):
                x = to_tensor(x, device=self.device)
                _, X1 = self.module0_(x, **fit_params)
                output = self.module1_(X1)
                return output['softmax']

            def get_loss(self, y_pred, y_true, *args, **kwargs):
                y_true = to_tensor(y_true, device=self.device)
                return self.criterion0_(y_pred, y_true)

        return MyNet

    @pytest.fixture(scope='module')
    def doctor_cls(self):
        from skorch.helper import SkorchDoctor
        return SkorchDoctor

    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        X, y = classifier_data
        # a small amount of data is enough
        return X[:50], y[:50]

    @pytest.fixture(scope='module')
    def doctor(self, module0_cls, criterion_cls, net_cls, doctor_cls, data):
        # the passed module doesn't matter as they are hard-coded
        torch.manual_seed(0)
        net = net_cls(
            module0_cls, criterion=criterion_cls, max_epochs=3, batch_size=32
        )
        doctor = doctor_cls(net)
        doctor.fit(*data)
        return doctor

    def test_it(self, doctor):
        breakpoint()
