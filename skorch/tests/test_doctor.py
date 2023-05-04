"""Tests for skorch._doctor.py"""

import itertools
from unittest import mock

import numpy as np
import pytest
import torch
from torch import nn


class TestSkorchDoctorSimple:  # pylint: disable=too-many-public-methods
    """Test functionality of SkorchDoctor using a simple model"""
    @pytest.fixture(scope='module')
    def module_cls(self):
        """Return a simple module class with predictable parameters"""
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
        """Split train/valid deterministically so that we know all training
        samples"""
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

    def test_activation_recs_general_content(self, doctor):
        recs = doctor.activation_recs_
        assert set(recs.keys()) == {'module', 'criterion'}

        # nothing recorded for criterion
        assert recs['criterion'] == []

        recs_module = recs['module']
        # 3 epochs, 2 batches per epoch
        assert len(recs_module) == 6

        # each batch has layers lin0, lin1, softmax
        for batch in recs_module:
            assert set(batch.keys()) == {'lin0', 'lin1', 'softmax'}

    def test_activation_recs_values(self, doctor, data):
        recs_module = doctor.activation_recs_['module']

        for key in ('lin0', 'lin1', 'softmax'):
            # 80% of 50 samples is 40, batch size 32 => 32 + 8 samples per batch
            batch_sizes = [len(batch[key]) for batch in recs_module]
            assert batch_sizes == [32, 8, 32, 8, 32, 8]

        X, _ = data
        # for the very first batch, before any update, we actually know the values
        batch0 = recs_module[0]
        lin0_0 = batch0['lin0']
        # since it is the identity function, batches should equal the data
        np.testing.assert_array_almost_equal(lin0_0, X[:32])

        lin1_0 = batch0['lin1']
        # since weights are 0 and bias is 1, all values should be 1
        np.testing.assert_array_almost_equal(lin1_0, 1.0)

        softmax_0 = batch0['softmax']
        # since all inputs are equal, probabilities should be uniform
        np.testing.assert_array_almost_equal(softmax_0, 0.5)

    def test_activation_recs_not_all_identical(self, doctor):
        # make sure that values are not just all identical, using a large
        # tolerance to exclude small deviations
        recs = doctor.activation_recs_['module']

        recs_lin0 = [rec['lin0'] for rec in recs]
        for act0, act1 in itertools.combinations(recs_lin0, r=2):
            if act0.shape == act1.shape:
                assert not np.allclose(act0, act1, rtol=1e-3)

        recs_lin1 = [rec['lin1'] for rec in recs]
        for act0, act1 in itertools.combinations(recs_lin1, r=2):
            if act0.shape == act1.shape:
                assert not np.allclose(act0, act1, rtol=1e-3)

        softmax = [rec['softmax'] for rec in recs]
        for act0, act1 in itertools.combinations(softmax, r=2):
            if act0.shape == act1.shape:
                assert not np.allclose(act0, act1, rtol=1e-3)

    def test_gradient_recs_general_content(self, doctor):
        recs = doctor.gradient_recs_
        assert set(recs.keys()) == {'module', 'criterion'}

        # nothing recorded for criterion, 3 epochs
        assert recs['criterion'] == []

        recs_module = recs['module']
        # 3 epochs, 2 batches per epoch
        assert len(recs_module) == 6

        # each batch has weights and biases for lin0 & lin1
        expected = {'lin0.weight', 'lin0.bias', 'lin1.weight', 'lin1.bias'}
        for batch in recs_module:
            assert set(batch.keys()) == expected

    def test_gradient_recs_values(self, doctor):
        recs_module = doctor.gradient_recs_['module']

        expected_shapes = {
            'lin0.weight': (20, 20),
            'lin0.bias': (20,),
            'lin1.weight': (2, 20),
            'lin1.bias': (2,),
        }
        for key in ('lin0.weight', 'lin0.bias', 'lin1.weight', 'lin1.bias'):
            grad_shapes = [batch[key].shape for batch in recs_module]
            expected_shape = expected_shapes[key]
            # 2 batches, 3 epochs
            assert grad_shapes == [expected_shape] * 6

        # There is not really much we know about the gradient values, we just
        # rely on the gradient hooks doing what they're supposed to do. The only
        # gradients we actually know are for the first layer in the first batch:
        # They have to be zero because in the second layer, we have weights of 0.

        batch0 = recs_module[0]
        grad_weight = batch0['lin0.weight']
        grad_bias = batch0['lin0.bias']
        assert np.allclose(grad_weight, 0.0)
        assert np.allclose(grad_bias, 0.0)

    def test_gradient_recs_not_all_identical(self, doctor):
        # make sure that values are not just all identical, using a large
        # tolerance to exclude small deviations
        recs = doctor.gradient_recs_['module']

        recs_lin0_weight = [rec['lin0.weight'] for rec in recs]
        for grad0, grad1 in itertools.combinations(recs_lin0_weight, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

        recs_lin0_bias = [rec['lin0.bias'] for rec in recs]
        for grad0, grad1 in itertools.combinations(recs_lin0_bias, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

        recs_lin1_weight = [rec['lin1.weight'] for rec in recs]
        for grad0, grad1 in itertools.combinations(recs_lin1_weight, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

        recs_lin1_bias = [rec['lin1.bias'] for rec in recs]
        for grad0, grad1 in itertools.combinations(recs_lin1_bias, r=2):
            assert not np.allclose(grad0, grad1, rtol=1e-3)

    def test_param_update_recs_general_content(self, doctor):
        recs = doctor.param_update_recs_
        assert set(recs.keys()) == {'module', 'criterion'}

        # nothing recorded for criterion
        assert recs['criterion'] == []

        recs_module = recs['module']
        # 3 epochs, 2 batches per epoch
        assert len(recs_module) == 6

        # each batch has weights and biases for lin0 & lin1
        expected = {'lin0.weight', 'lin0.bias', 'lin1.weight', 'lin1.bias'}
        for batch in recs_module:
            assert set(batch.keys()) == expected

    def test_param_update_recs_values(self, doctor):
        recs= doctor.param_update_recs_['module']
        assert all(np.isscalar(val) for d in recs for val in d.values())

        # for the very first batch, before any update, we actually know that the
        # updates must be 0 because the gradients are 0.
        batch0 = recs[0]
        assert np.isclose(batch0['lin0.weight'], 0)
        assert np.isclose(batch0['lin0.bias'], 0)

    def test_param_update_recs_not_all_identical(self, doctor):
        # make sure that values are not just all identical, using a large
        # tolerance to exclude small deviations
        recs = doctor.param_update_recs_['module']

        recs_lin0_weight = [rec['lin0.weight'] for rec in recs]
        for upd0, upd1 in itertools.combinations(recs_lin0_weight, r=2):
            assert not np.isclose(upd0, upd1, rtol=1e-3)

        recs_lin0_bias = [rec['lin0.bias'] for rec in recs]
        for upd0, upd1 in itertools.combinations(recs_lin0_bias, r=2):
            assert not np.isclose(upd0, upd1, rtol=1e-3)

        recs_lin1_weight = [rec['lin1.weight'] for rec in recs]
        for upd0, upd1 in itertools.combinations(recs_lin1_weight, r=2):
            assert not np.isclose(upd0, upd1, rtol=1e-3)

        recs_lin1_bias = [rec['lin1.bias'] for rec in recs]
        for upd0, upd1 in itertools.combinations(recs_lin1_bias, r=2):
            assert not np.isclose(upd0, upd1, rtol=1e-3)

    def test_hooks_cleaned_up_after_fit(self, doctor, data):
        # make sure that the hooks are cleaned up by checking that no more recs
        # are written when continuing to fit the net
        num_activation_recs_before = len(doctor.activation_recs_['module'])
        num_gradient_recs_before = len(doctor.gradient_recs_['module'])

        net = doctor.net
        net.partial_fit(*data)

        num_activation_recs_after = len(doctor.activation_recs_['module'])
        num_gradient_recs_after = len(doctor.gradient_recs_['module'])

        assert num_activation_recs_before == num_activation_recs_after
        assert num_gradient_recs_before == num_gradient_recs_after

    def test_callbacks_cleaned_up_after_fit_with_initial_callbacks(
            self, doctor_cls, net_cls, module_cls, data
    ):
        # make sure that the callbacks are the same before and after, this is
        # important because SkorchDoctor will temporarily add a callback
        from skorch.callbacks import EpochScoring, GradientNormClipping

        net = net_cls(
            module_cls,
            callbacks=[EpochScoring('f1'), GradientNormClipping(1.0)],
        ).initialize()
        callbacks_without_doctor = net.callbacks_[:]

        doctor = doctor_cls(net).fit(*data)
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

    def test_recs_with_filter(self, module_cls, net_cls, doctor_cls, data, custom_split):
        # when initializing SkorchDoctor with a match_fn, only records whose
        # keys match should be kept
        def match_fn(name):
            return "lin0" in name

        net = net_cls(module_cls, max_epochs=3, batch_size=32, train_split=custom_split)
        doctor = doctor_cls(net, match_fn=match_fn)
        doctor.fit(*data)

        # check trivial case of empty lists
        assert doctor.activation_recs_['module']
        assert doctor.gradient_recs_['module']
        assert doctor.param_update_recs_['module']

        for rec in doctor.activation_recs_['module']:
            for key in rec.keys():
                assert match_fn(key)

        for rec in doctor.gradient_recs_['module']:
            for key in rec.keys():
                assert match_fn(key)

        for rec in doctor.param_update_recs_['module']:
            for key in rec.keys():
                assert match_fn(key)

    def test_recs_with_filter_no_match(
            self, module_cls, net_cls, doctor_cls, data, custom_split
    ):
        # raise a helpful error if the match function filters away everything
        def match_fn(name):
            return "this-substring-does-not-exist" in name

        net = net_cls(module_cls, max_epochs=3, batch_size=32, train_split=custom_split)
        doctor = doctor_cls(net, match_fn=match_fn)

        msg = (
            "No activations, gradients, or updates are being recorded, "
            "please check the match_fn"
        )
        with pytest.raises(ValueError, match=msg):
            doctor.fit(*data)

    ############
    # PLOTTING #
    ############

    # Just do very basic plotting tests, not exact content, just that it works

    @pytest.fixture
    def mock_matplotlib_not_installed(self):
        # fixture to make it seem like matplotlib was not installed
        orig_import = __import__

        def import_mock(name, *args):
            if name == 'matplotlib':
                # pretend that matplotlib is not installed
                raise ModuleNotFoundError("no module named 'matplotlib'")
            return orig_import(name, *args)

        with mock.patch('builtins.__import__', side_effect=import_mock):
            yield import_mock

    # pylint: disable=unused-argument
    def test_matplotlib_not_installed(self, mock_matplotlib_not_installed, doctor):
        # Note: Unfortunately, the order of tests matters here: This test should
        # run before the ones below that use matplotlib, otherwise the import
        # mock doesn't work correctly.
        msg = (
            r"This feature requires matplotlib to be installed; "
            r"please install it first, e.g. using "
            r"\'python -m pip install matplotlib\'"
        )
        with pytest.raises(ImportError, match=msg):
            doctor.plot_loss()

    @pytest.fixture(scope='module')
    def plt(self):
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

    def test_plot_activations_passed_axes_2d(self, plt, doctor):
        _, axes = plt.subplots(2, 2)
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
    """Tests based on a more non-standard model

    Specifically, add modules with non-standard names and non-standard outputs
    like tuples or dicts.

    This test class does not re-iterate all the tests performed on the standard
    model but focuses on the parts that change, e.g. how the outputs are
    recorded.

    """
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
    def module2_cls(self, module0_cls, module1_cls):
        """Module that combines module0 and module1"""
        class MyModule(nn.Module):
            """Module that returns a dict"""
            def __init__(self):
                super().__init__()
                self.module0 = module0_cls()
                self.module1 = module1_cls()

            def forward(self, X):
                _, X1 = self.module0(X)
                output = self.module1(X1)
                return output['softmax']

        return MyModule

    @pytest.fixture(scope='module')
    def criterion_cls(self):
        class MyCriterion(nn.Module):
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
    def net_cls(self, module2_cls):
        """Customize net to work with complex modules"""
        from skorch import NeuralNetClassifier
        from skorch.utils import to_tensor

        class MyNet(NeuralNetClassifier):
            """Customize net that works with non-standard modules"""
            def initialize_module(self):
                kwargs = self.get_params_for('mymodule')
                module = self.initialized_instance(module2_cls, kwargs)
                # pylint: disable=attribute-defined-outside-init
                self.mymodule_ = module
                self.seq_ = nn.Sequential(nn.Linear(2, 2))

                return self

            def initialize_criterion(self):
                # use non-standard name 'mycriterion'
                kwargs = self.get_params_for('mycriterion')
                criterion = self.initialized_instance(self.criterion, kwargs)
                # pylint: disable=attribute-defined-outside-init
                self.mycriterion_ = criterion
                return self

            def infer(self, x, **fit_params):
                x = to_tensor(x, device=self.device)
                x = self.mymodule_(x, **fit_params)
                x = x + self.seq_(x)
                return x

            def get_loss(self, y_pred, y_true, *args, **kwargs):
                y_true = to_tensor(y_true, device=self.device)
                return self.mycriterion_(y_pred, y_true)

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

    def test_activation_recs_general_content(self, doctor):
        recs = doctor.activation_recs_
        assert set(recs.keys()) == {'mymodule', 'seq', 'mycriterion'}

        for rec in recs.values():
            # 3 epochs, 2 batches per epoch
            assert len(rec) == 6

        expected_mymodule = {
            'module0.lin0', 'module0.lin1', 'module0[0]', 'module0[1]',
            'module1.softmax', 'module1["logits"]', 'module1["softmax"]',
        }
        assert set(recs['mymodule'][0].keys()) == expected_mymodule
        # nn.Sequential just enumerates the layers
        assert set(recs['seq'][0].keys()) == {'0'}
        assert set(recs['mycriterion'][0].keys()) == {'lin0'}

    def test_gradient_recs_general_content(self, doctor):
        recs = doctor.gradient_recs_
        assert len(recs) == 3
        assert set(recs.keys()) == {'mymodule', 'seq', 'mycriterion'}

        for rec in recs.values():
            # 3 epochs, 2 batches per epoch
            assert len(rec) == 6

        # each batch has weights and biases for lin1, lin0 has no gradient
        expected = {'module0.lin1.weight', 'module0.lin1.bias'}
        for batch in recs['mymodule']:
            assert set(batch.keys()) == expected

        # each batch has weights and biases for lin1, lin0 has no gradient
        expected = {'0.weight', '0.bias'}
        for batch in recs['seq']:
            assert set(batch.keys()) == expected

        # each batch has weights and biases for lin1, lin0 has no gradient
        expected = {'lin0.weight', 'lin0.bias'}
        for batch in recs['mycriterion']:
            assert set(batch.keys()) == expected

    def test_get_layer_names(self, doctor):
        layer_names = doctor.get_layer_names()
        expected = {
            'mymodule': [
                'module0.lin0', 'module0.lin1', 'module0[0]', 'module0[1]',
                'module1.softmax', 'module1["logits"]', 'module1["softmax"]',
            ],
            'seq': ['0'],
            'mycriterion': ['lin0'],
        }
        assert layer_names == expected

    def test_get_parameter_names(self, doctor):
        param_names = doctor.get_param_names()
        expected = {
            'mymodule': ['module0.lin1.weight', 'module0.lin1.bias'],
            'seq': ['0.weight', '0.bias'],
            'mycriterion': ['lin0.weight', 'lin0.bias'],
        }
        assert param_names == expected
