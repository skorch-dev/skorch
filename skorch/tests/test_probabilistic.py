"""Tests for probabilistic.py"""

import pickle
import re

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import torch

from skorch.utils import is_torch_data_type


gpytorch = pytest.importorskip('gpytorch')


class RbfModule(gpytorch.models.ExactGP):
    """Defined on root to make it pickleable"""
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalRegressionModule(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, eps=1e-6):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(eps=eps))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalBinaryClassificationModule(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, eps=1e-6):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(eps=eps))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class BaseProbabilisticTests:
    ##########################
    # constants and fixtures #
    ##########################

    @property
    def n_samples(self):
        # expects int
        return NotImplementedError

    @property
    def n_targets(self):
        # expects int
        raise NotImplementedError

    @property
    def supports_predict_proba(self):
        # expects bool
        raise NotImplementedError

    @property
    def supports_return_std(self):
        # expects bool
        # This only checks if the argument is allowed by predict, not whether it
        # actually implements a solution
        raise NotImplementedError

    @property
    def supports_return_cov(self):
        # expects bool
        # This only checks if the argument is allowed by predict, not whether it
        # actually implements a solution
        raise NotImplementedError

    @property
    def settable_params(self):
        # expects dict of parameters that can be set with set_params
        raise NotImplementedError

    @property
    def scoring(self):
        # the default scoring function of this estimator, must be sklearn
        # compatible
        raise NotImplementedError

    @pytest.fixture
    def gp_cls(self):
        raise NotImplementedError

    @pytest.fixture
    def module_cls(self):
        raise NotImplementedError

    @pytest.fixture
    def data(self):
        raise NotImplementedError

    @pytest.fixture
    def gp(self, gp_cls, data):
        raise NotImplementedError

    @pytest.fixture
    def gp_fit(self, gp, data):
        X, y = data
        return gp.fit(X, y)

    @pytest.fixture
    def pipe(self, gp):
        return Pipeline([
            ('noop', None),
            ('gp', gp),
        ])

    ######################
    # saving and loading #
    ######################

    @pytest.mark.xfail(strict=True)
    def test_pickling(self, gp_fit):
        # Currently fails becaues of issues outside of our control, this test
        # should alert us to when the issue has been fixed
        pickle.dumps(gp_fit)

    def test_pickle_error_msg(self, gp_fit):
        # Should eventually be replaced by a test that saves and loads the model
        # using pickle or deepcopy and checks that the predictions are identical
        msg = "TODO"
        with pytest.raises(pickle.PicklingError, match=msg):
            pickle.dumps(gp_fit)

    def test_save_load_params(self, gp_fit):
        pass

    ##############
    # functional #
    ##############

    def test_fit(self, gp_fit):
        # fitting does not raise anything
        pass

    def test_gp_learns(self, gp_fit):
        history = gp_fit.history
        assert history[0, 'train_loss'] > 0.5 * history[-1, 'train_loss']

    def test_forward(self, gp_fit, data):
        X = data[0]
        y_forward = gp_fit.forward(X)

        for yi in y_forward:
            assert isinstance(yi, torch.distributions.distribution.Distribution)

        total_shape = sum(p.shape()[0] for p in y_forward)
        assert total_shape == self.n_samples

    def test_predict(self, gp_fit, data):
        X = data[0]
        y_pred = gp_fit.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (self.n_samples,)

    def test_predict_proba(self, gp_fit, data):
        if not self.supports_predict_proba:
            return

        X = data[0]
        y_proba = gp_fit.predict_proba(X)

        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (self.n_samples, self.n_targets)

    def test_fit_and_predict_with_cuda(self, gp, data):
        if not torch.cuda.is_available():
            pytest.skip()

        gp.set_params(device='cuda')
        X, y = data
        gp.fit(X, y)
        gp.predict(X)

    @pytest.mark.skip
    def test_multioutput_forward_iter(self, gp_multiouput, data):
        X = data[0]
        y_infer = next(gp_multiouput.forward_iter(X))

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 3
        assert y_infer[0].shape[0] == min(len(X), gp_multiouput.batch_size)

    @pytest.mark.skip
    def test_multioutput_forward(self, gp_multiouput, data):
        X = data[0]
        y_infer = gp_multiouput.forward(X)

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 3
        for arr in y_infer:
            assert is_torch_data_type(arr)

        for output in y_infer:
            assert len(output) == self.n_samples

    @pytest.mark.skip
    def test_multioutput_predict(self, gp_multiouput, data):
        X = data[0]

        # does not raise
        y_pred = gp_multiouput.predict(X)

        # Expecting only 1 column containing predict class:
        # (number of samples,)
        assert y_pred.shape == (self.n_samples)

    @pytest.mark.skip
    def test_multiouput_predict_proba(self, gp_multiouput, data):
        X = data[0]

        # does not raise
        y_proba = gp_multiouput.predict_proba(X)

        # Expecting full output: (number of samples, number of output units)
        assert y_proba.shape == (self.n_samples, self.n_targets)
        # Probabilities, hence these limits
        assert y_proba.min() >= 0
        assert y_proba.max() <= 1

    def test_in_sklearn_pipeline(self, pipe, data):
        X, y = data
        # none of this raises an error
        pipe.fit(X, y)
        pipe.predict(X)
        pipe.set_params(**self.settable_params)

    def test_grid_search_works(self, gp, data, recwarn):
        X, y = data
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
            'likelihood__max_plate_nesting': [1, 2],
        }
        gs = GridSearchCV(gp, params, refit=True, cv=3, scoring=self.scoring)
        gs.fit(X[:60], y[:60])  # for speed

        # sklearn will catch fit failures and raise a warning, we should thus
        # check that no warnings are generated
        assert not recwarn.list

    ##################
    # initialization #
    ##################

    @pytest.mark.parametrize('kwargs,expected', [
        ({}, ""),
        ({
            'likelihood__noise_prior': gpytorch.priors.NormalPrior(0, 1)
            , 'likelihood__batch_shape': (345,),
        }, ""),
        ({
            'likelihood__noise_prior': gpytorch.priors.NormalPrior(0, 1),
            'optimizer__momentum': 0.567,
        }, ""),
    ])
    def test_set_params_uninitialized_net_correct_message(
            self, gp, kwargs, expected, capsys):
        # When gp is initialized, if module or optimizer need to be
        # re-initialized, alert the user to the fact what parameters
        # were responsible for re-initialization. Note that when the
        # module parameters but not optimizer parameters were changed,
        # the optimizer is re-initialized but not because the
        # optimizer parameters changed.
        gp.set_params(**kwargs)
        msg = capsys.readouterr()[0].strip()
        assert msg == expected

    @pytest.mark.skip
    @pytest.mark.parametrize('kwargs,expected', [
        ({}, ""),
        (
            {
                'likelihood__noise_prior': gpytorch.priors.NormalPrior(0, 1),
                'likelihood__batch_shape': (345,),
            },
            ("Re-initializing module because the following "
             "parameters were re-set: noise_prior, batch_shape.\n"
             "Re-initializing criterion.\n"
             "Re-initializing optimizer.")
        ),
        (
            {
                'likelihood__noise_prior': gpytorch.priors.NormalPrior(0, 1),
                'optimizer__momentum': 0.567,
            },
            ("Re-initializing module because the following "
             "parameters were re-set: noise_prior.\n"
             "Re-initializing criterion.\n"
             "Re-initializing optimizer.")
        ),
    ])
    def test_set_params_initialized_net_correct_message(
            self, gp, kwargs, expected, capsys):
        # When gp is initialized, if module or optimizer need to be
        # re-initialized, alert the user to the fact what parameters
        # were responsible for re-initialization. Note that when the
        # module parameters but not optimizer parameters were changed,
        # the optimizer is re-initialized but not because the
        # optimizer parameters changed.
        gp.initialize().set_params(**kwargs)
        msg = capsys.readouterr()[0].strip()
        assert msg == expected

    ##########################
    # probabalistic specific #
    ##########################

    @pytest.mark.parametrize("n_samples", [1, 2, 10])
    def test_sampling(self, gp, data, n_samples):
        X, _ = data
        samples = gp.initialize().sample(X, n_samples=n_samples)
        assert samples.shape == (n_samples, len(X))

    def test_confidence_region(self, gp_fit, data):
        X, _ = data

        # lower bound should always be lower than upper bound
        lower_1, upper_1 = gp_fit.confidence_region(X, sigmas=1)
        assert (lower_1 < upper_1).all()

        lower_2, upper_2 = gp_fit.confidence_region(X, sigmas=2)
        assert (lower_2 < upper_2).all()

        # higher sigmas -> wider regions
        assert (lower_2 < lower_1).all()
        assert (upper_2 > upper_1).all()

    def test_predict_return_std(self, gp_fit, data):
        if not self.supports_return_std:
            return

        X, _ = data
        y_proba, y_std = gp_fit.predict(X, return_std=True)

        # not a lot we know for sure about the values of the standard deviation,
        # hence only test shape and that they're positive
        assert y_proba.shape == y_std.shape
        assert (y_std > 0).all()

    def test_predict_return_cov(self, gp_fit, data):
        if not self.supports_return_cov:
            return

        X, _ = data
        msg = ("The 'return_cov' argument is not supported. Please try: "
               "'posterior = next(gpr.forward_iter(X)); posterior.covariance_matrix'.")
        with pytest.raises(NotImplementedError, match=re.escape(msg)):
            gp_fit.predict(X, return_cov=True)


class TestExactGPRegressor(BaseProbabilisticTests):
    ##########################
    # constants and fixtures #
    ##########################

    n_samples = 34
    n_targets = 1
    supports_predict_proba = False
    supports_return_std = True
    supports_return_cov = True
    settable_params = {'gp__module__y': torch.zeros(n_samples).float()}
    scoring = 'neg_mean_squared_error'

    @pytest.fixture
    def data(self):
        X = np.linspace(-8, 8.01, self.n_samples).astype(np.float32)
        y = (np.sin(X) + np.random.randn(len(X)) * 0.2).astype(np.float32)
        return X, y

    @pytest.fixture
    def gp_cls(self):
        from skorch.probabilistic import ExactGPRegressor
        return ExactGPRegressor

    @pytest.fixture
    def module_cls(self):
        return RbfModule

    @pytest.fixture
    def gp(self, gp_cls, module_cls, data):
        X, y = data
        gpr = gp_cls(
            module_cls,
            module__X=torch.from_numpy(X),
            module__y=torch.from_numpy(y),

            optimizer=torch.optim.Adam,
            lr=0.1,
            max_epochs=20,
            batch_size=-1,
        )
        return gpr

    # grid search currently doesn't work with ExactGP because each grid search
    # fit uses a different split of data but ExactGP is initialized with a fixed
    # X and y that is not allowed to change
    test_grid_search_works = pytest.mark.xfail(
        BaseProbabilisticTests.test_grid_search_works, strict=True,
    )

    def test_grid_search_works_with_debug_turned_off(self, gp, data, recwarn):
        with gpytorch.settings.debug(False):
            X, y = data
            params = {
                'lr': [0.01, 0.02],
                'max_epochs': [10, 20],
                'likelihood__noise_prior': [None, gpytorch.priors.GammaPrior(1, 2)],
            }
            gs = GridSearchCV(gp, params, refit=True, cv=3, scoring=self.scoring)
            gs.fit(X[:60], y[:60])  # for speed

            # sklearn will catch fit failures and raise a warning, we should thus
            # check that no warnings are generated
            assert not recwarn.list

    def test_wrong_module_type_raises(self, gp_cls):
        # ExactGPRegressor requires the module to be an ExactGP, if it's not,
        # raise an appropriate error message to the user.
        class VariationalModule(gpytorch.models.ApproximateGP):
            """Defined on root to make it pickleable"""
            def __init__(self, likelihood):
                pass

            def forward(self, x):
                pass

        gp = gp_cls(VariationalModule)
        msg = "ExactGPRegressor requires 'module' to be a gpytorch.models.ExactGP."
        with pytest.raises(TypeError, match=msg):
            gp.initialize()


class TestGPRegressorVariational(BaseProbabilisticTests):
    ##########################
    # constants and fixtures #
    ##########################

    n_samples = 60
    n_targets = 1
    supports_predict_proba = False
    supports_return_std = True
    supports_return_cov = True
    settable_params = {'gp__module__eps': 1e-5}
    scoring = 'neg_mean_squared_error'

    @pytest.fixture
    def data(self):
        X = np.linspace(-8, 8.01, self.n_samples).astype(np.float32)
        y = (np.sin(X) + np.random.randn(len(X)) * 0.2).astype(np.float32)
        return X, y

    @pytest.fixture
    def gp_cls(self):
        from skorch.probabilistic import GPRegressor
        return GPRegressor

    @pytest.fixture
    def module_cls(self):
        return VariationalRegressionModule

    @pytest.fixture
    def gp(self, gp_cls, module_cls, data):
        X, y = data
        gpr = gp_cls(
            module_cls,
            module__inducing_points=torch.from_numpy(X[:10]),

            criterion=gpytorch.mlls.VariationalELBO,
            criterion__num_data=int(0.8 * len(y)),
            batch_size=24,
        )
        # we want to make sure batching is properly tested
        assert gpr.batch_size < self.n_samples
        return gpr


class TestGPBinaryClassifier(BaseProbabilisticTests):
    ##########################
    # constants and fixtures #
    ##########################

    n_samples = 50
    n_targets = 2
    supports_predict_proba = True
    supports_return_std = False
    supports_return_cov = False
    settable_params = {'gp__module__eps': 1e-5}
    scoring = 'neg_mean_squared_error'

    @pytest.fixture
    def data(self):
        X = np.linspace(-8, 8.01, self.n_samples).astype(np.float32)
        y = (np.sin(X) + np.random.randn(len(X)) * 0.2 > 0).astype(np.int64)
        return X, y

    @pytest.fixture
    def gp_cls(self):
        from skorch.probabilistic import GPBinaryClassifier
        return GPBinaryClassifier

    @pytest.fixture
    def module_cls(self):
        return VariationalBinaryClassificationModule

    @pytest.fixture
    def gp(self, gp_cls, module_cls, data):
        X, y = data
        gpc = gp_cls(
            module_cls,
            module__inducing_points=torch.from_numpy(X[:10]),

            criterion=gpytorch.mlls.VariationalELBO,
            criterion__num_data=int(0.8 * len(y)),
            batch_size=24,
        )
        # we want to make sure batching is properly tested
        assert gpc.batch_size < self.n_samples
        return gpc
