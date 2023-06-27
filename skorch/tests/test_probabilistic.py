"""Tests for probabilistic.py"""

import copy
import pickle
import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import torch

from skorch._version import Version
from skorch.utils import is_torch_data_type
from skorch.utils import to_numpy


try:
    gpytorch = pytest.importorskip('gpytorch')
except AttributeError:
    pytest.skip("Incompatible gpytorch + torch version", allow_module_level=True)

# check that torch version is sufficiently high for gpytorch, otherwise skip
version_gpytorch = Version(gpytorch.__version__)
version_torch = Version(torch.__version__)
# TODO: remove if newer GPyTorch versions are released that no longer require
# the check.
if (version_gpytorch >= Version('1.9')) and (version_torch < Version('1.11')):
    pytest.skip("Incompatible gpytorch + torch version", allow_module_level=True)
elif (version_gpytorch >= Version('1.7')) and (version_torch < Version('1.10')):
    pytest.skip("Incompatible gpytorch + torch version", allow_module_level=True)


def get_batch_size(dist):
    """Return the shape of the distribution

    The method/attribute required to determine the shape depends on the kind of
    distrubtion.

    """
    shape = getattr(dist, 'shape', None)
    if shape:
        return shape[0]

    get_base_samples = getattr(dist, 'get_base_samples', None)
    if get_base_samples:
        return get_base_samples().shape[0]

    shape = getattr(dist, 'batch_shape', None)
    if shape:
        return shape[0]

    raise AttributeError(f"Could not determine shape of {dist}")


# PyTorch Modules are defined on the module root to make them pickleable.

class RbfModule(gpytorch.models.ExactGP):
    """Simple exact GP regression module"""
    def __init__(self, likelihood):
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalRegressionModule(gpytorch.models.ApproximateGP):
    """GP regression for variational inference"""
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
    """GP classification for variational inference"""
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


class MyBernoulliLikelihood(gpytorch.likelihoods.BernoulliLikelihood):
    """This class only exists to add a param to BernoulliLikelihood

    BernoulliLikelihood used to have parameters before gpytorch v1.10, but now
    it does not have any parameters anymore. This is not an issue per se, but
    there are a few things we cannot test anymore, e.g. that parameters are
    passed to the likelihood correctly when using grid search. Therefore, create
    a custom class with a (pointless) parameter.

    """
    def __init__(self, *args, some_parameter=1, **kwargs):
        self.some_parameter = some_parameter
        super().__init__(*args, **kwargs)


class BaseProbabilisticTests:
    """Base class for all GP estimators.

    This class defined all fixtures, most of which need to be implemented by the
    respective subclass, as well as all the tests. The tests take care of using
    attributes and properties that are true for all sorts of GPs (e.g. only
    using parameters shared by all likelihoods).

    """
    #####################
    # testing functions #
    #####################

    @staticmethod
    def assert_values_differ(x):
        x = to_numpy(x)
        assert len(np.unique(x)) > 1

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
    def module_multioutput_cls(self):
        # since multioutput is not currently being tested, not an abstract
        # method
        pass

    @pytest.fixture
    def data(self):
        raise NotImplementedError

    @pytest.fixture
    def gp(self, gp_cls, module_cls, data):
        raise NotImplementedError

    @pytest.fixture
    def gp_fit(self, gp, data):
        X, y = data
        return gp.fit(X, y)

    @pytest.fixture
    def gp_multioutput(self, gp_cls, module_multioutput_cls, data):
        # should be fitted; since it's not currently being tested, not an
        # abstract method
        pass

    @pytest.fixture
    def pipe(self, gp):
        return Pipeline([
            ('noop', None),
            ('gp', gp),
        ])

    ######################
    # saving and loading #
    ######################

    def test_pickling(self, gp_fit, data):
        loaded = pickle.loads(pickle.dumps(gp_fit))
        X, _ = data

        y_pred_before = gp_fit.predict(X)
        y_pred_after = loaded.predict(X)
        assert np.allclose(y_pred_before, y_pred_after)

    def test_deepcopy(self, gp_fit, data):
        copied = copy.deepcopy(gp_fit)
        X, _ = data

        y_pred_before = gp_fit.predict(X)
        y_pred_after = copied.predict(X)
        assert np.allclose(y_pred_before, y_pred_after)

    def test_clone(self, gp_fit, data):
        clone(gp_fit)  # does not raise

    def test_save_load_params(self, gp_fit, tmpdir):
        gp2 = clone(gp_fit).initialize()

        # check first that parameters are not equal
        for (_, p0), (_, p1) in zip(
                gp_fit.get_all_learnable_params(), gp2.get_all_learnable_params(),
        ):
            assert not (p0 == p1).all()

        # save and load params to gp2
        p_module = tmpdir.join('module.pt')
        p_likelihood = tmpdir.join('likelihood.pt')
        with open(str(p_module), 'wb') as fm, open(str(p_likelihood), 'wb') as fll:
            gp_fit.save_params(f_params=fm, f_likelihood=fll)
        with open(str(p_module), 'rb') as fm, open(str(p_likelihood), 'rb') as fll:
            gp2.load_params(f_params=fm, f_likelihood=fll)

        # now parameters should be equal
        for (n0, p0), (n1, p1) in zip(
                gp_fit.get_all_learnable_params(), gp2.get_all_learnable_params(),
        ):
            assert n0 == n1
            torch.testing.assert_close(p0, p1)

    ##############
    # functional #
    ##############

    def test_fit(self, gp_fit, recwarn):
        # fitting does not raise anything and triggers no warning
        assert not recwarn.list

    def test_gp_learns(self, gp_fit):
        history = gp_fit.history
        assert history[0, 'train_loss'] > 0.5 * history[-1, 'train_loss']

    def test_forward(self, gp_fit, data):
        X = data[0]
        y_forward = gp_fit.forward(X)

        for yi in y_forward:
            assert isinstance(yi, torch.distributions.distribution.Distribution)

        total_shape = sum(get_batch_size(p) for p in y_forward)
        assert total_shape == self.n_samples

    def test_predict(self, gp_fit, data):
        X = data[0]
        y_pred = gp_fit.predict(X)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (self.n_samples,)
        self.assert_values_differ(y_pred)

    def test_predict_proba(self, gp_fit, data):
        if not self.supports_predict_proba:
            return

        X = data[0]
        y_proba = gp_fit.predict_proba(X)

        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == (self.n_samples, self.n_targets)
        self.assert_values_differ(y_proba)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_fit_and_predict_with_cuda(self, gp, data):

        gp.set_params(device='cuda')
        X, y = data
        gp.fit(X, y)
        y_pred = gp.predict(X)
        self.assert_values_differ(y_pred)

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
            # this parameter does not exist but that's okay
            'likelihood__some_parameter': [1, 2],
        }
        gp.set_params(verbose=0)
        gs = GridSearchCV(gp, params, refit=True, cv=3, scoring=self.scoring)
        gs.fit(X[:60], y[:60])  # for speed

        # sklearn will catch fit failures and raise a warning, we should thus
        # check that no warnings are generated
        assert not recwarn.list

    # Multioutput doesn't work because GPyTorch makes assumptions about the
    # module output that are not compatible with multiple outputs. The tests are
    # left in case this is fixed but they're not being executed.

    @pytest.mark.skip
    def test_fit_multioutput(self, gp_multioutput):
        # doesn't raise
        pass

    @pytest.mark.skip
    def test_multioutput_forward_iter(self, gp_multioutput, data):
        X = data[0]
        y_infer = next(gp_multioutput.forward_iter(X))

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 3
        assert y_infer[0].shape[0] == min(len(X), gp_multioutput.batch_size)

    @pytest.mark.skip
    def test_multioutput_forward(self, gp_multioutput, data):
        X = data[0]
        y_infer = gp_multioutput.forward(X)

        assert isinstance(y_infer, tuple)
        assert len(y_infer) == 2
        for arr in y_infer:
            assert is_torch_data_type(arr)

        for output in y_infer:
            assert len(output) == self.n_samples

    @pytest.mark.skip
    def test_multioutput_predict(self, gp_multioutput, data):
        X = data[0]

        # does not raise
        y_pred = gp_multioutput.predict(X)

        # Expecting only 1 column containing predict class:
        # (number of samples,)
        assert y_pred.shape == (self.n_samples)
        self.assert_values_differ(y_pred)

    @pytest.mark.skip
    def test_multioutput_predict_proba(self, gp_multioutput, data):
        X = data[0]

        # does not raise
        y_proba = gp_multioutput.predict_proba(X)
        self.assert_values_differ(y_proba)

        # Expecting full output: (number of samples, number of output units)
        assert y_proba.shape == (self.n_samples, self.n_targets)
        # Probabilities, hence these limits
        assert y_proba.min() >= 0
        assert y_proba.max() <= 1

    ##################
    # initialization #
    ##################

    @pytest.mark.parametrize('kwargs,expected', [
        ({}, ""),
        ({
            'likelihood__noise_prior': gpytorch.priors.NormalPrior(0, 1),
            'likelihood__batch_shape': (345,),
        }, ""),
        ({
            'likelihood__noise_prior': gpytorch.priors.NormalPrior(0, 1),
            'optimizer__momentum': 0.567,
        }, ""),
    ])
    def test_set_params_uninitialized_net_correct_message(
            self, gp, kwargs, expected, capsys):
        # When gp is uninitialized, there is nothing to alert the user to
        gp.set_params(**kwargs)
        msg = capsys.readouterr()[0].strip()
        assert msg == expected

    @pytest.mark.parametrize('kwargs,expected', [
        ({}, ""),
        (
            # this parameter does not exist but that's okay
            {'likelihood__some_parameter': 2},
            ("Re-initializing module because the following "
             "parameters were re-set: likelihood__some_parameter.\n"
             "Re-initializing criterion.\n"
             "Re-initializing optimizer.")
        ),
        (
            {
                # this parameter does not exist but that's okay
                'likelihood__some_parameter': 2,
                'optimizer__momentum': 0.567,
            },
            ("Re-initializing module because the following "
             "parameters were re-set: likelihood__some_parameter.\n"
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

    def test_likelihood_already_initialized_does_not_reinit(self, gp, gp_cls):
        # When the likelihood is already initialized and no params changed, it
        # should just be set as is instead of creating a new instance. In
        # theory, the same should apply to modules but in all the examples here,
        # modules require params, so we cannot test it.

        gp_init = gp.initialize()
        # create a new GP instance using this somewhat convoluted approach
        # because we don't know what arguments are required to initialize from
        # scratch
        params = gp_init.get_params()
        # set likelihood and likelihood to be initialized already
        params['likelihood'] = gp_init.likelihood_
        gp = gp_cls(**params).initialize()

        assert gp.likelihood_ is gp_init.likelihood_

    ##########################
    # probabalistic specific #
    ##########################

    @pytest.mark.parametrize("n_samples", [1, 2, 10])
    def test_sampling(self, gp, data, n_samples):
        X, _ = data
        samples = gp.initialize().sample(X, n_samples=n_samples)
        assert samples.shape == (n_samples, len(X))

        # check that values are not all the same -- this can happen when
        # posterior variances are skipped via a setting
        self.assert_values_differ(samples)

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
        self.assert_values_differ(y_std)

    def test_predict_return_cov(self, gp_fit, data):
        if not self.supports_return_cov:
            return

        X, _ = data
        msg = ("The 'return_cov' argument is not supported. Please try: "
               "'posterior = next(gpr.forward_iter(X)); posterior.covariance_matrix'.")
        with pytest.raises(NotImplementedError, match=re.escape(msg)):
            gp_fit.predict(X, return_cov=True)


class TestExactGPRegressor(BaseProbabilisticTests):
    """Tests for ExactGPRegressor."""

    ##########################
    # constants and fixtures #
    ##########################

    n_samples = 34
    n_targets = 1
    supports_predict_proba = False
    supports_return_std = True
    supports_return_cov = True
    settable_params = {'gp__likelihood__noise_prior': None}
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
    def gp(self, gp_cls, module_cls):
        gpr = gp_cls(
            module_cls,
            optimizer=torch.optim.Adam,
            lr=0.1,
            max_epochs=20,
        )
        return gpr

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
    """Tests for GPRegressor."""

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

    # Since GPyTorch v1.10, GPRegressor works with pickle/deepcopy.

    def test_pickling(self, gp_fit, data):
        loaded = pickle.loads(pickle.dumps(gp_fit))
        X, _ = data

        y_pred_before = gp_fit.predict(X)
        y_pred_after = loaded.predict(X)
        assert np.allclose(y_pred_before, y_pred_after)

    def test_deepcopy(self, gp_fit, data):
        copied = copy.deepcopy(gp_fit)
        X, _ = data

        y_pred_before = gp_fit.predict(X)
        y_pred_after = copied.predict(X)
        assert np.allclose(y_pred_before, y_pred_after)


class TestGPBinaryClassifier(BaseProbabilisticTests):
    """Tests for GPBinaryClassifier."""

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
            likelihood=MyBernoulliLikelihood,
            criterion=gpytorch.mlls.VariationalELBO,
            criterion__num_data=int(0.8 * len(y)),
            batch_size=24,
        )
        # we want to make sure batching is properly tested
        assert gpc.batch_size < self.n_samples
        return gpc

    # Since GPyTorch v1.10, GPBinaryClassifier is the only estimator left that
    # still has issues with pickling/deepcopying.

    @pytest.mark.xfail(strict=True)
    def test_pickling(self, gp_fit, data):
        # Currently fails because of issues outside of our control, this test
        # should alert us to when the issue has been fixed. Some issues have
        # been fixed in https://github.com/cornellius-gp/gpytorch/pull/1336 but
        # not all.
        pickle.dumps(gp_fit)

    def test_pickle_error_msg(self, gp_fit, data):
        # Should eventually be replaced by a test that saves and loads the model
        # using pickle and checks that the predictions are identical
        msg = ("This GPyTorch model cannot be pickled. The reason is probably this:"
               " https://github.com/pytorch/pytorch/issues/38137. "
               "Try using 'dill' instead of 'pickle'.")
        with pytest.raises(pickle.PicklingError, match=msg):
            pickle.dumps(gp_fit)

    def test_deepcopy(self, gp_fit, data):
        # Should eventually be replaced by a test that saves and loads the model
        # using deepcopy and checks that the predictions are identical
        msg = ("This GPyTorch model cannot be pickled. The reason is probably this:"
               " https://github.com/pytorch/pytorch/issues/38137. "
               "Try using 'dill' instead of 'pickle'.")
        with pytest.raises(pickle.PicklingError, match=msg):
            copy.deepcopy(gp_fit)  # doesn't raise
