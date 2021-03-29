"""Tests for probabilistic.py"""

import pickle

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import torch


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
        raise NotImplementedError

    @property
    def settable_params(self):
        # expects dict of parameters that can be set with set_params
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
    def pipe(self):
        raise NotImplementedError

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

    # def test_multioutput_forward_iter(self, gp_multiouput, data):
    #     X = data[0]
    #     y_infer = next(gp_multiouput.forward_iter(X))

    #     assert isinstance(y_infer, tuple)
    #     assert len(y_infer) == 3
    #     assert y_infer[0].shape[0] == min(len(X), gp_multiouput.batch_size)

    # def test_multioutput_forward(self, gp_multiouput, data):
    #     X = data[0]
    #     y_infer = gp_multiouput.forward(X)

    #     assert isinstance(y_infer, tuple)
    #     assert len(y_infer) == 3
    #     for arr in y_infer:
    #         assert is_torch_data_type(arr)

    #     for output in y_infer:
    #         assert len(output) == self.n_samples

    # def test_multioutput_predict(self, gp_multiouput, data):
    #     X = data[0]

    #     # does not raise
    #     y_pred = gp_multiouput.predict(X)

    #     # Expecting only 1 column containing predict class:
    #     # (number of samples,)
    #     assert y_pred.shape == (self.n_samples)

    # def test_multiouput_predict_proba(self, gp_multiouput, data):
    #     X = data[0]

    #     # does not raise
    #     y_proba = gp_multiouput.predict_proba(X)

    #     # Expecting full output: (number of samples, number of output units)
    #     assert y_proba.shape == (self.n_samples, self.n_targets)
    #     # Probabilities, hence these limits
    #     assert y_proba.min() >= 0
    #     assert y_proba.max() <= 1

    def test_in_sklearn_pipeline(self, pipe, data):
        X, y = data
        # none of this raises an error
        pipe.fit(X, y)
        pipe.predict(X)
        pipe.set_params(**self.settable_params)

    def test_grid_search_works(self, gp, data):
        X, y = data
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
            'likelihood__noise_prior': [None, gpytorch.priors.GammaPrior(1, 2)],
        }
        gs = GridSearchCV(gp, params, refit=True, cv=3, scoring='accuracy')
        gs.fit(X[:100], y[:100])  # for speed
        print(gs.best_score_, gs.best_params_)

    ##################
    # initialization #
    ##################

    @pytest.mark.parametrize('kwargs,expected', [
        ({}, ""),
        (
            {'likelihood__noise_prior': 132, 'likelihood__batch_shape': 345},
            ("Re-initializing module because the following "
             "parameters were re-set: noise_prior, batch_shape.\n"
             "Re-initializing optimizer.")
        ),
        (
            {'likelihood__noise_prior': 132,
             'optimizer__momentum': 0.567},
            ("Re-initializing module because the following "
             "parameters were re-set: noise_prior.\n"
             "Re-initializing optimizer.")
        ),
    ])
    def test_reinitializing_module_optimizer_message(
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

    ##########################
    # probabalistic specific #
    ##########################

    def test_sampling(self):
        pass

    def test_confidence_region(self):
        pass

    def test_predict_return_std(self):
        pass

    def test_predict_return_cov(self):
        pass


class TestGPRegressorExact(BaseProbabilisticTests):
    ##########################
    # constants and fixtures #
    ##########################

    n_samples = 34
    n_targets = 1
    supports_predict_proba = False
    supports_return_std = True
    settable_params = {'gp__module__y': torch.zeros(n_samples).float()}

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
        return RbfModule

    @pytest.fixture
    def gp(self, gp_cls, module_cls, data):
        from skorch.utils import futureattr

        X, y = data
        gpr = gp_cls(
            module_cls,
            module__X=torch.from_numpy(X),
            module__y=torch.from_numpy(y),
            module__likelihood=futureattr('likelihood_'),

            optimizer=torch.optim.Adam,
            lr=0.1,
            max_epochs=20,
            batch_size=-1,
        )
        return gpr

    @pytest.fixture
    def pipe(self, gp):
        return Pipeline([
            ('noop', None),
            ('gp', gp),
        ])


class TestGPRegressorVariational(BaseProbabilisticTests):
    ##########################
    # constants and fixtures #
    ##########################

    n_targets = 1
    supports_predict_proba = False
    supports_return_std = True


class TestGPBinaryClassifier(BaseProbabilisticTests):
    ##########################
    # constants and fixtures #
    ##########################

    n_targets = 2
    supports_predict_proba = True
    supports_return_std = False
