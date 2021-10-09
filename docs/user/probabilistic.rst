==================
Gaussian Processes
==================

skorch integrates with GPyTorch_ to make it easy to train Gaussian Process (GP)
models. You should already know how Gaussian Processes work. Please refer to
other resources if you want to learn about them, this section assumes
familiarity with the concept.

GPyTorch adopts many patterns from PyTorch, thus making it easy to pick up for
seasoned PyTorch users. Similarly, the skorch GPyTorch integration should look
familiar to seasoned skorch users. However, GPs are a different beast than the
more common, non-probabilistic machine learning techniques. It is important to
understand the basic concepts before using them in practice.

Installation
------------

In addition to the normal skorch dependencies and PyTorch, you need to install
GPyTorch as well. It wasn't added as a normal dependency since most users
probably are not interested in using skorch for GPs. To install GPyTorch, use
either pip or conda:

.. code:: bash

    # using pip
    pip install -U gpytorch
    # using conda
    conda install gpytorch -c gpytorch

When to use GPyTorch with skorch
--------------------------------

Here we want to quickly explain when it would be a good idea for you to use
GPyTorch with skorch. There are a couple of offerings in the Python ecosystem
when it comes to Gaussian Processes. We cannot provide an exhaustive list of
pros and cons of each possibility. There are, however, two obvious alternatives
that are worth discussing: using the sklearn_ implementation and using GPyTorch
without skorch.

When to use skorch + GPyTorch over sklearn:

* When you are more familiar with PyTorch than with sklearn
* When the kernels provided by sklearn are not sufficient for your use case and
  you would like to implement custom kernels with PyTorch
* When you want to use the rich set of optimizers available in PyTorch
* When sklearn is too slow and you want to use the GPU or scale across machines
* When you like to use the skorch extras, e.g. callbacks

When to use skorch + GPyTorch over pure GPyTorch

* When you're already familiar with skorch and want an easy entry into GPs
* When you like to use the skorch extras, e.g. callbacks and grid search
* When you don't want to bother with writing your own training loop

However, if you are researching GPs and would like to have control over every
detail, using all the rich but very specific featues that GPyTorch has on offer,
it is better to use it directly without skorch.

Examples
--------

Exact Gaussian Processes
^^^^^^^^^^^^^^^^^^^^^^^^

Same as GPyTorch, skorch supports exact and approximate Gaussian Processes
regression. For exact GPs, use the
:class:`~skorch.probabilistic.ExactGPRegressor`. The likelihood has to be a
:class:`~gpytorch.likelihoods.GaussianLikelihood` and the criterion
:class:`~gpytorch.mlls.ExactMarginalLogLikelihood`, but those are the defaults
and thus don't need to be specified. For exact GPs, the module needs to be an
:class:`~gpytorch.models.ExactGP`. For this example, we use a simple RBF kernel.

.. code:: python

    import gpytorch
    from skorch.probabilistic import ExactGPRegressor

    class RbfModule(gpytorch.models.ExactGP):
        def __init__(likelihood, self):
            # detail: We don't set train_inputs and train_targets here skorch because
            # will take care of that.
            super().__init__()
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.RBFKernel()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    gpr = ExactGPRegressor(RbfModule)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

As you can see, this almost looks like a normal skorch regressor with a normal
PyTorch module. We can fit as normal using the ``fit`` method and predict using
the ``predict`` method.

Inside the module, we determine the mean by using a mean function (just constant
in this case) and the covariance matrix using the RBF kernel function. You
should know about mean and kernel functions already. Having the mean and
covariance matrix, we assume that the output distribution is a multivariate
normal function, since exact GPs rely on this assumption. We could send the
``x`` through an MLP for `Deep Kernel Learning
<https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/index.html>`_
but left it out to keep the example simple.

One major difference to usual deep learning models is that we actually predict a
distribution, not just a point estimate. That means that if we choose an
appropriate model that fits the data well, we can express the **uncertainty** of
the model:

.. code:: python

    y_pred, y_std = gpr.predict(X, return_std=True)
    lower_conf_region = y_pred - y_std
    upper_conf_region = y_pred + y_std

Here we not only returned the mean of the prediction, ``y_pred``, but also its
standard deviation, ``y_std``. This tells us how uncertain the model is about
its prediction. E.g., it could be the case that the model is fairly certain when
*interpolating* between data points but uncertain about *extrapolating*. This is
not possible to know when models only learn point predictions.

The obtain the confidence region, you can also use the ``confidence_region``
method:

.. code:: python

    # 1 standard deviation
    lower, upper = gpr.confidence_region(X, sigmas=1)

    # 2 standard deviation, the default
    lower, upper = gpr.confidence_region(X, sigmas=2)

Furthermore, a GP allows you to sample from the distribution even *before
fitting* it. The GP needs to be initialized, however:

.. code:: python

    gpr = ExactGPRegressor(...)
    gpr.initialize()
    samples = gpr.sample(X, n_samples=100)

By visualizing the samples and comparing them to the true underlying
distribution of the target, you can already get a feel about whether the model
you built is capable of generating the distribution of the target. If fitting
takes a long time, it is therefore recommended to check the distribution first,
otherwise you may try to fit a model that is incapable of generating the true
distribution and waste a lot of time.

Approximate Gaussian Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For some situations, fitting an exact GP might be infeasible, e.g. because the
distribution is not Gaussian or because you want to perform stochastic
optimization with mini-batches. For this, GPyTorch provides facilities to train
variational and approximate GPs. The module should inherit from
:class:`~gpytorch.models.ApproximateGP` and should define a *variational
strategy*. From the skorch side of things, use
:class:`~skorch.probabilistic.GPRegressor`.

.. code:: python

    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from skorch.probabilistic import GPRegressor

    class VariationalModule(ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True,
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    X, y = get_data(...)
    X_incuding = X[:100]
    X_train, y_train = X[100:], y[100:]
    num_training_samples = len(X_train)

    gpr = GPRegressor(
        VariationalModule,
        module__inducing_points=X_inducing,
        criterion__num_data=num_training_samples,
    )

    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_train)

As you can see, the variational strategy requires us to use inducing points. We
split off 100 of our training data samples to use as inducing points, assuming
that they are representative of the whole distribution. Apart from this, there
is basically no difference to using exact GP regression.

Finally, skorch also provides :class:`~skorch.probabilistic.GPBinaryClassifier`
for binary classification with GPs. It uses a Bernoulli likelihood by default.
However, using GPs for classification is not very common, GPs are most commonly
used for regression tasks where data points have a known relationship to each
other (e.g. in time series forecasts).

Multiclass classification is not currently provided, but you can use
:class:`~skorch.probabilistic.GPBinaryClassifier` in conjunction with
:class:`~sklearn.multiclass.OneVsRestClassifier` to achieve the same result.

Further examples
----------------

To see all of this in action, we provide a notebook that shows using skorch with GPs on real world data: `Gaussian Processes notebook <https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Gaussian_Processes.ipynb)>`_.

.. _GPyTorch: https://gpytorch.ai/
.. _sklearn: https://scikit-learn.org/stable/modules/gaussian_process.html
