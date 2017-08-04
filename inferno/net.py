"""Neural net classes."""

import pickle

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from inferno.callbacks import AverageLoss
from inferno.callbacks import BestLoss
from inferno.callbacks import Callback
from inferno.callbacks import EpochTimer
from inferno.callbacks import PrintLog
from inferno.callbacks import Scoring
from inferno.dataset import Dataset
from inferno.exceptions import NotInitializedError
from inferno.utils import get_dim
from inferno.utils import to_numpy
from inferno.utils import to_var


class History(list):
    """A list-like collection that facilitates some of the more common
    tasks that are required.

    It is basically a list of dicts for each epoch, that again
    contains a list of dicts for each batch. For convenience, it has
    enhanced slicing notation and some methods to write new items.

    To access items from history, you may pass a tuple of up to four
    items:

      1. Slices along the epochs.
      2. Selects columns from history epochs, may be a single one or a
      tuple of column names.
      3. Slices along the batches.
      4. Selects columns from history batchs, may be a single one or a
      tuple of column names.

    You may use a combination of the four items.

    If you select columns that are not present in all epochs/batches,
    only those epochs/batches are chosen that contain said columns. If
    this set is empty, a KeyError is raised.

    Examples
    --------
    >>> # ACCESSING ITEMS
    >>> # history of a fitted neural net
    >>> history = net.history
    >>> # get current epoch, a dict
    >>> history[-1]
    >>> # get train losses from all epochs, a list of floats
    >>> history[:, 'train_loss']
    >>> # get train and valid losses from all epochs, a list of tuples
    >>> history[:, ('train_loss', 'valid_loss')]
    >>> # get current batches, a list of dicts
    >>> history[-1, 'batches']
    >>> # get latest batch, a dict
    >>> history[-1, 'batches', -1]
    >>> # get train losses from current batch, a list of floats
    >>> history[-1, 'batches', :, 'train_loss']
    >>> # get train and valid losses from current batch, a list of tuples
    >>> history[-1, 'batches', :, ('train_loss', 'valid_loss')]

    >>> # WRITING ITEMS
    >>> # add new epoch row
    >>> history.new_epoch()
    >>> # add an entry to current epoch
    >>> history.record('my-score', 123)
    >>> # add a batch row to the current epoch
    >>> history.new_batch()
    >>> # add an entry to the current batch
    >>> history.record_batch('my-batch-score', 456)
    >>> # overwrite entry of current batch
    >>> history.record_batch('my-batch-score', 789)

    """

    def new_epoch(self):
        """Register a new epoch row."""
        self.append({'batches': []})

    def new_batch(self):
        """Register a new batch row for the current epoch."""
        self[-1]['batches'].append({})

    def record(self, attr, value):
        """Add a new value to the given column for the current
        epoch.

        """
        msg = "Call new_epoch before recording for the first time."
        assert len(self) > 0, msg
        self[-1][attr] = value

    def record_batch(self, attr, value):
        """Add a new value to the given column for the current
        batch.

        """
        self[-1]['batches'][-1][attr] = value

    def to_list(self):
        """Return history object as a list."""
        return list(self)

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return super().__getitem__(i)

        class __missingno:
            def __init__(self, e):
                self.e = e
            def __repr__(self):
                return 'missingno'

        def partial_index(l, idx):
            is_list_like = lambda x: isinstance(x, list)

            needs_unrolling = is_list_like(l) \
                    and len(l) > 0 and is_list_like(l[0])
            needs_indirection = is_list_like(l) \
                    and not isinstance(idx, (int, tuple, list, slice))

            if needs_unrolling or needs_indirection:
                return [partial_index(n, idx) for n in l]

            # join results of multiple indices
            if isinstance(idx, (tuple, list)):
                def incomplete_mapper(x):
                    for xs in x:
                        if type(xs) is __missingno:
                            return xs
                    return x
                zz = [partial_index(l, n) for n in idx]
                if is_list_like(l):
                    total_join = zip(*zz)
                    inner_join = list(map(incomplete_mapper, total_join))
                else:
                    total_join = tuple(zz)
                    inner_join = incomplete_mapper(total_join)
                return inner_join

            try:
                return l[idx]
            except KeyError as e:
                return __missingno(e)

        def filter_missing(x):
            if isinstance(x, list):
                children = [filter_missing(n) for n in x]
                filtered = list(filter(lambda x: type(x) != __missingno, children))

                if len(children) > 0 and len(filtered) == 0:
                    return next(filter(lambda x: type(x) == __missingno, children))
                return filtered
            return x

        x = self
        if isinstance(i, tuple):
            for part in i:
                x_dirty = partial_index(x, part)
                x = filter_missing(x_dirty)
                if type(x) is __missingno:
                    raise x.e
            return x
        raise ValueError("Invalid parameter type passed to index. "
                         "Pass string, int or tuple.")


class NeuralNet(Callback):
    """NeuralNet base class.

    The base class covers more generic cases. Depending on your use
    case, you might want to use `NeuralNetClassifier` or
    `NeuralNetRegressor`.

    In addition to the parameters listed below, there are parameters
    with specific prefixes that are handled separately. To illustrate
    this, here is an example:

    ```
    net = NeuralNet(
        ...,
        optim=torch.optim.SGD,
        optim__momentum=0.95,
    )
    ```

    This way, when `optim` is initialized, `NeuralNet` will take care
    of setting the `momentum` parameter to 0.95.

    (Note that the double underscore notation in `optim__momentum`
    means that the parameter `momentum` should be set on the object
    `optim`. This is the same semantic as used by sklearn.)

    Furthermore, this allows to change those parameters later:

    ```
    net.set_params(optim__momentum=0.99)
    ```

    This can be useful when you want to change certain parameters using
    a callback, when using the net in an sklearn grid search, etc.

    Parameters
    ----------
    module : torch module (class or instance)
      A torch module. In general, the uninstantiated class should be
      passed, although instantiated modules will also work.

    criterion : torch criterion (class)
      The uninitialized criterion (loss) used to optimize the
      module.

    optim : torch optim (class, default=torch.optim.SGD)
      The uninitialized optimizer (update rule) used to optimize the
      module

    lr : float (default=0.01)
      Learning rate passed to the optimizer. You may use `lr` instead
      of using `optim__lr`, which would result in the same outcome.

    max_epochs : int (default=10)
      The number of epochs to train for each `fit` call. Note that you
      may keyboard-interrupt training at any time.

    batch_size : int (default=128)
      Mini-batch size. Use this instead of setting
      `iterator_train__batch_size` and `iterator_test__batch_size`,
      which would result in the same outcome.

    iterator_train : torch DataLoader
      TODO: Will probably change.

    iterator_test : torch DataLoader
      TODO: Will probably change.

    callbacks : None or list of Callback instances (default=None)
      More callbacks, in addition to those specified in
      `default_callbacks`. Each callback should inherit from
      inferno.Callback. If not None, a list of tuples (name, callback)
      should be passed, where names should be unique. Callbacks may or
      may not be instantiated.
      Alternatively, it is possible to just pass a list of callbacks,
      which results in names being inferred from the class name.
      The callback name can be used to set parameters on specific
      callbacks (e.g., for the callback with name `'print_log'`, use
      `net.set_params(callbacks__print_log__keys=['epoch',
      'train_loss'])`).

    cold_start : bool (default=True)
      Whether each fit call should lead to a re-initialization of the
      module (cold start) or whether the module should be trained
      further (warm start).

    use_cuda : bool (default=False)
      Whether usage of cuda is intended. If True, data in torch
      tensors will be pushed to cuda tensors before being sent to the
      module.

    history : None or inferno.History instance (default=None)
      If not None, start from the given history. In general, this
      parameter should be left at None. Only set it if you want to
      start with a specific training history.

    Attributes
    ----------
    prefixes_ : list of str
      Contains the prefixes to special parameters. E.g., since there
      is the `'module'` prefix, it is possible to set parameters like
      so: `NeuralNet(..., optim__momentum=0.95)`.

    default_callbacks : list of str
      Callbacks that come by default. They are mainly set for the
      user's convenience. By default, an EpochTimer, AverageLoss,
      BestLoss, and PrintLog are set.

    initialized_ : bool
      Whether the NeuralNet was initialized.

    module_ : torch module (instance)
      The instantiated module.

    criterion_ : torch criterion (instance)
      The instantiated criterion.

    callbacks_ : list of tuples
      The complete (i.e. default and other), initialized callbacks, in
      a tuple with unique names.

    """
    prefixes_ = ['module', 'iterator_train', 'iterator_test', 'optim',
                 'criterion', 'callbacks']

    default_callbacks = [
        ('epoch_timer', EpochTimer),
        ('average_loss', AverageLoss),
        ('best_loss', BestLoss),
        ('print_log', PrintLog),
    ]

    def __init__(
            self,
            module,
            criterion,
            optim=torch.optim.SGD,
            lr=0.01,
            max_epochs=10,
            batch_size=128,
            iterator_train=DataLoader,
            iterator_test=DataLoader,
            callbacks=None,
            cold_start=True,
            use_cuda=False,
            history=None,
            **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.optim = optim
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.iterator_train = iterator_train
        self.iterator_test = iterator_test
        self.callbacks = callbacks
        self.cold_start = cold_start
        self.use_cuda = use_cuda
        self.history = history

        for key in kwargs:
            assert not hasattr(self, key)
            key_has_prefix = any(key.startswith(p) for p in self.prefixes_)
            assert key.endswith('_') or key_has_prefix
        vars(self).update(kwargs)

        # e.g. if object was cloned, don't overwrite this attr
        if not hasattr(self, 'initialized_'):
            self.initialized_ = False

    def notify(self, method_name, **cb_kwargs):
        """Call the callback method specified in `method_name` with
        parameters specified in `cb_kwargs`.

        Method names can be one of:
        * on_train_begin
        * on_train_end
        * on_epoch_begin
        * on_epoch_end
        * on_batch_begin
        * on_batch_end

        """
        getattr(self, method_name)(self, **cb_kwargs)
        for _, cb in self.callbacks_:
            getattr(cb, method_name)(self, **cb_kwargs)

    def on_epoch_begin(self, net, **kwargs):
        self.history.new_epoch()
        self.history.record('epoch', len(self.history))

    def on_batch_begin(self, net, train=False, **kwargs):
        self.history.new_batch()

    def _yield_callbacks(self):
        # handles cases:
        #   * default and user callbacks
        #   * callbacks with and without name
        #   * initialized and uninitialized callbacks
        for item in self.default_callbacks + (self.callbacks or []):
            if isinstance(item, (tuple, list)):
                name, cb = item
            else:
                cb = item
                if isinstance(cb, type):  # uninitialized:
                    name = cb.__name__
                else:
                    name = cb.__class__.__name__
            yield name, cb

    def initialize_callbacks(self):
        """Initializes all callbacks and save the result in the
        `callbacks_` attribute.

        Both `default_callbacks` and `callbacks` are used (in that
        order). Callbacks may either be initialized or not, and if
        they don't have a name, the name is inferred from the class
        name. The `initialize` method is called on all callbacks.

        The final result will be a list of tuples, where each tuple
        consists of a name and an initialized callback. If names are
        not unique, a ValueError is raised.

        """
        names_seen = set()
        callbacks_ = []

        for name, cb in self._yield_callbacks():
            if name in names_seen:
                raise ValueError("The callback name '{}' appears more than "
                                 "once.".format(name))
            names_seen.add(name)

            params = self._get_params_for('callbacks__{}'.format(name))
            if isinstance(cb, type):  # uninitialized:
                cb = cb(**params)
            else:
                cb.set_params(**params)
            cb.initialize()
            callbacks_.append((name, cb))

        self.callbacks_ = callbacks_

    def initialize_criterion(self):
        """Initializes the criterion."""
        criterion_params = self._get_params_for('criterion')
        self.criterion_ = self.criterion(**criterion_params)

    def initialize_module(self):
        """Initializes the module.

        Note that if the module has learned parameters, those will be
        reset.

        """
        if self.initialized_:
            print("Re-initializing module!")

        kwargs = self._get_params_for('module')
        self.module_ = self.module(**kwargs)
        if self.use_cuda:
            self.module_.cuda()
        return self

    def initialize(self):
        """Initializes all components of the NeuralNet."""
        self.initialize_callbacks()
        self.initialize_criterion()
        self.initialize_module()
        self.history = self.history if self.history is not None else History()

        self.initialized_ = True
        return self

    def check_data(self, *data):
        pass

    def validation_step(self, xi, yi):
        """Perform a forward step using batched data and return the
        resulting loss.

        The module is set to be in evaluation mode (e.g. dropout is
        not applied).

        """
        xi, yi = to_var(xi), to_var(yi)
        self.module_.eval()

        y_pred = self.infer(xi)
        return self.get_loss(y_pred, yi, train=False)

    def train_step(self, xi, yi, optimizer):
        """Perform a forward step using batched data, update module
        parameters, and return the loss.

        The module is set to be in train mode (e.g. dropout is
        applied).

        """
        xi, yi = to_var(xi), to_var(yi)
        self.module_.train()

        optimizer.zero_grad()
        y_pred = self.infer(xi)
        loss = self.get_loss(y_pred, yi, train=True)
        loss.backward()
        optimizer.step()
        return loss

    def fit_loop(self, X, y, epochs=None):
        """The proper fit loop.

        Contains the logic of what actually happens during the fit
        loop.

        Parameters
        ----------
        X : TODO

        y : TODO

        epochs : int or None (default=None)
          If int, train for this number of epochs; if None, use
          `self.max_epochs`.

        **fit_params : TODO

        """
        self.check_data(X, y)
        epochs = epochs or self.max_epochs
        optimizer = self.get_optimizer()
        for epoch in range(epochs):
            self.notify('on_epoch_begin', X=X, y=y)

            for xi, yi in self.get_iterator(X, y, train=True):
                self.notify('on_batch_begin', X=xi, y=yi, train=True)
                loss = self.train_step(xi, yi, optimizer)
                self.history.record_batch('train_loss', loss.data[0])
                self.history.record_batch('train_batch_size', len(xi))
                self.notify('on_batch_end', X=xi, y=yi, train=True)

            for xi, yi in self.get_iterator(X, y, train=False):
                self.notify('on_batch_begin', X=xi, y=yi, train=False)
                loss = self.validation_step(xi, yi)
                self.history.record_batch('valid_loss', loss.data[0])
                self.history.record_batch('valid_batch_size', len(xi))
                self.notify('on_batch_end', X=xi, y=yi, train=False)

            self.notify('on_epoch_end', X=X, y=y)
        return self

    def partial_fit(self, X, y, classes=None, **fit_params):
        """Fit the module.

        The module is not re-initialized, which means that this method
        should be used if you want to continue training a model (warm
        start).

        Parameters
        ----------
        X : TODO

        y : TODO

        classes : array, sahpe (n_classes,)
          Solely for sklearn compatibility, currently unused.

        **fit_params : TODO

        """
        self.notify('on_train_begin')
        try:
            self.fit_loop(X, y)
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end')
        return self

    def fit(self, X, y, **fit_params):
        """Initialize and fit the module.

        Unless `cold_start` is False, the module will be re-initialized.

        Parameters
        ----------
        X : TODO

        y : TODO

        **fit_params : TODO

        """
        if self.cold_start or not hasattr(self, 'initialized_'):
            self.initialize()

        self.partial_fit(X, y, **fit_params)
        return self

    def forward(self, X, training_behavior=False):
        """Perform a forward steps on the module with batches derived
        from data.

        Parameters
        ----------
        X : TODO

        training_behavior : bool (default=False)
          Whether to set the module to train mode or not.

        Returns
        -------
        y_infer : torch tensor
          The result from the forward step.

        """
        self.module_.train(training_behavior)

        iterator = self.get_iterator(X, train=training_behavior)
        y_infer = []
        for x, _ in iterator:
            x = to_var(x, use_cuda=self.use_cuda)
            y_infer.append(self.infer(x))
        return torch.cat(y_infer, dim=0)

    def infer(self, x):
        x = to_var(x)
        if isinstance(x, dict):
            return self.module_(**x)
        return self.module_(x)

    def predict_proba(self, X):
        """Where applicable, return probability estimates for
        samples.

        Parameters
        ----------
        X : TODO

        Returns
        -------
        y_proba : numpy ndarray

        """
        y_proba = self.forward(X, training_behavior=False)
        y_proba = to_numpy(y_proba)
        return y_proba

    def predict(self, X):
        """Where applicable, return class labels for samples in X.

        Parameters
        ----------
        X : TODO

        Returns
        -------
        y_pred : numpy ndarray

        """
        self.module_.train(False)
        return self.predict_proba(X).argmax(1)

    def get_optimizer(self):
        """Get the initialized model optimizer. If `self.optim__lr` is
        not set, use `self.lr` instead.

        """
        kwargs = self._get_params_for('optim')
        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr
        return self.optim(self.module_.parameters(), **kwargs)

    def get_loss(self, y_pred, y_true, train=False):
        """Return the loss for this batch.

        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values

        y_true : torch tensor
          True target values.

        """
        return self.criterion_(y_pred, y_true)

    def get_iterator(self, X, y=None, train=False):
        """Get an iterator that allows to loop over the batches of the
        given data.

        If `self.iterator_train__batch_size` and/or
        `self.iterator_test__batch_size` are not set, use
        `self.batch_size` instead.

        Parameters
        ----------
        X : TODO

        y : TODO

        train : bool (default=False)
          Whether to use `iterator_train` or `iterator_test`.

        Returns
        -------
        iterator
          An instantiated iterator that allows to loop over the
          mini-batches.

        """
        dataset = Dataset(X, y, use_cuda=self.use_cuda)

        if train:
            kwargs = self._get_params_for('iterator_train')
            iterator = self.iterator_train
        else:
            kwargs = self._get_params_for('iterator_test')
            iterator = self.iterator_test

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        return iterator(dataset, **kwargs)

    def _get_params_for(self, prefix):
        if not prefix.endswith('__'):
            prefix += '__'
        return {key[len(prefix):]: val for key, val in self.__dict__.items()
                if key.startswith(prefix)}

    def _get_param_names(self):
        return self.__dict__.keys()

    def get_params(self, deep=True, **kwargs):
        return BaseEstimator.get_params(self, deep=deep, **kwargs)

    def set_params(self, **kwargs):
        normal_params, special_params = {}, {}
        for key, val in kwargs.items():
            if any(key.startswith(prefix) for prefix in self.prefixes_):
                special_params[key] = val
            else:
                normal_params[key] = val
        BaseEstimator.set_params(self, **normal_params)

        for key, val in special_params.items():
            if key.endswith('_'):
                raise ValueError("Not sure: Should this ever happen?")
            else:
                setattr(self, key, val)

        if any(key.startswith('criterion') for key in special_params):
            self.initialize_criterion()
        if any(key.startswith('callbacks') for key in special_params):
            self.initialize_callbacks()
        if any(key.startswith('module') for key in special_params):
            self.initialize_module()

        return self

    def __getstate__(self):
        state = BaseEstimator.__getstate__(self)
        if 'module_' in state:
            module_ = state.pop('module_')
            module_dump = pickle.dumps(module_)
            state['module_'] = module_dump
        return state

    def __setstate__(self, state):
        if 'module_' in state:
            module_dump = state.pop('module_')
            module_ = pickle.loads(module_dump)
            state['module_'] = module_
        BaseEstimator.__setstate__(self, state)

    def save_params(self, f):
        """Save only the module's parameters, not the whole object.

        To save the whole object, use pickle.

        Parameters
        ----------
        f : file-like object or str
          See `torch.save` documentation.

        Example
        -------
        >>> before = NeuralNetClassifier(mymodule)
        >>> before.save_params('path/to/file')
        >>> after = NeuralNetClassifier(mymodule).initialize()
        >>> after.load_params('path/to/file')

        """
        if not hasattr(self, 'module_'):
            raise NotInitializedError(
                "Cannot save parameters of an un-initialized model. "
                "Please initialize first by calling `.initialize()` "
                "or by fitting the model with `.fit(...)`.")
        torch.save(self.module_.state_dict(), f)

    def load_params(self, f):
        """Load only the module's parameters, not the whole object.

        To save and load the whole object, use pickle.

        Parameters
        ----------
        f : file-like object or str
          See `torch.load` documentation.

        Example
        -------
        >>> before = NeuralNetClassifier(mymodule)
        >>> before.save_params('path/to/file')
        >>> after = NeuralNetClassifier(mymodule).initialize()
        >>> after.load_params('path/to/file')

        """
        if not hasattr(self, 'module_'):
            raise NotInitializedError(
                "Cannot load parameters of an un-initialized model. "
                "Please initialize first by calling `.initialize()` "
                "or by fitting the model with `.fit(...)`.")
        self.module_.load_state_dict(torch.load(f))


def accuracy_pred_extractor(y):
    return np.argmax(to_numpy(y), axis=1)


class NeuralNetClassifier(NeuralNet):

    default_callbacks = [
        ('epoch_timer', EpochTimer()),
        ('average_loss', AverageLoss(
            key_sizes={'valid_acc': 'valid_batch_size'},
            keys_optional=['valid_acc'],
        )),
        ('accuracy', Scoring(
            name='valid_acc',
            scoring='accuracy_score',
            pred_extractor=accuracy_pred_extractor,
        )),
        ('best_loss', BestLoss(
            key_signs={'valid_acc': 1},
            keys_optional=['valid_acc'],
        )),
        ('print_log', PrintLog(
            keys=['valid_acc', 'valid_acc_best'],
            keys_optional=['valid_acc', 'valid_acc_best'],
        )),
    ]

    def __init__(
            self,
            module,
            criterion=torch.nn.NLLLoss,
            *args,
            **kwargs
    ):
        super(NeuralNetClassifier, self).__init__(
            module,
            criterion=criterion,
            *args,
            **kwargs
        )

    def get_loss(self, y_pred, y, train=False):
        y_pred_log = torch.log(y_pred)
        return self.criterion_(y_pred_log, y)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def fit(self, X, y, **fit_params):
        """See `NeuralNet.fit`.

        In contrast to `NeuralNet.fit`, `y` is non-optional to avoid mistakenly
        forgetting about `y`. However, `y` can be set to `None` in case it
        is derived dynamically from `X`.
        """
        return super(NeuralNetClassifier, self).fit(X, y, **fit_params)


class NeuralNetRegressor(NeuralNet):
    def __init__(
            self,
            module,
            criterion=torch.nn.MSELoss,
            *args,
            **kwargs
    ):
        super(NeuralNetRegressor, self).__init__(
            module,
            criterion=criterion,
            *args,
            **kwargs
        )

    def check_data(self, _, y):
        # The problem with 1-dim float y is that the pytorch DataLoader will
        # somehow upcast it to DoubleTensor
        if get_dim(y) == 1:
            raise ValueError("The target data shouldn't be 1-dimensional; "
                             "please reshape (e.g. y.reshape(-1, 1).")

    def fit(self, X, y, **fit_params):
        """See `NeuralNet.fit`.

        In contrast to `NeuralNet.fit`, `y` is non-optional to avoid mistakenly
        forgetting about `y`. However, `y` can be set to `None` in case it
        is derived dynamically from `X`.
        """
        return super(NeuralNetRegressor, self).fit(X, y, **fit_params)
