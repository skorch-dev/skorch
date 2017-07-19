"""Neural net classes."""

import pickle

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from inferno.callbacks import AverageLoss
from inferno.callbacks import Callback
from inferno.callbacks import EpochTimer


def to_var(X, use_cuda=False):
    X = to_tensor(X, use_cuda=use_cuda)
    return Variable(X)


def to_tensor(X, use_cuda=False):
    """Turn to torch Variable.

    Handles the cases:
      * Variable
      * PackedSequence
      * dict
      * numpy array
      * torch Tensor

    """
    if isinstance(X, (Variable, nn.utils.rnn.PackedSequence)):
        return X

    if isinstance(X, dict):
        return {key: to_tensor(val) for key, val in X.items()}

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if use_cuda:
        X = X.cuda()
    return X


def to_numpy(X):
    if X.is_cuda:
        X = X.cpu()
    try:
        data = X.data
    except AttributeError:
        data = X
    return data.numpy()


class History(list):

    def new_epoch(self):
        self.append({'batches': []})

    def new_batch(self):
        self[-1]['batches'].append({})

    def record(self, attr, value):
        assert len(self) > 0, "Call new_epoch before recording for the first time."
        self[-1][attr] = value

    def record_batch(self, attr, value):
        self[-1]['batches'][-1][attr] = value

    def to_list(self):
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
            if type(idx) is tuple or type(idx) is list:
                def incomplete_mapper(x):
                    for xs in x:
                        if type(xs) is __missingno:
                            return xs
                    return x
                total_join = zip(*[partial_index(l, n) for n in idx])
                inner_join = map(incomplete_mapper, total_join)
                return list(inner_join)

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
    prefixes_ = ['module', 'iterator_train', 'iterator_test', 'optim',
                 'criterion', 'callbacks']

    default_callbacks = [EpochTimer, AverageLoss]

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
        getattr(self, method_name)(self, **cb_kwargs)
        for _, cb in self.callbacks_:
            getattr(cb, method_name)(self, **cb_kwargs)

    def on_epoch_begin(self, net, **kwargs):
        self.history.new_epoch()
        self.history.record('epoch', len(self.history))

    def on_batch_begin(self, net, train=False, **kwargs):
        self.history.new_batch()

    def _get_callbacks_and_names(self, callbacks):
        names_and_cbs = []
        for item in callbacks:
            if isinstance(item, (tuple, list)):
                name, cb = item
            else:
                name = item.__class__.__name__
                cb = item
            names_and_cbs.append((name, cb))
        return names_and_cbs

    def initialize_callbacks(self):
        callbacks = [cb() for cb in self.default_callbacks]
        callbacks += self.callbacks or []
        names_and_cbs = self._get_callbacks_and_names(callbacks)

        names = list(zip(*names_and_cbs))[0]
        if len(names) != len(set(names)):
            # TODO: more useful message
            raise ValueError("There are callbacks with duplicate names.")

        callbacks_ = []
        for name, cb in names_and_cbs:
            params = self._get_params_for('callbacks__{}'.format(name))
            cb.set_params(**params)
            cb.initialize()
            callbacks_.append((name, cb))
        self.callbacks_ = callbacks_

    def initialize_criterion(self):
        criterion_params = self._get_params_for('criterion')
        self.criterion_ = self.criterion(**criterion_params)

    def initialize_module(self):
        if self.initialized_:
            print("Re-initializing module!")

        kwargs = self._get_params_for('module')
        self.module_ = self.module(**kwargs)
        return self

    def initialize(self):
        self.initialize_callbacks()
        self.initialize_criterion()
        self.initialize_module()
        self.history = self.history if self.history is not None else History()

        self.initialized_ = True
        return self

    def validation_step(self, xi, yi):
        xi, yi = Variable(xi), Variable(yi)
        self.module_.eval()

        y_pred = self.module_(xi)
        return self.get_loss(y_pred, yi, train=False)

    def train_step(self, xi, yi, optimizer):
        xi, yi = Variable(xi), Variable(yi)
        self.module_.train()

        optimizer.zero_grad()
        y_pred = self.module_(xi)
        loss = self.get_loss(y_pred, yi, train=True)
        loss.backward()
        optimizer.step()
        return loss

    def fit_loop(self, X, y):
        optimizer = self.get_optimizer()
        for epoch in range(self.max_epochs):
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

    def fit(self, X, y, **fit_params):
        if self.cold_start or not hasattr(self, 'initialized_'):
            self.initialize()

        self.notify('on_train_begin')
        try:
            self.fit_loop(X, y)
        except KeyboardInterrupt:
            pass
        self.notify('on_train_end')

        return self

    def predict_proba(self, X):
        y_proba = self.forward(X, training_behavior=False)
        y_proba = to_numpy(y_proba)
        return y_proba

    def forward(self, X, training_behavior=False):
        self.module_.train(training_behavior)

        iterator = self.get_iterator(X)
        y_probas = []
        for x in iterator:
            x = to_var(x, use_cuda=self.use_cuda)
            y_probas.append(self.module_(x))
        return torch.cat(y_probas, dim=0)

    def predict(self, X):
        self.module_.train(False)
        return self.predict_proba(X).argmax(1)

    def get_optimizer(self):
        kwargs = self._get_params_for('optim')
        if 'lr' not in kwargs:
            kwargs['lr'] = self.lr
        return self.optim(self.module_.parameters(), **kwargs)

    def get_loss(self, y_pred, y_true, train=False):
        return self.criterion_(y_pred, y_true)

    def get_iterator(self, X, y=None, train=False):
        X = to_tensor(X, use_cuda=self.use_cuda)
        if y is None:
            dataset = X
        else:
            y = to_tensor(y, use_cuda=self.use_cuda)
            dataset = TensorDataset(X, y)

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

    def __getstate__(self):
        state = BaseEstimator.__getstate__(self)
        module_ = state.pop('module_')
        module_dump = pickle.dumps(module_)
        state['module_'] = module_dump
        return state

    def __setstate__(self, state):
        module_dump = state.pop('module_')
        module_ = pickle.loads(module_dump)
        state['module_'] = module_
        BaseEstimator.__setstate__(self, state)


class NeuralNetClassifier(NeuralNet):
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
