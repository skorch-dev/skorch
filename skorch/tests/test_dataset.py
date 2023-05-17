"""Tests for dataset.py."""

import unittest
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from skorch.utils import data_from_dataset
from skorch.utils import is_torch_data_type
from skorch.utils import to_tensor
from skorch.tests.conftest import pandas_installed


class TestGetLen:
    @pytest.fixture
    def get_len(self):
        from skorch.dataset import get_len
        return get_len

    @pytest.mark.parametrize('data, expected', [
        (np.zeros(5), 5),
        (np.zeros((3, 4, 5)), 3),
        ([np.zeros(5), np.zeros((5, 4))], 5),
        ((np.zeros((5, 4)), np.zeros(5)), 5),
        ({'0': np.zeros(3), '1': np.zeros((3, 4))}, 3),
        (torch.zeros(5), 5),
        (torch.zeros((3, 4, 5)), 3),
        ([torch.zeros(5), torch.zeros((5, 4))], 5),
        ((torch.zeros((5, 4)), torch.zeros(5)), 5),
        (sparse.csr_matrix(np.zeros((5, 3))), 5),
        ({'0': torch.zeros(3), '1': torch.zeros((3, 4))}, 3),
        ([0, 1, 2], 3),
        ([[0, 1, 2], [3, 4, 5]], 3),
        ({'0': [0, 1, 2], '1': (3, 4, 5)}, 3),
        ((
            [0, 1, 2],
            np.zeros(3),
            torch.zeros(3),
            sparse.csr_matrix(np.zeros((3, 5))),
            {'0': (1, 2, 3)}),
         3),
    ])
    def test_valid_lengths(self, get_len, data, expected):
        length = get_len(data)
        assert length == expected

    @pytest.mark.parametrize('data', [
        [np.zeros(5), np.zeros((4, 5))],
        {'0': np.zeros(3), '1': np.zeros((4, 3))},
        [torch.zeros(5), torch.zeros((4, 5))],
        {'0': torch.zeros(3), '1': torch.zeros((4, 3))},
        [[0, 1, 2], [3, 4]],
        ([0, 1, 2], [3, 4]),
        {'0': [0, 1, 2], '1': (3, 4)},
        ([0, 1, 2], np.zeros(3), torch.zeros(2), {'0': (1, 2, 3)}),
    ])
    def test_inconsistent_lengths(self, get_len, data):
        with pytest.raises(ValueError):
            get_len(data)

    def test_get_len_transformers_tokenizer(self, get_len):
        transformers = pytest.importorskip('transformers')

        X = ['hello there'] * 10
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(X)
        assert get_len(tokens) == 10


class TestNetWithoutY:
    net_fixture_params = [
        {'classification': True, 'batch_size': 1},
        {'classification': False, 'batch_size': 1},
        {'classification': True, 'batch_size': 2},
        {'classification': False, 'batch_size': 2},
    ]

    @pytest.fixture
    def net_cls_1d(self):
        from skorch.toy import make_regressor
        return make_regressor(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

    @pytest.fixture
    def net_cls_2d(self):
        from skorch.toy import make_regressor
        return make_regressor(
            input_units=2,
            num_hidden=0,
            output_units=1,
        )

    @pytest.fixture
    def loader_clf(self):
        class Loader(torch.utils.data.DataLoader):
            def __iter__(self):
                z = super().__iter__()
                return ((x, torch.zeros(x.size(0)).long()) for x, _ in z)
        return Loader

    @pytest.fixture
    def loader_reg(self):
        class Loader(torch.utils.data.DataLoader):
            def __iter__(self):
                z = super().__iter__()
                return ((x, torch.zeros(x.size(0), 1).float()) for x, _ in z)
        return Loader

    @pytest.fixture
    def train_split(self):
        from skorch.dataset import ValidSplit
        return ValidSplit(0.2, stratified=False)

    @pytest.fixture(params=net_fixture_params)
    def net_1d(self, request, net_cls_1d, train_split):
        if request.param['classification']:
            from skorch import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
        else:
            from skorch import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor

        return wrap_cls(
            net_cls_1d,
            max_epochs=2,
            train_split=train_split,
            batch_size=request.param['batch_size']
        )

    @pytest.fixture(params=net_fixture_params)
    def net_2d(self, request, net_cls_2d, train_split):
        if request.param['classification']:
            from skorch import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
        else:
            from skorch import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor

        return wrap_cls(
            net_cls_2d,
            max_epochs=2,
            train_split=train_split,
            batch_size=request.param['batch_size']
        )

    @pytest.fixture(params=net_fixture_params)
    def net_1d_custom_loader(self, request, net_cls_1d,
                             loader_clf, loader_reg, train_split):
        """Parametrized fixture returning a NeuralNet
        classifier/regressor, for different batch sizes, working on 1d
        data.

        """
        if request.param['classification']:
            from skorch import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
            loader = loader_clf
        else:
            from skorch import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor
            loader = loader_reg

        return wrap_cls(
            net_cls_1d,
            iterator_train=loader,
            iterator_valid=loader,
            max_epochs=2,
            train_split=train_split,
            batch_size=request.param['batch_size']
        )

    @pytest.fixture(params=net_fixture_params)
    def net_2d_custom_loader(self, request, net_cls_2d,
                             loader_clf, loader_reg, train_split):
        """Parametrized fixture returning a NeuralNet
        classifier/regressor, for different batch sizes, working on 2d
        data.

        """
        if request.param['classification']:
            from skorch import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
            loader = loader_clf
        else:
            from skorch import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor
            loader = loader_reg

        return wrap_cls(
            net_cls_2d,
            iterator_train=loader,
            iterator_valid=loader,
            max_epochs=2,
            train_split=train_split,
            batch_size=request.param['batch_size']
        )

    def test_net_1d_tensor_raises_error(self, net_1d):
        X = torch.arange(0, 8).view(-1, 1).long()
        # We expect check_data to throw an exception
        # because we did not specify a custom data loader.
        with pytest.raises(ValueError):
            net_1d.fit(X, None)

    def test_net_2d_tensor_raises_error(self, net_2d):
        X = torch.arange(0, 8).view(4, 2).long()
        # We expect check_data to throw an exception
        # because we did not specify a custom data loader.
        with pytest.raises(ValueError):
            net_2d.fit(X, None)

    def test_net_1d_custom_loader(self, net_1d_custom_loader):
        X = torch.arange(0, 8).view(-1, 1).float()
        # throw away all callbacks since those may raise unrelated errors
        net_1d_custom_loader.initialize()
        net_1d_custom_loader.callbacks_ = []
        # Should not raise an exception.
        net_1d_custom_loader.partial_fit(X, None)

    def test_net_2d_custom_loader(self, net_2d_custom_loader):
        X = torch.arange(0, 8).view(4, 2).float()
        # throw away all callbacks since those may raise unrelated errors
        net_2d_custom_loader.initialize()
        net_2d_custom_loader.callbacks_ = []
        # Should not raise an exception.
        net_2d_custom_loader.partial_fit(X, None)


class TestNetWithDict:
    @pytest.fixture(scope='module')
    def module_cls(self):
        """Return a simple module that concatenates its 2 inputs in
        forward step.

        """
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

            # pylint: disable=arguments-differ
            def forward(self, X0, X1):
                X = torch.cat((X0, X1), 1)
                X = F.softmax(self.dense(X), dim=-1)
                return X

        return MyModule

    @pytest.fixture(scope='module')
    def data(self):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        return X[:, :10], X[:, 10:], y

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            max_epochs=2,
            lr=0.1,
        )

    def test_fit_predict_proba(self, net, data):
        X = {'X0': data[0], 'X1': data[1]}
        y = data[2]
        net.fit(X, y)
        y_proba = net.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1)

        # Issue #142: check that all batch sizes are consistent with
        # `net.batch_size`, even when the input type is a dictionary.
        # Note that we allow for different batch sizes as the total
        # number of samples may not be divisible by the batch size.
        batch_sizes = lambda n: set(sum(net.history[:, 'batches', :, n], []))
        train_batch_sizes = batch_sizes('train_batch_size')
        valid_batch_sizes = batch_sizes('valid_batch_size')
        assert net.batch_size in train_batch_sizes
        assert net.batch_size in valid_batch_sizes


class TestNetWithList:
    @pytest.fixture(scope='module')
    def module_cls(self):
        """Return a simple module that concatenates the input."""
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

            # pylint: disable=arguments-differ
            def forward(self, X):
                X = torch.cat(X, 1)
                X = F.softmax(self.dense(X), dim=-1)
                return X

        return MyModule

    @pytest.fixture(scope='module')
    def data(self):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        return [X[:, :10], X[:, 10:]], y

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            max_epochs=2,
            lr=0.1,
        )

    def test_fit_predict_proba(self, net, data):
        X, y = data
        net.fit(X, y)
        y_proba = net.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1)


@pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
class TestNetWithPandas:
    @pytest.fixture(scope='module')
    def module_cls(self):
        """Return a simple module that concatenates all input values
        in forward step.

        """
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

            # pylint: disable=arguments-differ
            def forward(self, **X):
                X = torch.cat(list(X.values()), 1)
                X = F.softmax(self.dense(X), dim=-1)
                return X

        return MyModule

    @pytest.fixture(scope='module')
    def pd(self):
        import pandas as pd
        return pd

    @pytest.fixture(scope='module')
    def data(self, pd):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        df = pd.DataFrame(X, columns=map(str, range(X.shape[1])))
        return df, y

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            max_epochs=2,
            lr=0.1,
        )

    def test_fit_predict_proba(self, net, data):
        X, y = data
        net.fit(X, y)
        y_proba = net.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1)


class TestNetWithTokenizers:
    """Huggingface tokenizers should work without special adjustments"""
    @pytest.fixture(scope='session')
    def tokenizer(self):
        transformers = pytest.importorskip('transformers')
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer

    @pytest.fixture(scope='session')
    def data(self, tokenizer):
        """A simple dataset that the model should be able to learn (or overfit)
        on

        """
        X = [paragraph for paragraph in unittest.__doc__.split('\n') if paragraph]
        Xt = tokenizer(
            X,
            max_length=12,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        y = np.array(['test' in x.lower() for x in X], dtype=np.int64)
        return Xt, y

    @pytest.fixture(scope='session')
    def module_cls(self, tokenizer):
        """Return a simple module using embedding + linear + softmax instead of
        a full-fledged BERT module.

        """
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(tokenizer.vocab_size, 6)
                self.dense = nn.Linear(6, 2)
                self.sm = nn.Softmax(dim=-1)

            # pylint: disable=arguments-differ
            def forward(self, input_ids, attention_mask):
                assert input_ids.shape == attention_mask.shape
                X = self.emb(input_ids).mean(1)
                return self.sm(self.dense(X))

        return MyModule

    @pytest.fixture(scope='session')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def net(self, net_cls, module_cls):
        return net_cls(
            module_cls,
            optimizer=torch.optim.Adam,
            max_epochs=5,
            batch_size=8,
            lr=0.1,
        )

    def test_fit_predict_proba(self, net, data):
        X, y = data
        net.fit(X, y)
        y_proba = net.predict_proba(X)
        assert np.allclose(y_proba.sum(1), 1)

        train_losses = net.history[:, 'train_loss']
        # make sure the network trained successfully with an arbitrary wide margin
        assert train_losses[0] > 5 * train_losses[-1]


class TestDataset:
    """Note: we don't need to test multi_indexing here, since that is
    already covered.

    """
    @pytest.fixture
    def dataset_cls(self):
        from skorch.dataset import Dataset
        return Dataset

    def test_len_correct(self, dataset_cls):
        pass

    def test_user_defined_len(self, dataset_cls):
        pass

    def test_inconsistent_lengths_raises(self, dataset_cls):
        pass

    def test_with_numpy_array(self, dataset_cls):
        pass

    def test_with_torch_tensor(self, dataset_cls):
        pass

    @pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
    def test_with_pandas_df(self, dataset_cls):
        pass

    @pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
    def test_with_pandas_series(self, dataset_cls):
        pass

    def test_with_dict(self, dataset_cls):
        pass

    def test_with_list_of_numpy_arrays(self, dataset_cls):
        pass

    @pytest.fixture
    def dataset_sparse_csr(self, dataset_cls):
        Xs = sparse.csr_matrix(np.zeros((10, 5)))
        return dataset_cls(Xs)

    @pytest.mark.parametrize('batch_size', [1, 3, 10, 17])
    def test_dataloader_with_sparse_csr(self, dataset_sparse_csr, batch_size):
        loader = DataLoader(dataset_sparse_csr, batch_size=batch_size)
        for Xb, _ in loader:
            assert is_torch_data_type(Xb)


class TestTrainSplitIsUsed:
    @pytest.fixture
    def iterator(self):
        """Return a simple iterator that yields the input data."""
        class Iterator:
            """An iterator that just yield the input data."""
            # pylint: disable=unused-argument
            def __init__(self, dataset, *args, **kwargs):
                self.dataset = dataset

            def __iter__(self):
                yield self.dataset.X, self.dataset.y

        return Iterator

    @pytest.fixture
    def data(self):
        X = torch.arange(0, 12, dtype=torch.float32).view(4, 3)
        y = torch.LongTensor([0, 1, 1, 0])
        return X, y

    @pytest.fixture
    def data_split(self, data):
        from skorch.dataset import Dataset

        X, y = data
        dataset_train = Dataset(X[:2], y[:2])
        dataset_valid = Dataset(X[2:], y[2:])
        return dataset_train, dataset_valid

    @pytest.fixture
    def module(self, classifier_module):
        return classifier_module

    @pytest.fixture
    def train_split(self, data_split):
        return Mock(side_effect=[data_split])

    @pytest.fixture
    def net_and_mock(self, module, data, train_split, iterator):
        """Return a NeuralNetClassifier with mocked train and
        validation step which save the args and kwargs the methods are
        calld with.

        """
        from skorch import NeuralNetClassifier

        X, y = data
        net = NeuralNetClassifier(
            module,
            module__input_units=3,
            max_epochs=1,
            iterator_train=iterator,
            iterator_valid=iterator,
            train_split=train_split
        )
        net.initialize()
        net.callbacks_ = []

        mock = Mock()

        def decorator(func):
            def wrapper(*args, **kwargs):
                mock(*args, **kwargs)
                func.__dict__['mock_'] = mock
                return func(*args[1:], **kwargs)
            return wrapper

        import types
        net.get_iterator = types.MethodType(decorator(net.get_iterator), net)
        return net.partial_fit(X, y), mock

    def test_steps_called_with_split_data(self, net_and_mock, data_split):
        mock = net_and_mock[1]
        assert mock.call_count == 2  # once for train, once for valid
        assert mock.call_args_list[0][0][1] == data_split[0]
        assert mock.call_args_list[0][1]['training'] is True
        assert mock.call_args_list[1][0][1] == data_split[1]
        assert mock.call_args_list[1][1]['training'] is False


class TestValidSplit:
    num_samples = 100

    @staticmethod
    def assert_datasets_equal(ds0, ds1):
        """Generic function to test equality of dataset values."""
        assert len(ds0) == len(ds1)
        # pylint: disable=consider-using-enumerate
        for i in range(len(ds0)):
            x0, y0 = ds0[i]
            x1, y1 = ds1[i]
            try:
                assert x0 == x1
            except (RuntimeError, ValueError):
                assert (x0 == x1).all()
            try:
                assert y0 == y1
            except (RuntimeError, ValueError):
                assert (y0 == y1).all()

    @pytest.fixture
    def dataset_cls(self):
        from skorch.dataset import Dataset
        return Dataset

    @pytest.fixture
    def data(self, dataset_cls):
        X = np.random.random((self.num_samples, 10))
        assert self.num_samples % 4 == 0
        y = np.repeat([0, 1, 2, 3], self.num_samples // 4)
        return dataset_cls(X, y)

    @pytest.fixture
    def valid_split_cls(self):
        from skorch.dataset import ValidSplit
        return ValidSplit

    def test_reproducible(self, valid_split_cls, data):
        dataset_train0, dataset_valid0 = valid_split_cls(5)(data)
        dataset_train1, dataset_valid1 = valid_split_cls(5)(data)
        self.assert_datasets_equal(dataset_train0, dataset_train1)
        self.assert_datasets_equal(dataset_valid0, dataset_valid1)

    @pytest.mark.parametrize('cv', [2, 4, 5, 10])
    def test_different_kfolds(self, valid_split_cls, cv, data):
        if self.num_samples % cv != 0:
            raise ValueError("Num samples not divisible by {}".format(cv))

        dataset_train, dataset_valid = valid_split_cls(cv)(data)
        assert len(dataset_train) + len(dataset_valid) == self.num_samples
        assert len(dataset_valid) == self.num_samples // cv

    @pytest.mark.parametrize('cv', [5, 0.2])
    def test_stratified(self, valid_split_cls, data, cv):
        num_expected = self.num_samples // 4
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        data.y = y

        dataset_train, dataset_valid = valid_split_cls(
            cv, stratified=True)(data, y)
        y_train = data_from_dataset(dataset_train)[1]
        y_valid = data_from_dataset(dataset_valid)[1]

        assert y_train.sum() == 0.8 * num_expected
        assert y_valid.sum() == 0.2 * num_expected

    @pytest.mark.parametrize('cv', [0.1, 0.2, 0.5, 0.75])
    def test_different_fractions(self, valid_split_cls, cv, data):
        if not (self.num_samples * cv).is_integer() != 0:
            raise ValueError("Num samples cannot be evenly distributed for "
                             "fraction {}".format(cv))

        dataset_train, dataset_valid = valid_split_cls(cv)(data)
        assert len(dataset_train) + len(dataset_valid) == self.num_samples
        assert len(dataset_valid) == self.num_samples * cv

    @pytest.mark.parametrize('cv', [0.1, 0.2, 0.5, 0.75])
    def test_fraction_no_y(self, valid_split_cls, data, cv):
        if not (self.num_samples * cv).is_integer() != 0:
            raise ValueError("Num samples cannot be evenly distributed for "
                             "fraction {}".format(cv))

        m = int(cv * self.num_samples)
        n = int((1 - cv) * self.num_samples)
        dataset_train, dataset_valid = valid_split_cls(
            cv, stratified=False)(data, None)
        assert len(dataset_valid) == m
        assert len(dataset_train) == n

    def test_fraction_no_classifier(self, valid_split_cls, data):
        y = np.random.random(self.num_samples)
        data.y = y

        cv = 0.2
        m = int(cv * self.num_samples)
        n = int((1 - cv) * self.num_samples)
        dataset_train, dataset_valid = valid_split_cls(
            cv, stratified=False)(data, y)

        assert len(dataset_valid) == m
        assert len(dataset_train) == n

    @pytest.mark.parametrize('cv', [0, -0.001, -0.2, -3])
    def test_bad_values_raise(self, valid_split_cls, cv):
        with pytest.raises(ValueError) as exc:
            valid_split_cls(cv)

        expected = ("Numbers less than 0 are not allowed for cv "
                    "but ValidSplit got {}".format(cv))
        assert exc.value.args[0] == expected

    @pytest.mark.parametrize('cv', [5, 0.2])
    def test_not_stratified(self, valid_split_cls, data, cv):
        num_expected = self.num_samples // 4
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        data.y = y

        dataset_train, dataset_valid = valid_split_cls(
            cv, stratified=False)(data, y)
        y_train = data_from_dataset(dataset_train)[1]
        y_valid = data_from_dataset(dataset_valid)[1]

        # when not stratified, we cannot know the distribution of targets
        assert y_train.sum() + y_valid.sum() == num_expected

    def test_predefined_split(self, valid_split_cls, data):
        from sklearn.model_selection import PredefinedSplit
        indices = (data.y > 0).astype(int)
        split = PredefinedSplit(indices)

        dataset_train, dataset_valid = valid_split_cls(split)(data)
        y_train = data_from_dataset(dataset_train)[1]
        y_valid = data_from_dataset(dataset_valid)[1]

        assert (y_train > 0).all()
        assert (y_valid == 0).all()

    def test_with_y_none(self, valid_split_cls, data):
        data.y = None
        m = self.num_samples // 5
        n = self.num_samples - m
        dataset_train, dataset_valid = valid_split_cls(5)(data)

        assert len(dataset_train) == n
        assert len(dataset_valid) == m

        y_train = data_from_dataset(dataset_train)[1]
        y_valid = data_from_dataset(dataset_valid)[1]

        assert y_train is None
        assert y_valid is None

    def test_with_torch_tensors(self, valid_split_cls, data):
        data.X = to_tensor(data.X, device='cpu')
        data.y = to_tensor(data.y, device='cpu')
        m = self.num_samples // 5
        n = self.num_samples - m
        dataset_train, dataset_valid = valid_split_cls(5)(data)

        assert len(dataset_valid) == m
        assert len(dataset_train) == n

    def test_with_torch_tensors_and_stratified(self, valid_split_cls, data):
        num_expected = self.num_samples // 4
        data.X = to_tensor(data.X, device='cpu')
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        data.y = to_tensor(y, device='cpu')

        dataset_train, dataset_valid = valid_split_cls(5, stratified=True)(data, y)
        y_train = data_from_dataset(dataset_train)[1]
        y_valid = data_from_dataset(dataset_valid)[1]

        assert y_train.sum() == 0.8 * num_expected
        assert y_valid.sum() == 0.2 * num_expected

    def test_with_list_of_arrays(self, valid_split_cls, data):
        data.X = [data.X, data.X]
        m = self.num_samples // 5
        n = self.num_samples - m

        dataset_train, dataset_valid = valid_split_cls(5)(data)
        X_train, y_train = data_from_dataset(dataset_train)
        X_valid, y_valid = data_from_dataset(dataset_valid)

        assert len(X_train[0]) == len(X_train[1]) == len(y_train) == n
        assert len(X_valid[0]) == len(X_valid[1]) == len(y_valid) == m

    def test_with_dict(self, valid_split_cls, data):
        data.X = {'1': data.X, '2': data.X}
        dataset_train, dataset_valid = valid_split_cls(5)(data)

        m = self.num_samples // 5
        n = self.num_samples - m

        X_train, y_train = data_from_dataset(dataset_train)
        X_valid, y_valid = data_from_dataset(dataset_valid)

        assert len(X_train['1']) == len(X_train['2']) == len(y_train) == n
        assert len(X_valid['1']) == len(X_valid['2']) == len(y_valid) == m

    @pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
    def test_with_pandas(self, valid_split_cls, data):
        import pandas as pd

        data.X = pd.DataFrame(
            data.X,
            columns=[str(i) for i in range(data.X.shape[1])],
        )
        dataset_train, dataset_valid = valid_split_cls(5)(data)

        m = self.num_samples // 5
        X_train, y_train = data_from_dataset(dataset_train)
        X_valid, y_valid = data_from_dataset(dataset_valid)

        assert len(X_train) + len(X_valid) == self.num_samples
        assert len(y_train) + len(y_valid) == self.num_samples
        assert len(X_valid) == len(y_valid) == m

    def test_y_str_val_stratified(self, valid_split_cls, data):
        y = np.array(['a', 'a', 'a', 'b'] * (self.num_samples // 4))
        if len(data.X) != len(y):
            raise ValueError
        data.y = y

        dataset_train, dataset_valid = valid_split_cls(
            5, stratified=True)(data, y)
        y_train = data_from_dataset(dataset_train)[1]
        y_valid = data_from_dataset(dataset_valid)[1]

        assert np.isclose(np.mean(y_train == 'b'), 0.25)
        assert np.isclose(np.mean(y_valid == 'b'), 0.25)

    def test_y_list_of_arr_does_not_raise(self, valid_split_cls, data):
        y = [np.zeros(self.num_samples), np.ones(self.num_samples)]
        data.y = y
        valid_split_cls(5, stratified=False)(data)

    def test_y_list_of_arr_stratified(self, valid_split_cls, data):
        y = [np.zeros(self.num_samples), np.ones(self.num_samples)]
        data.y = y
        with pytest.raises(ValueError) as exc:
            valid_split_cls(5, stratified=True)(data, y)

        expected = "Stratified CV requires explicitly passing a suitable y."
        assert exc.value.args[0] == expected

    def test_y_dict_does_not_raise(self, valid_split_cls, data):
        y = {'a': np.zeros(self.num_samples), 'b': np.ones(self.num_samples)}
        data.y = y

        valid_split_cls(5, stratified=False)(data)

    def test_y_dict_stratified_raises(self, valid_split_cls, data):
        X = data[0]
        y = {'a': np.zeros(len(X)), 'b': np.ones(len(X))}

        with pytest.raises(ValueError):
            # an sklearn error is raised
            valid_split_cls(5, stratified=True)(X, y)

    @pytest.mark.parametrize('cv', [5, 0.2])
    @pytest.mark.parametrize('X', [np.zeros((100, 10)), torch.zeros((100, 10))])
    def test_y_none_stratified(self, valid_split_cls, data, cv, X):
        data.X = X
        with pytest.raises(ValueError) as exc:
            valid_split_cls(cv, stratified=True)(data, None)

        expected = "Stratified CV requires explicitly passing a suitable y."
        assert exc.value.args[0] == expected

    def test_shuffle_split_reproducible_with_random_state(
            self, valid_split_cls, dataset_cls):
        n = self.num_samples
        X, y = np.random.random((n, 10)), np.random.randint(0, 10, size=n)
        cv = valid_split_cls(0.2, stratified=False)

        dst0, dsv0 = cv(dataset_cls(X, y))
        dst1, dsv1 = cv(dataset_cls(X, y))

        Xt0, yt0 = data_from_dataset(dst0)
        Xv0, yv0 = data_from_dataset(dsv0)
        Xt1, yt1 = data_from_dataset(dst1)
        Xv1, yv1 = data_from_dataset(dsv1)

        assert not np.allclose(Xt0, Xt1)
        assert not np.allclose(Xv0, Xv1)
        assert not np.allclose(yt0, yt1)
        assert not np.allclose(yv0, yv1)

    def test_group_kfold(self, valid_split_cls, data):
        from sklearn.model_selection import GroupKFold

        X, y = data.X, data.y
        n = self.num_samples // 2
        groups = np.asarray(
            [0 for _ in range(n)] + [1 for _ in range(self.num_samples - n)])

        dataset_train, dataset_valid = valid_split_cls(
            GroupKFold(n_splits=2))(data, groups=groups)
        X_train, y_train = data_from_dataset(dataset_train)
        X_valid, y_valid = data_from_dataset(dataset_valid)

        assert np.allclose(X[:n], X_train)
        assert np.allclose(y[:n], y_train)
        assert np.allclose(X[n:], X_valid)
        assert np.allclose(y[n:], y_valid)

    def test_random_state_not_used_raises(self, valid_split_cls):
        # Since there is no randomness involved, raise a ValueError when
        # random_state is set, same as sklearn is now doing.
        msg = (
            r"Setting a random_state has no effect since cv is not a float. "
            r"You should leave random_state to its default \(None\), or set cv "
            r"to a float value."
        )
        with pytest.raises(ValueError, match=msg):
            valid_split_cls(5, random_state=0)

    def test_random_state_and_float_does_not_raise(self, valid_split_cls):
        valid_split_cls(0.5, random_state=0)  # does not raise
