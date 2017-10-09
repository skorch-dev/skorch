"""Tests for dataset.py."""

from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.datasets import make_classification
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

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
        ({'0': torch.zeros(3), '1': torch.zeros((3, 4))}, 3),
        ([0, 1, 2], 3),
        ([[0, 1, 2], [3, 4, 5]], 3),
        ({'0': [0, 1, 2], '1': (3, 4, 5)}, 3),
        (([0, 1, 2], np.zeros(3), torch.zeros(3), {'0': (1, 2, 3)}), 3),
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


class TestNetWithoutY:

    net_fixture_params = [
        {'classification': True, 'batch_size': 1},
        {'classification': False, 'batch_size': 1},
        {'classification': True, 'batch_size': 2},
        {'classification': False, 'batch_size': 2},
    ]

    @pytest.fixture
    def net_cls_1d(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(1, 1)

            # pylint: disable=arguments-differ
            def forward(self, X):
                return self.dense(X.float())
        return MyModule

    @pytest.fixture
    def net_cls_2d(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(2, 1)

            # pylint: disable=arguments-differ
            def forward(self, X):
                return self.dense(X.float())
        return MyModule

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
                return ((x, torch.zeros(x.size(0)).float()) for x, _ in z)
        return Loader

    @pytest.fixture
    def train_split(self):
        from skorch.dataset import CVSplit
        return CVSplit(0.2, stratified=False)

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
            iterator_test=loader,
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
            iterator_test=loader,
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
        X = torch.arange(0, 8).view(-1, 1).long()
        # Should not raise an exception.
        net_1d_custom_loader.fit(X, None)

    def test_net_2d_custom_loader(self, net_2d_custom_loader):
        X = torch.arange(0, 8).view(4, 2).long()
        # Should not raise an exception.
        net_2d_custom_loader.fit(X, None)


class TestMultiIndexing:
    @pytest.fixture
    def multi_indexing(self):
        from skorch.dataset import multi_indexing
        return multi_indexing

    @pytest.mark.parametrize('data, i, expected', [
        (
            np.arange(12).reshape(4, 3),
            slice(None),
            np.arange(12).reshape(4, 3),
        ),
        (
            np.arange(12).reshape(4, 3),
            np.s_[2],
            np.array([6, 7, 8]),
        ),
        (
            np.arange(12).reshape(4, 3),
            np.s_[-2:],
            np.array([[6, 7, 8], [9, 10, 11]]),
        ),
    ])
    def test_ndarray(self, multi_indexing, data, i, expected):
        result = multi_indexing(data, i)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize('data, i, expected', [
        (
            torch.arange(0, 12).view(4, 3),
            slice(None),
            np.arange(12).reshape(4, 3),
        ),
        (
            torch.arange(0, 12).view(4, 3),
            np.s_[2],
            np.array([6, 7, 8]),
        ),
        (
            torch.arange(0, 12).view(4, 3),
            np.s_[-2:],
            np.array([[6, 7, 8], [9, 10, 11]]),
        ),
    ])
    def test_torch_tensor(self, multi_indexing, data, i, expected):
        result = multi_indexing(data, i).long().numpy()
        assert np.allclose(result, expected)

    @pytest.mark.parametrize('data, i, expected', [
        ([1, 2, 3, 4], slice(None), [1, 2, 3, 4]),
        ([1, 2, 3, 4], slice(None, 2), [1, 2]),
        ([1, 2, 3, 4], 2, 3),
        ([1, 2, 3, 4], -2, 3),
    ])
    def test_list(self, multi_indexing, data, i, expected):
        result = multi_indexing(data, i)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize('data, i, expected', [
        ({'a': [0, 1, 2], 'b': [3, 4, 5]}, 0, {'a': 0, 'b': 3}),
        (
            {'a': [0, 1, 2], 'b': [3, 4, 5]},
            np.s_[:2],
            {'a': [0, 1], 'b': [3, 4]},
        )
    ])
    def test_dict_of_lists(self, multi_indexing, data, i, expected):
        result = multi_indexing(data, i)
        assert result == expected

    @pytest.mark.parametrize('data, i, expected', [
        (
            {'a': np.arange(3), 'b': np.arange(3, 6)},
            0,
            {'a': 0, 'b': 3}
        ),
        (
            {'a': np.arange(3), 'b': np.arange(3, 6)},
            np.s_[:2],
            {'a': np.arange(2), 'b': np.arange(3, 5)}
        ),
    ])
    def test_dict_of_arrays(self, multi_indexing, data, i, expected):
        result = multi_indexing(data, i)
        assert result.keys() == expected.keys()
        for k in result:
            assert np.allclose(result[k], expected[k])

    @pytest.mark.parametrize('data, i, expected', [
        (
            {'a': torch.arange(0, 3), 'b': torch.arange(3, 6)},
            0,
            {'a': 0, 'b': 3}
        ),
        (
            {'a': torch.arange(0, 3), 'b': torch.arange(3, 6)},
            np.s_[:2],
            {'a': np.arange(2), 'b': np.arange(3, 5)}
        ),
    ])
    def test_dict_of_torch_tensors(self, multi_indexing, data, i, expected):
        result = multi_indexing(data, i)
        assert result.keys() == expected.keys()
        for k in result:
            try:
                val = result[k].long().numpy()
            except AttributeError:
                val = result[k]
            assert np.allclose(val, expected[k])

    def test_mixed_data(self, multi_indexing):
        data = [
            [1, 2, 3],
            np.arange(3),
            torch.arange(3, 6),
            {'a': [4, 5, 6], 'b': [7, 8, 9]},
        ]
        result = multi_indexing(data, 0)
        expected = [1, 0, 3, {'a': 4, 'b': 7}]
        assert result == expected

    def test_mixed_data_slice(self, multi_indexing):
        data = [
            [1, 2, 3],
            np.arange(3),
            torch.arange(3, 6),
            {'a': [4, 5, 6], 'b': [7, 8, 9]},
        ]
        result = multi_indexing(data, np.s_[:2])
        assert result[0] == [1, 2]
        assert np.allclose(result[1], np.arange(2))
        assert np.allclose(result[2].long().numpy(), np.arange(3, 5))
        assert result[3] == {'a': [4, 5], 'b': [7, 8]}

    @pytest.fixture
    def pd(self):
        if not pandas_installed:
            pytest.skip()
        import pandas as pd
        return pd

    def test_pandas_dataframe(self, multi_indexing, pd):
        df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=[2, 1, 0])
        result = multi_indexing(df, 0)
        # Note: taking one row of a DataFrame returns a Series
        expected = pd.Series(data=[0, 3], index=['a', 'b'], name=2)
        assert result.equals(expected)

    def test_pandas_dataframe_slice(self, multi_indexing, pd):
        import pandas as pd
        df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=[2, 1, 0])
        result = multi_indexing(df, np.s_[:2])
        expected = pd.DataFrame({'a': [0, 1], 'b': [3, 4]}, index=[2, 1])
        assert result.equals(expected)

    def test_pandas_series(self, multi_indexing, pd):
        series = pd.Series(data=[0, 1, 2], index=[2, 1, 0])
        result = multi_indexing(series, 0)
        assert result == 0

    def test_pandas_series_slice(self, multi_indexing, pd):
        series = pd.Series(data=[0, 1, 2], index=[2, 1, 0])
        result = multi_indexing(series, np.s_[:2])
        expected = pd.Series(data=[0, 1], index=[2, 1])
        assert result.equals(expected)

    def test_list_of_dataframe_and_series(self, multi_indexing, pd):
        data = [
            pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=[2, 1, 0]),
            pd.Series(data=[0, 1, 2], index=[2, 1, 0]),
        ]
        result = multi_indexing(data, 0)
        assert result[0].equals(
            pd.Series(data=[0, 3], index=['a', 'b'], name=2))
        assert result[1] == 0

    def test_list_of_dataframe_and_series_slice(self, multi_indexing, pd):
        data = [
            pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=[2, 1, 0]),
            pd.Series(data=[0, 1, 2], index=[2, 1, 0]),
        ]
        result = multi_indexing(data, np.s_[:2])
        assert result[0].equals(
            pd.DataFrame({'a': [0, 1], 'b': [3, 4]}, index=[2, 1]))
        assert result[1].equals(pd.Series(data=[0, 1], index=[2, 1]))

    def test_index_torch_tensor_with_numpy_int_array(self, multi_indexing):
        X = torch.zeros((1000, 10))
        i = np.arange(100)
        result = multi_indexing(X, i)
        assert (result == X[:100]).all()

    def test_index_torch_tensor_with_numpy_bool_array(self, multi_indexing):
        X = torch.zeros((1000, 10))
        i = np.asarray([True] * 100 + [False] * 900)
        result = multi_indexing(X, i)
        assert (result == X[:100]).all()

    def test_index_with_float_array_raises(self, multi_indexing):
        X = np.zeros(10)
        i = np.arange(3, 0.5)
        with pytest.raises(IndexError) as exc:
            multi_indexing(X, i)

        assert exc.value.args[0] == (
            "arrays used as indices must be of integer (or boolean) type")

    def test_boolean_index_2d(self, multi_indexing):
        X = np.arange(9).reshape(3, 3)
        i = np.eye(3).astype(bool)
        result = multi_indexing(X, i)
        expected = np.asarray([0, 4, 8])
        assert np.allclose(result, expected)

    def test_boolean_index_2d_with_torch_tensor(self, multi_indexing):
        X = torch.LongTensor(np.arange(9).reshape(3, 3))
        i = np.eye(3).astype(bool)

        res = multi_indexing(X, i)
        expected = torch.LongTensor([0, 4, 8])
        assert all(res == expected)


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
                X = F.softmax(self.dense(X))
                return X

        return MyModule

    @pytest.fixture(scope='module')
    def data(self):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        return X[:, :10], X[:, 10:], y

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch.net import NeuralNetClassifier
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


class TestNetWithList:
    @pytest.fixture(scope='module')
    def module_cls(self):
        """Return a simple module that concatenates in input."""
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

            # pylint: disable=arguments-differ
            def forward(self, X):
                X = torch.cat(X, 1)
                X = F.softmax(self.dense(X))
                return X

        return MyModule

    @pytest.fixture(scope='module')
    def data(self):
        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        return [X[:, :10], X[:, 10:]], y

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch.net import NeuralNetClassifier
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
                X = F.softmax(self.dense(X))
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
        from skorch.net import NeuralNetClassifier
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
        X = torch.arange(0, 12).view(4, 3)
        y = torch.LongTensor([0, 1, 1, 0])
        return X, y

    @pytest.fixture
    def data_split(self, data):
        X, y = data
        return X[:2], X[2:], y[:2], y[2:]

    @pytest.fixture
    def module(self):
        """Return a simple classifier module class."""
        class MyClassifier(nn.Module):
            """Simple classifier module"""
            def __init__(self, input_units=20, num_units=10, nonlin=F.relu):
                super(MyClassifier, self).__init__()

                self.dense0 = nn.Linear(input_units, num_units)
                self.nonlin = nonlin
                self.dropout = nn.Dropout(0.5)
                self.dense1 = nn.Linear(num_units, 10)
                self.output = nn.Linear(10, 2)

            # pylint: disable=arguments-differ
            def forward(self, X):
                X = self.nonlin(self.dense0(X))
                X = self.dropout(X)
                X = self.nonlin(self.dense1(X))
                X = F.softmax(self.output(X))
                return X
        return MyClassifier

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
            iterator_test=iterator,
            train_split=train_split
        )

        mock = Mock()

        def decorator(func):
            def wrapper(*args, **kwargs):
                mock(*args, **kwargs)
                func.__dict__['mock_'] = mock
                return func(*args[1:], **kwargs)
            return wrapper

        import types
        net.train_step = types.MethodType(decorator(net.train_step), net)
        net.validation_step = types.MethodType(
            decorator(net.validation_step), net)
        return net.fit(X, y), mock

    def test_steps_called_with_split_data(self, net_and_mock, data_split):
        mock = net_and_mock[1]
        assert mock.call_count == 2  # once for train, once for valid
        assert (mock.call_args_list[0][0][1] == data_split[0]).all()
        assert (mock.call_args_list[0][0][2] == data_split[2]).all()
        assert (mock.call_args_list[1][0][1] == data_split[1]).all()
        assert (mock.call_args_list[1][0][2] == data_split[3]).all()


class TestCVSplit:
    num_samples = 100

    @pytest.fixture
    def data(self):
        X = np.random.random((self.num_samples, 10))
        assert self.num_samples % 4 == 0
        y = np.repeat([0, 1, 2, 3], self.num_samples // 4)
        return X, y

    @pytest.fixture
    def cv_split_cls(self):
        from skorch.dataset import CVSplit
        return CVSplit

    def test_reproducible(self, cv_split_cls, data):
        X, y = data
        X_train1, X_valid1, y_train1, y_valid1 = cv_split_cls(5)(X, y)
        X_train2, X_valid2, y_train2, y_valid2 = cv_split_cls(5)(X, y)
        assert np.all(X_train1 == X_train2)
        assert np.all(X_valid1 == X_valid2)
        assert np.all(y_train1 == y_train2)
        assert np.all(y_valid1 == y_valid2)

    @pytest.mark.parametrize('cv', [2, 4, 5, 10])
    def test_different_kfolds(self, cv_split_cls, cv, data):
        if self.num_samples % cv != 0:
            raise ValueError("Num samples not divisible by {}".format(cv))

        X, y = data
        X_train, X_valid, y_train, y_valid = cv_split_cls(cv)(X, y)
        assert len(X_train) + len(X_valid) == self.num_samples
        assert len(y_train) + len(y_valid) == self.num_samples
        assert len(X_valid) == len(y_valid) == self.num_samples // cv

    def test_stratified(self, cv_split_cls, data):
        X = data[0]
        num_expected = self.num_samples // 4
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        _, _, y_train, y_valid = cv_split_cls(5, stratified=True)(X, y)
        assert y_train.sum() == 0.8 * num_expected
        assert y_valid.sum() == 0.2 * num_expected

    @pytest.mark.parametrize('cv', [0.1, 0.2, 0.5, 0.75])
    def test_different_fractions(self, cv_split_cls, cv, data):
        if not (self.num_samples * cv).is_integer() != 0:
            raise ValueError("Num samples cannot be evenly distributed for "
                             "fraction {}".format(cv))

        X, y = data
        X_train, X_valid, y_train, y_valid = cv_split_cls(cv)(X, y)
        assert len(X_train) + len(X_valid) == self.num_samples
        assert len(y_train) + len(y_valid) == self.num_samples
        assert len(X_valid) == len(y_valid) == self.num_samples * cv

    def test_stratified_fraction(self, cv_split_cls, data):
        X = data[0]
        frac = 0.2
        num_expected = self.num_samples // 4
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        _, _, y_train, y_valid = cv_split_cls(frac, stratified=True)(X, y)
        assert y_train.sum() == 0.8 * num_expected
        assert y_valid.sum() == 0.2 * num_expected

    @pytest.mark.parametrize('cv', [0.1, 0.2, 0.5, 0.75])
    def test_fraction_no_y(self, cv_split_cls, data, cv):
        if not (self.num_samples * cv).is_integer() != 0:
            raise ValueError("Num samples cannot be evenly distributed for "
                             "fraction {}".format(cv))

        X = data[0]
        m = int(cv * self.num_samples)
        n = int((1 - cv) * self.num_samples)
        X_train, X_valid, _, _ = cv_split_cls(cv, stratified=False)(X, None)
        assert len(X_valid) == m
        assert len(X_train) == n

    def test_fraction_no_classifier(self, cv_split_cls, data):
        X = data[0]
        y = np.random.random(len(X))
        cv = 0.2
        m = int(cv * self.num_samples)
        n = int((1 - cv) * self.num_samples)
        X_train, X_valid, y_train, y_valid = cv_split_cls(
            cv, stratified=False)(X, y)
        assert len(X_valid) == len(y_valid) == m
        assert len(X_train) == len(y_train) == n

    @pytest.mark.parametrize('cv', [0, -0.001, -0.2, -3])
    def test_bad_values_raise(self, cv_split_cls, cv):
        with pytest.raises(ValueError) as exc:
            cv_split_cls(cv)

        expected = ("Numbers less than 0 are not allowed for cv "
                    "but CVSplit got {}".format(cv))
        assert exc.value.args[0] == expected

    def test_not_stratified(self, cv_split_cls, data):
        X = data[0]
        num_expected = self.num_samples // 4
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        _, _, y_train, y_valid = cv_split_cls(5, stratified=False)(X, y)
        assert y_train.sum() == num_expected
        assert y_valid.sum() == 0

    def test_predefined_split(self, cv_split_cls, data):
        X, y = data

        from sklearn.model_selection import PredefinedSplit
        indices = (y > 0).astype(int)
        split = PredefinedSplit(indices)

        _, _, y_train, y_valid = cv_split_cls(split)(X, y)
        assert (y_train > 0).all()
        assert (y_valid == 0).all()

    def test_with_y_none(self, cv_split_cls, data):
        X = data[0]
        y = None
        m = self.num_samples // 5
        n = self.num_samples - m

        X_train, X_valid, y_train, y_valid = cv_split_cls(5)(X, y)
        assert len(X_train) == n
        assert len(X_valid) == m
        assert y_train is None
        assert y_valid is None

    def test_with_torch_tensors(self, cv_split_cls, data):
        X, y = data
        X = to_tensor(X)
        y = to_tensor(y)
        m = self.num_samples // 5
        n = self.num_samples - m

        X_train, X_valid, y_train, y_valid = cv_split_cls(5)(X, y)
        assert len(X_train) == len(y_train) == n
        assert len(X_valid) == len(y_valid) == m

    def test_with_torch_tensors_and_stratified(self, cv_split_cls, data):
        X = to_tensor(data[0])
        num_expected = self.num_samples // 4
        y = np.hstack([np.repeat([0, 0, 0], num_expected),
                       np.repeat([1], num_expected)])
        y = to_tensor(y)
        _, _, y_train, y_valid = cv_split_cls(5, stratified=True)(X, y)
        assert y_train.sum() == 0.8 * num_expected
        assert y_valid.sum() == 0.2 * num_expected

    def test_with_list_of_arrays(self, cv_split_cls, data):
        X, y = data
        X = [X, X]
        m = self.num_samples // 5
        n = self.num_samples - m

        X_train, X_valid, y_train, y_valid = cv_split_cls(5)(X, y)
        assert len(X_train[0]) == len(X_train[1]) == len(y_train) == n
        assert len(X_valid[0]) == len(X_valid[1]) == len(y_valid) == m

    def test_with_dict(self, cv_split_cls, data):
        X, y = data
        X = {'1': X, '2': X}
        X_train, X_valid, y_train, y_valid = cv_split_cls(5)(X, y)

        m = self.num_samples // 5
        n = self.num_samples - m

        assert len(X_train['1']) == len(X_train['2']) == len(y_train) == n
        assert len(X_valid['1']) == len(X_valid['2']) == len(y_valid) == m

    @pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
    def test_with_pandas(self, cv_split_cls, data):
        import pandas as pd

        X, y = data
        df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        y = pd.Series(y)

        X_train, X_valid, y_train, y_valid = cv_split_cls(5)(df, y)

        m = self.num_samples // 5

        assert len(X_train) + len(X_valid) == self.num_samples
        assert len(y_train) + len(y_valid) == self.num_samples
        assert len(X_valid) == len(y_valid) == m

    def test_y_str_val_stratified(self, cv_split_cls, data):
        X = data[0]
        y = np.array(['a', 'a', 'a', 'b'] * (len(X) // 4))
        if len(X) != len(y):
            raise ValueError

        _, _, y_train, y_valid = cv_split_cls(5, stratified=True)(X, y)

        assert np.isclose(np.mean(y_train == 'b'), 0.25)
        assert np.isclose(np.mean(y_valid == 'b'), 0.25)

    def test_y_list_of_arr_does_not_raise(self, cv_split_cls, data):
        X = data[0]
        y = [np.zeros(len(X)), np.ones(len(X))]

        cv_split_cls(5, stratified=False)(X, y)

    def test_y_list_of_arr_stratified(self, cv_split_cls, data):
        X = data[0]
        y = [np.zeros(len(X)), np.ones(len(X))]
        with pytest.raises(ValueError) as exc:
            cv_split_cls(5, stratified=True)(X, y)

        expected = "Stratified CV not possible with given y."
        assert exc.value.args[0] == expected

    def test_y_dict_does_not_raise(self, cv_split_cls, data):
        X = data[0]
        y = {'a': np.zeros(len(X)), 'b': np.ones(len(X))}

        cv_split_cls(5, stratified=False)(X, y)

    def test_y_dict_stratified_raises(self, cv_split_cls, data):
        X = data[0]
        y = {'a': np.zeros(len(X)), 'b': np.ones(len(X))}

        with pytest.raises(ValueError):
            # an sklearn error is raised
            cv_split_cls(5, stratified=True)(X, y)

    def test_y_none_cv_int(self, cv_split_cls, data):
        cv = 5
        X, y = data[0], None

        with pytest.raises(ValueError) as exc:
            cv_split_cls(cv, stratified=True)(X, y)

        expected = "Stratified CV not possible with given y."
        assert exc.value.args[0] == expected

    def test_y_none_cv_float_regular_X(self, cv_split_cls, data):
        cv = 0.2
        X, y = data[0], None

        with pytest.raises(ValueError) as exc:
            cv_split_cls(cv, stratified=True)(X, y)

        expected = "Stratified CV not possible with given y."
        assert exc.value.args[0] == expected

    def test_y_none_cv_float_irregular_X(self, cv_split_cls):
        cv = 0.2
        X, y = torch.zeros((100, 10)), None

        with pytest.raises(ValueError) as exc:
            cv_split_cls(cv, stratified=True)(X, y)

        expected = "Stratified CV not possible with given y."
        assert exc.value.args[0] == expected

    def test_shuffle_split_reproducible_with_random_state(self, cv_split_cls):
        X, y = np.random.random((100, 10)), np.random.randint(0, 10, size=100)
        cv = cv_split_cls(0.2, stratified=False)
        Xt0, Xv0, yt0, yv0 = cv(X, y)
        Xt1, Xv1, yt1, yv1 = cv(X, y)

        assert not np.allclose(Xt0, Xt1)
        assert not np.allclose(Xv0, Xv1)
        assert not np.allclose(yt0, yt1)
        assert not np.allclose(yv0, yv1)

        cv = cv_split_cls(0.2, stratified=False, random_state=0)
        Xt0, Xv0, yt0, yv0 = cv(X, y)
        Xt1, Xv1, yt1, yv1 = cv(X, y)

        assert np.allclose(Xt0, Xt1)
        assert np.allclose(Xv0, Xv1)
        assert np.allclose(yt0, yt1)
        assert np.allclose(yv0, yv1)
