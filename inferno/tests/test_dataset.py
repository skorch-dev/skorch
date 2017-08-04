import numpy as np
import pytest
from sklearn.datasets import make_classification
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

from .conftest import pandas_installed


class TestGetLen:
    @pytest.fixture
    def get_len(self):
        from inferno.dataset import get_len
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
    def net_cls_1d_clf(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(1,1)
            def forward(self, X):
                return F.softmax(self.dense(X.float()))
        return MyModule

    @pytest.fixture
    def net_cls_2d_clf(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(2,1)
            def forward(self, X):
                return F.softmax(self.dense(X.float()))
        return MyModule

    @pytest.fixture
    def net_cls_1d_reg(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(1,1)
            def forward(self, X):
                return F.tanh(self.dense(X.float()))
        return MyModule

    @pytest.fixture
    def net_cls_2d_reg(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(2,1)
            def forward(self, X):
                return F.tanh(self.dense(X.float()))
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

    @pytest.fixture(params=net_fixture_params)
    def net_1d(self, request, net_cls_1d_clf, net_cls_1d_reg):
        if request.param['classification']:
            from inferno import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
            net_cls = net_cls_1d_clf
        else:
            from inferno import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor
            net_cls = net_cls_1d_reg

        return wrap_cls(
            net_cls,
            max_epochs=2,
            batch_size=request.param['batch_size']
        )

    @pytest.fixture(params=net_fixture_params)
    def net_2d(self, request, net_cls_2d_clf, net_cls_2d_reg):
        if request.param['classification']:
            from inferno import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
            net_cls = net_cls_2d_clf
        else:
            from inferno import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor
            net_cls = net_cls_2d_clf

        return wrap_cls(
            net_cls,
            max_epochs=2,
            batch_size=request.param['batch_size']
        )

    @pytest.fixture(params=net_fixture_params)
    def net_1d_custom_loader(self, request, net_cls_1d_clf, net_cls_1d_reg,
                             loader_clf, loader_reg):
        if request.param['classification']:
            from inferno import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
            loader = loader_clf
            net_cls = net_cls_1d_clf
        else:
            from inferno import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor
            loader = loader_reg
            net_cls = net_cls_1d_reg

        return wrap_cls(
            net_cls,
            iterator_train=loader,
            iterator_test=loader,
            max_epochs=2,
            batch_size=request.param['batch_size']
        )

    @pytest.fixture(params=net_fixture_params)
    def net_2d_custom_loader(self, request, net_cls_2d_clf, net_cls_2d_reg,
                             loader_clf, loader_reg):
        if request.param['classification']:
            from inferno import NeuralNetClassifier
            wrap_cls = NeuralNetClassifier
            loader = loader_clf
            net_cls = net_cls_2d_clf
        else:
            from inferno import NeuralNetRegressor
            wrap_cls = NeuralNetRegressor
            loader = loader_reg
            net_cls = net_cls_2d_reg

        return wrap_cls(
            net_cls,
            iterator_train=loader,
            iterator_test=loader,
            max_epochs=2,
            batch_size=request.param['batch_size']
        )

    def test_net_1d_tensor_raises_error(self, net_1d):
        X = torch.Tensor([[1],[2],[3],[4],[5],[6],[7]])
        # We expect check_data to throw an exception
        # because we did not specify a custom data loader.
        with pytest.raises(ValueError):
            net_1d.fit(X, None)

    def test_net_2d_tensor_raises_error(self, net_2d):
        X = torch.Tensor([[1,2],[3,4],[5,6]])
        # We expect check_data to throw an exception
        # because we did not specify a custom data loader.
        with pytest.raises(ValueError):
            net_2d.fit(X, None)

    def test_net_1d_custom_loader(self, net_1d_custom_loader):
        X = torch.Tensor([[1],[2],[3],[4],[5],[6],[7]])
        # Should not raise an exception.
        net_1d_custom_loader.fit(X, None)

    def test_net_2d_custom_loader(self, net_2d_custom_loader):
        X = torch.Tensor([[1,2],[3,4],[5,6]])
        # Should not raise an exception.
        net_2d_custom_loader.fit(X, None)



class TestMultiIndexing:
    @pytest.fixture
    def multi_indexing(self):
        from inferno.dataset import multi_indexing
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
        result.keys() == expected.keys()
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
        result.keys() == expected.keys()
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

    @pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
    @pytest.fixture
    def pd(self):
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


class TestNetWithDict:
    @pytest.fixture(scope='module')
    def module_cls(self):
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

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
        from inferno.net import NeuralNetClassifier
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
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

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
        from inferno.net import NeuralNetClassifier
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
        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dense = nn.Linear(20, 2)

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
        from inferno.net import NeuralNetClassifier
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
