"""Test for utils.py"""
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import sparse
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence

from skorch.tests.conftest import pandas_installed


class TestToTensor:
    @pytest.fixture
    def to_tensor(self):
        from skorch.utils import to_tensor
        return to_tensor

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda device")
    def test_device_setting_cuda(self, to_tensor):
        x = np.ones((2, 3, 4))
        t = to_tensor(x, device='cpu')
        assert t.device.type == 'cpu'

        t = to_tensor(x, device='cuda')
        assert t.device.type.startswith('cuda')

        t = to_tensor(t, device='cuda')
        assert t.device.type.startswith('cuda')

        t = to_tensor(t, device='cpu')
        assert t.device.type == 'cpu'

    def tensors_equal(self, x, y):
        """"Test that tensors in diverse containers are equal."""
        if isinstance(x, PackedSequence):
            return self.tensors_equal(x[0], y[0]) and self.tensors_equal(x[1], y[1])

        if isinstance(x, dict):
            return (
                (x.keys() == y.keys()) and
                all(self.tensors_equal(x[k], y[k]) for k in x)
            )

        if isinstance(x, (list, tuple)):
            return all(self.tensors_equal(xi, yi) for xi, yi in zip(x, y))

        if x.is_sparse is not y.is_sparse:
            return False

        if x.is_sparse:
            x, y = x.to_dense(), y.to_dense()

        return (x == y).all()

    # pylint: disable=no-method-argument
    def parameters():
        """Yields data, expected value, and device for tensor conversion
        test.

        Stops earlier when no cuda device is available.

        """
        device = 'cpu'
        x = torch.zeros((5, 3)).float()
        y = torch.as_tensor([2, 2, 1])
        z = np.arange(15).reshape(5, 3)
        for X, expected in [
                (x, x),
                (y, y),
                ([x, y], [x, y]),
                ((x, y), (x, y)),
                (z, torch.as_tensor(z)),
                (
                    {'a': x, 'b': y, 'c': z},
                    {'a': x, 'b': y, 'c': torch.as_tensor(z)}
                ),
                (torch.as_tensor(55), torch.as_tensor(55)),
                (pack_padded_sequence(x, y), pack_padded_sequence(x, y)),
        ]:
            yield X, expected, device

        if not torch.cuda.is_available():
            return

        device = 'cuda'
        x = x.to('cuda')
        y = y.to('cuda')
        for X, expected in [
                (x, x),
                (y, y),
                ([x, y], [x, y]),
                ((x, y), (x, y)),
                (z, torch.as_tensor(z).to('cuda')),
                (
                    {'a': x, 'b': y, 'c': z},
                    {'a': x, 'b': y, 'c': torch.as_tensor(z).to('cuda')}
                ),
                (torch.as_tensor(55), torch.as_tensor(55).to('cuda')),
                (
                    pack_padded_sequence(x, y),
                    pack_padded_sequence(x, y).to('cuda')
                ),
        ]:
            yield X, expected, device

    @pytest.mark.parametrize('X, expected, device', parameters())
    def test_tensor_conversion_cuda(self, to_tensor, X, expected, device):
        result = to_tensor(X, device)
        assert self.tensors_equal(result, expected)
        assert self.tensors_equal(expected, result)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_sparse_tensor(self, to_tensor, device):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip()

        inp = sparse.csr_matrix(np.zeros((5, 3)).astype(np.float32))
        expected = torch.sparse_coo_tensor(size=(5, 3)).to(device)

        result = to_tensor(inp, device=device, accept_sparse=True)
        assert self.tensors_equal(result, expected)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_sparse_tensor_not_accepted_raises(self, to_tensor, device):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip()

        inp = sparse.csr_matrix(np.zeros((5, 3)).astype(np.float32))
        with pytest.raises(TypeError) as exc:
            to_tensor(inp, device=device)

        msg = ("Sparse matrices are not supported. Set "
               "accept_sparse=True to allow sparse matrices.")
        assert exc.value.args[0] == msg


class TestDuplicateItems:
    @pytest.fixture
    def duplicate_items(self):
        from skorch.utils import duplicate_items
        return duplicate_items

    @pytest.mark.parametrize('collections', [
        ([],),
        ([], []),
        ([], [], []),
        ([1, 2]),
        ([1, 2], [3]),
        ([1, 2], [3, '1']),
        ([1], [2], [3], [4]),
        ({'1': 1}, [2]),
        ({'1': 1}, {'2': 1}, ('3', '4')),
    ])
    def test_no_duplicates(self, duplicate_items, collections):
        assert duplicate_items(*collections) == set()

    @pytest.mark.parametrize('collections, expected', [
        ([1, 1], {1}),
        (['1', '1'], {'1'}),
        ([[1], [1]], {1}),
        ([[1, 2, 1], [1]], {1}),
        ([[1, 1], [2, 2]], {1, 2}),
        ([[1], {1: '2', 2: '2'}], {1}),
        ([[1, 2], [3, 4], [2], [3]], {2, 3}),
        ([{'1': 1}, {'1': 1}, ('3', '4')], {'1'}),
    ])
    def test_duplicates(self, duplicate_items, collections, expected):
        assert duplicate_items(*collections) == expected


class TestParamsFor:
    @pytest.fixture
    def params_for(self):
        from skorch.utils import params_for
        return params_for

    @pytest.mark.parametrize('prefix, kwargs, expected', [
        ('p1', {'p1__a': 1, 'p1__b': 2}, {'a': 1, 'b': 2}),
        ('p2', {'p1__a': 1, 'p1__b': 2}, {}),
        ('p1', {'p1__a': 1, 'p1__b': 2, 'p2__a': 3}, {'a': 1, 'b': 2}),
        ('p2', {'p1__a': 1, 'p1__b': 2, 'p2__a': 3}, {'a': 3}),
    ])
    def test_params_for(self, params_for, prefix, kwargs, expected):
        assert params_for(prefix, kwargs) == expected


class TestDataFromDataset:
    @pytest.fixture
    def data_from_dataset(self):
        from skorch.utils import data_from_dataset
        return data_from_dataset

    @pytest.fixture
    def data(self):
        X = np.arange(8).reshape(4, 2)
        y = np.array([1, 3, 0, 2])
        return X, y

    @pytest.fixture
    def skorch_ds(self, data):
        from skorch.dataset import Dataset
        return Dataset(*data)

    @pytest.fixture
    def subset(self, skorch_ds):
        from torch.utils.data.dataset import Subset
        return Subset(skorch_ds, [1, 3])

    @pytest.fixture
    def subset_subset(self, subset):
        from torch.utils.data.dataset import Subset
        return Subset(subset, [0])

    # pylint: disable=missing-docstring
    @pytest.fixture
    def other_ds(self, data):
        class MyDataset:
            """Non-compliant dataset"""
            def __init__(self, data):
                self.data = data

            def __getitem__(self, idx):
                return self.data[0][idx], self.data[1][idx]

            def __len__(self):
                return len(self.data[0])
        return MyDataset(data)

    def test_with_skorch_ds(self, data_from_dataset, data, skorch_ds):
        X, y = data_from_dataset(skorch_ds)
        assert (X == data[0]).all()
        assert (y == data[1]).all()

    def test_with_subset(self, data_from_dataset, data, subset):
        X, y = data_from_dataset(subset)
        assert (X == data[0][[1, 3]]).all()
        assert (y == data[1][[1, 3]]).all()

    def test_with_subset_subset(self, data_from_dataset, data, subset_subset):
        X, y = data_from_dataset(subset_subset)
        assert (X == data[0][1]).all()
        assert (y == data[1][1]).all()

    def test_with_other_ds(self, data_from_dataset, other_ds):
        with pytest.raises(AttributeError):
            data_from_dataset(other_ds)

    def test_with_dict_data(self, data_from_dataset, data, subset):
        subset.dataset.X = {'X': subset.dataset.X}
        X, y = data_from_dataset(subset)
        assert (X['X'] == data[0][[1, 3]]).all()
        assert (y == data[1][[1, 3]]).all()

    def test_subset_with_y_none(self, data_from_dataset, data, subset):
        subset.dataset.y = None
        X, y = data_from_dataset(subset)
        assert (X == data[0][[1, 3]]).all()
        assert y is None


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
            np.int64(2),
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
    def test_sparse_csr_matrix(self, multi_indexing, data, i, expected):
        data = sparse.csr_matrix(data)
        result = multi_indexing(data, i).toarray()
        assert np.allclose(result, expected)


class TestIsSkorchDataset:

    @pytest.fixture
    def is_skorch_dataset(self):
        from skorch.utils import is_skorch_dataset
        return is_skorch_dataset

    # pylint: disable=no-method-argument
    def type_truth_table():
        """Return a table of (type, bool) tuples that describe what
        is_skorch_dataset should return when called with that type.
        """
        from skorch.dataset import Dataset
        from torch.utils.data.dataset import Subset

        numpy_data = np.array([1, 2, 3])
        tensor_data = torch.from_numpy(numpy_data)
        torch_dataset = torch.utils.data.TensorDataset(
            tensor_data, tensor_data)
        torch_subset = Subset(torch_dataset, [1, 2])
        skorch_dataset = Dataset(numpy_data)
        skorch_subset = Subset(skorch_dataset, [1, 2])

        return [
            (numpy_data, False),
            (torch_dataset, False),
            (torch_subset, False),
            (skorch_dataset, True),
            (skorch_subset, True),
        ]

    @pytest.mark.parametrize(
        'input_data,expected',
        type_truth_table())
    def test_data_types(self, is_skorch_dataset, input_data, expected):
        assert is_skorch_dataset(input_data) == expected


class TestTeeGenerator:

    @pytest.fixture
    def lazy_generator_cls(self):
        from skorch.utils import TeeGenerator
        return TeeGenerator

    def test_returns_copies_of_generator(self, lazy_generator_cls):
        expected_list = [1, 2, 3]

        def list_gen():
            yield from expected_list
        lazy_gen = lazy_generator_cls(list_gen())

        first_return = list(lazy_gen)
        second_return = [item for item in lazy_gen]

        assert first_return == expected_list
        assert second_return == expected_list
