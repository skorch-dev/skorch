"""Test for utils.py"""

from copy import deepcopy

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
                    pack_padded_sequence(x, y.to('cpu')),
                    pack_padded_sequence(x, y.to('cpu')).to('cuda')
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


class TestToNumpy:
    @pytest.fixture
    def to_numpy(self):
        from skorch.utils import to_numpy
        return to_numpy

    @pytest.fixture
    def x_tensor(self):
        return torch.zeros(3, 4)

    @pytest.fixture
    def x_tuple(self):
        return torch.ones(3), torch.zeros(3, 4)

    @pytest.fixture
    def x_list(self):
        return [torch.ones(3), torch.zeros(3, 4)]

    @pytest.fixture
    def x_dict(self):
        return {'a': torch.ones(3), 'b': (torch.zeros(2), torch.zeros(3))}

    def compare_array_to_tensor(self, x_numpy, x_tensor):
        assert isinstance(x_tensor, torch.Tensor)
        assert isinstance(x_numpy, np.ndarray)
        assert x_numpy.shape == x_tensor.shape
        for a, b in zip(x_numpy.flatten(), x_tensor.flatten()):
            assert np.isclose(a, b.item())

    def test_tensor(self, to_numpy, x_tensor):
        x_numpy = to_numpy(x_tensor)
        self.compare_array_to_tensor(x_numpy, x_tensor)

    def test_list(self, to_numpy, x_list):
        x_numpy = to_numpy(x_list)
        for entry_numpy, entry_torch in zip(x_numpy, x_list):
            self.compare_array_to_tensor(entry_numpy, entry_torch)

    def test_tuple(self, to_numpy, x_tuple):
        x_numpy = to_numpy(x_tuple)
        for entry_numpy, entry_torch in zip(x_numpy, x_tuple):
            self.compare_array_to_tensor(entry_numpy, entry_torch)

    def test_dict(self, to_numpy, x_dict):
        x_numpy = to_numpy(x_dict)
        self.compare_array_to_tensor(x_numpy['a'], x_dict['a'])
        self.compare_array_to_tensor(x_numpy['b'][0], x_dict['b'][0])
        self.compare_array_to_tensor(x_numpy['b'][1], x_dict['b'][1])

    @pytest.mark.parametrize('x_invalid', [
        1,
        [1, 2, 3],
        (1, 2, 3),
        {'a': 1},
    ])
    def test_invalid_inputs(self, to_numpy, x_invalid):
        # Inputs that are invalid for the scope of to_numpy.
        with pytest.raises(TypeError) as e:
            to_numpy(x_invalid)
        expected = "Cannot convert this data type to a numpy array."
        assert e.value.args[0] == expected

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason='Skipped because mps is not available as a torch backend'
    )
    def test_mps_support(self, to_numpy, x_tensor):
        device = torch.device('mps')
        x_tensor.to(device)
        x_numpy = to_numpy(x_tensor)
        self.compare_array_to_tensor(x_numpy, x_tensor)


class TestToDevice:
    @pytest.fixture
    def to_device(self):
        from skorch.utils import to_device
        return to_device

    @pytest.fixture
    def x(self):
        return torch.zeros(3)

    @pytest.fixture
    def x_tup(self):
        return torch.zeros(3), torch.ones((4, 5))

    @pytest.fixture
    def x_dict(self):
        return {
            'x': torch.zeros(3),
            'y': torch.ones((4, 5))
        }

    @pytest.fixture
    def x_pad_seq(self):
        value = torch.zeros((5, 3)).float()
        length = torch.as_tensor([2, 2, 1])
        return pack_padded_sequence(value, length)

    @pytest.fixture
    def x_list(self):
        return [torch.zeros(3), torch.ones(2, 4)]

    def check_device_type(self, tensor, device_input, prev_device):
        """assert expected device type conditioned on the input argument for
        `to_device`"""
        if None is device_input:
            assert tensor.device.type == prev_device

        else:
            assert tensor.device.type == device_input

    @pytest.mark.parametrize('device_from, device_to', [
        ('cpu', 'cpu'),
        ('cpu', 'cuda'),
        ('cuda', 'cpu'),
        ('cuda', 'cuda'),
        (None, None),
    ])
    def test_check_device_torch_tensor(self, to_device, x, device_from, device_to):
        if 'cuda' in (device_from, device_to) and not torch.cuda.is_available():
            pytest.skip()

        prev_device = None
        if None in (device_from, device_to):
            prev_device = x.device.type

        x = to_device(x, device=device_from)
        self.check_device_type(x, device_from, prev_device)

        x = to_device(x, device=device_to)
        self.check_device_type(x, device_to, prev_device)

    @pytest.mark.parametrize('device_from, device_to', [
        ('cpu', 'cpu'),
        ('cpu', 'cuda'),
        ('cuda', 'cpu'),
        ('cuda', 'cuda'),
        (None, None),
    ])
    def test_check_device_tuple_torch_tensor(
            self, to_device, x_tup, device_from, device_to):
        if 'cuda' in (device_from, device_to) and not torch.cuda.is_available():
            pytest.skip()

        prev_devices = [None for _ in range(len(x_tup))]
        if None in (device_from, device_to):
            prev_devices = [x.device.type for x in x_tup]

        x_tup = to_device(x_tup, device=device_from)
        for xi, prev_d in zip(x_tup, prev_devices):
            self.check_device_type(xi, device_from, prev_d)

        x_tup = to_device(x_tup, device=device_to)
        for xi, prev_d in zip(x_tup, prev_devices):
            self.check_device_type(xi, device_to, prev_d)

    @pytest.mark.parametrize('device_from, device_to', [
        ('cpu', 'cpu'),
        ('cpu', 'cuda'),
        ('cuda', 'cpu'),
        ('cuda', 'cuda'),
        (None, None),
    ])
    def test_check_device_dict_torch_tensor(
            self, to_device, x_dict, device_from, device_to):
        if 'cuda' in (device_from, device_to) and not torch.cuda.is_available():
            pytest.skip()

        original_x_dict = deepcopy(x_dict)

        prev_devices = [None for _ in range(len(list(x_dict.keys())))]
        if None in (device_from, device_to):
            prev_devices = [x.device.type for x in x_dict.values()]

        new_x_dict = to_device(x_dict, device=device_from)
        for xi, prev_d in zip(new_x_dict.values(), prev_devices):
            self.check_device_type(xi, device_from, prev_d)

        new_x_dict = to_device(new_x_dict, device=device_to)
        for xi, prev_d in zip(new_x_dict.values(), prev_devices):
            self.check_device_type(xi, device_to, prev_d)

        assert x_dict.keys() == original_x_dict.keys()
        for k in x_dict:
            assert np.allclose(x_dict[k], original_x_dict[k])

    @pytest.mark.parametrize('device_from, device_to', [
        ('cpu', 'cpu'),
        ('cpu', 'cuda'),
        ('cuda', 'cpu'),
        ('cuda', 'cuda'),
        (None, None),
    ])
    def test_check_device_packed_padded_sequence(
            self, to_device, x_pad_seq, device_from, device_to):
        if 'cuda' in (device_from, device_to) and not torch.cuda.is_available():
            pytest.skip()

        prev_device = None
        if None in (device_from, device_to):
            prev_device = x_pad_seq.data.device.type

        x_pad_seq = to_device(x_pad_seq, device=device_from)
        self.check_device_type(x_pad_seq.data, device_from, prev_device)

        x_pad_seq = to_device(x_pad_seq, device=device_to)
        self.check_device_type(x_pad_seq.data, device_to, prev_device)

    @pytest.mark.parametrize('device_from, device_to', [
        ('cpu', 'cpu'),
        ('cpu', 'cuda'),
        ('cuda', 'cpu'),
        ('cuda', 'cuda'),
        (None, None),
    ])
    def test_nested_data(self, to_device, x_list, device_from, device_to):
        # Sometimes data is nested because it would need to be padded so it's
        # easier to return a list of tensors with different shapes.
        # to_device should honor this.
        if 'cuda' in (device_from, device_to) and not torch.cuda.is_available():
            pytest.skip()

        prev_devices = [None for _ in range(len(x_list))]
        if None in (device_from, device_to):
            prev_devices = [x.device.type for x in x_list]

        x_list = to_device(x_list, device=device_from)
        assert isinstance(x_list, list)

        for xi, prev_d in zip(x_list, prev_devices):
            self.check_device_type(xi, device_from, prev_d)

        x_list = to_device(x_list, device=device_to)
        assert isinstance(x_list, list)

        for xi, prev_d in zip(x_list, prev_devices):
            self.check_device_type(xi, device_to, prev_d)


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
    def tensors(self, data):
        X, y = data
        return torch.from_numpy(X), torch.from_numpy(y)

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

    def test_with_tensordataset_2_vals(self, data_from_dataset, tensors):
        dataset = torch.utils.data.dataset.TensorDataset(*tensors)
        X, y = data_from_dataset(dataset)
        assert (X == tensors[0]).all()
        assert (y == tensors[1]).all()

    def test_with_tensordataset_1_val_raises(self, data_from_dataset, tensors):
        dataset = torch.utils.data.dataset.TensorDataset(tensors[0])
        msg = "Could not access X and y from dataset."
        with pytest.raises(AttributeError, match=msg):
            data_from_dataset(dataset)

    def test_with_tensordataset_3_vals_raises(self, data_from_dataset, tensors):
        dataset = torch.utils.data.dataset.TensorDataset(*tensors, tensors[0])
        msg = "Could not access X and y from dataset."
        with pytest.raises(AttributeError, match=msg):
            data_from_dataset(dataset)


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
        # sklearn < 0.22 raises IndexError with msg0
        # sklearn >= 0.22 raises ValueError with msg1
        X = np.zeros(10)
        i = np.arange(3, 0.5)

        with pytest.raises((IndexError, ValueError)) as exc:
            multi_indexing(X, i)

        msg0 = "arrays used as indices must be of integer (or boolean) type"
        msg1 = ("No valid specification of the columns. Only a scalar, list or "
                "slice of all integers or all strings, or boolean mask is allowed")

        result = exc.value.args[0]
        assert result in (msg0, msg1)

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


class TestInferPredictNonlinearity:
    @pytest.fixture
    def infer_predict_nonlinearity(self):
        from skorch.utils import _infer_predict_nonlinearity
        return _infer_predict_nonlinearity

    @pytest.fixture
    def net_clf_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture
    def net_bin_clf_cls(self):
        from skorch import NeuralNetBinaryClassifier
        return NeuralNetBinaryClassifier

    @pytest.fixture
    def net_regr_cls(self):
        from skorch import NeuralNetRegressor
        return NeuralNetRegressor

    def test_infer_neural_net_classifier_default(
            self, infer_predict_nonlinearity, net_clf_cls, module_cls):
        # default NeuralNetClassifier: no output nonlinearity
        net = net_clf_cls(module_cls).initialize()
        fn = infer_predict_nonlinearity(net)

        X = np.random.random((20, 5))
        out = fn(X)
        assert out is X

    def test_infer_neural_net_classifier_crossentropy_loss(
            self, infer_predict_nonlinearity, net_clf_cls, module_cls):
        # CrossEntropyLoss criteron: nonlinearity should return valid probabilities
        net = net_clf_cls(module_cls, criterion=torch.nn.CrossEntropyLoss).initialize()
        fn = infer_predict_nonlinearity(net)

        X = torch.rand((20, 5))
        out = fn(X).numpy()
        assert np.allclose(out.sum(axis=1), 1.0)
        # pylint: disable=misplaced-comparison-constant
        assert ((0 <= out) & (out <= 1.0)).all()

    def test_infer_neural_binary_net_classifier_default(
            self, infer_predict_nonlinearity, net_bin_clf_cls, module_cls):
        # BCEWithLogitsLoss should return valid probabilities
        net = net_bin_clf_cls(module_cls).initialize()
        fn = infer_predict_nonlinearity(net)

        X = torch.rand(20)  # binary classifier returns 1-dim output
        X = 10 * X - 5.0  # random values from -5 to 5
        out = fn(X).numpy()
        assert out.shape == (20, 2)  # output should be 2-dim
        assert np.allclose(out.sum(axis=1), 1.0)
        # pylint: disable=misplaced-comparison-constant
        assert ((0 <= out) & (out <= 1.0)).all()

    def test_infer_neural_net_regressor_default(
            self, infer_predict_nonlinearity, net_regr_cls, module_cls):
        # default NeuralNetRegressor: no output nonlinearity
        net = net_regr_cls(module_cls).initialize()
        fn = infer_predict_nonlinearity(net)

        X = np.random.random((20, 5))
        out = fn(X)
        assert out is X
