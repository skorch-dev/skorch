"""Test for utils.py"""

import pytest


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
