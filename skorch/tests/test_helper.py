"""Test for helper.py"""
import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
import torch
from torch import nn

from skorch.tests.conftest import pandas_installed


def assert_dicts_equal(d0, d1):
    assert d0.keys() == d1.keys()
    for key in d0.keys():
        val0, val1 = d0[key], d1[key]

        np.testing.assert_allclose(val0, val1)
        assert val0.dtype == val1.dtype


class TestSliceDict:
    @pytest.fixture
    def data(self):
        X, y = make_classification(100, 20, n_informative=10, random_state=0)
        return X.astype(np.float32), y

    @pytest.fixture(scope='session')
    def sldict_cls(self):
        from skorch.helper import SliceDict
        return SliceDict

    @pytest.fixture
    def sldict(self, sldict_cls):
        return sldict_cls(
            f0=np.arange(4),
            f1=np.arange(12).reshape(4, 3),
        )

    def test_init_inconsistent_shapes(self, sldict_cls):
        with pytest.raises(ValueError) as exc:
            sldict_cls(f0=np.ones((10, 5)), f1=np.ones((11, 5)))
        assert str(exc.value) == (
            "Initialized with items of different lengths: 10, 11")

    @pytest.mark.parametrize('item', [
        np.ones(4),
        np.ones((4, 1)),
        np.ones((4, 4)),
        np.ones((4, 10, 7)),
        np.ones((4, 1, 28, 28)),
    ])
    def test_set_item_correct_shape(self, sldict, item):
        # does not raise
        sldict['f2'] = item

    @pytest.mark.parametrize('item', [
        np.ones(3),
        np.ones((1, 100)),
        np.ones((5, 1000)),
        np.ones((1, 100, 10)),
        np.ones((28, 28, 1, 100)),
    ])
    def test_set_item_incorrect_shape_raises(self, sldict, item):
        with pytest.raises(ValueError) as exc:
            sldict['f2'] = item
        assert str(exc.value) == (
            "Cannot set array with shape[0] != 4")

    @pytest.mark.parametrize('key', [1, 1.2, (1, 2), [3]])
    def test_set_item_incorrect_key_type(self, sldict, key):
        with pytest.raises(TypeError) as exc:
            sldict[key] = np.ones((100, 5))
        assert str(exc.value).startswith("Key must be str, not <")

    @pytest.mark.parametrize('item', [
        np.ones(3),
        np.ones((1, 100)),
        np.ones((5, 1000)),
        np.ones((1, 100, 10)),
        np.ones((28, 28, 1, 100)),
    ])
    def test_update_incorrect_shape_raises(self, sldict, item):
        with pytest.raises(ValueError) as exc:
            sldict.update({'f2': item})
        assert str(exc.value) == (
            "Cannot set array with shape[0] != 4")

    @pytest.mark.parametrize('item', [123, 'hi', [1, 2, 3]])
    def test_set_first_item_no_shape_raises(self, sldict_cls, item):
        with pytest.raises(AttributeError):
            sldict_cls(f0=item)

    @pytest.mark.parametrize('kwargs, expected', [
        ({}, 0),
        (dict(a=np.zeros(12)), 12),
        (dict(a=np.zeros(12), b=np.ones((12, 5))), 12),
        (dict(a=np.ones((10, 1, 1)), b=np.ones((10, 10)), c=np.ones(10)), 10),
    ])
    def test_len_and_shape(self, sldict_cls, kwargs, expected):
        sldict = sldict_cls(**kwargs)
        assert len(sldict) == expected
        assert sldict.shape == (expected,)

    def test_get_item_str_key(self, sldict_cls):
        sldict = sldict_cls(a=np.ones(5), b=np.zeros(5))
        assert (sldict['a'] == np.ones(5)).all()
        assert (sldict['b'] == np.zeros(5)).all()

    @pytest.mark.parametrize('sl, expected', [
        (slice(0, 1), {'f0': np.array([0]), 'f1': np.array([[0, 1, 2]])}),
        (slice(1, 2), {'f0': np.array([1]), 'f1': np.array([[3, 4, 5]])}),
        (slice(0, 2), {'f0': np.array([0, 1]),
                       'f1': np.array([[0, 1, 2], [3, 4, 5]])}),
        (slice(0, None), dict(f0=np.arange(4),
                              f1=np.arange(12).reshape(4, 3))),
        (slice(-1, None), {'f0': np.array([3]),
                           'f1': np.array([[9, 10, 11]])}),
        (slice(None, None, -1), dict(f0=np.arange(4)[::-1],
                                     f1=np.arange(12).reshape(4, 3)[::-1])),
    ])
    def test_get_item_slice(self, sldict_cls, sldict, sl, expected):
        sliced = sldict[sl]
        assert_dicts_equal(sliced, sldict_cls(**expected))

    def test_slice_list(self, sldict, sldict_cls):
        result = sldict[[0, 2]]
        expected = sldict_cls(
            f0=np.array([0, 2]),
            f1=np.array([[0, 1, 2], [6, 7, 8]]))
        assert_dicts_equal(result, expected)

    def test_slice_mask(self, sldict, sldict_cls):
        result = sldict[np.array([1, 0, 1, 0]).astype(bool)]
        expected = sldict_cls(
            f0=np.array([0, 2]),
            f1=np.array([[0, 1, 2], [6, 7, 8]]))
        assert_dicts_equal(result, expected)

    def test_slice_int(self, sldict):
        with pytest.raises(ValueError) as exc:
            # pylint: disable=pointless-statement
            sldict[0]
        assert str(exc.value) == 'SliceDict cannot be indexed by integers.'

    def test_len_sliced(self, sldict):
        assert len(sldict) == 4
        for i in range(1, 4):
            assert len(sldict[:i]) == i

    def test_str_repr(self, sldict, sldict_cls):
        loc = locals().copy()
        loc.update({'array': np.array, 'SliceDict': sldict_cls})
        # pylint: disable=eval-used
        result = eval(str(sldict), globals(), loc)
        assert_dicts_equal(result, sldict)

    def test_iter_over_keys(self, sldict):
        found_keys = {key for key in sldict}
        expected_keys = {'f0', 'f1'}
        assert found_keys == expected_keys

    def test_grid_search_with_dict_works(
            self, sldict_cls, data, classifier_module):
        from sklearn.model_selection import GridSearchCV
        from skorch import NeuralNetClassifier

        net = NeuralNetClassifier(classifier_module)
        X, y = data
        X = sldict_cls(X=X)
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
        }
        gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy')
        gs.fit(X, y)
        print(gs.best_score_, gs.best_params_)

    def test_copy(self, sldict, sldict_cls):
        copied = sldict.copy()
        assert copied.shape == sldict.shape
        assert isinstance(copied, sldict_cls)

    def test_fromkeys_raises(self, sldict_cls):
        with pytest.raises(TypeError) as exc:
            sldict_cls.fromkeys(['f0', 'f1'])

        msg = "SliceDict does not support fromkeys."
        assert exc.value.args[0] == msg

    def test_update(self, sldict, sldict_cls):
        copied = sldict.copy()
        copied['f0'] = -copied['f0']

        sldict.update(copied)
        assert (sldict['f0'] == copied['f0']).all()
        assert isinstance(sldict, sldict_cls)

    def test_equals_arrays(self, sldict):
        copied = sldict.copy()
        copied['f0'] = -copied['f0']

        # pylint: disable=comparison-with-itself
        assert copied == copied
        assert not copied == sldict
        assert copied != sldict

    def test_equals_arrays_deep(self, sldict):
        copied = sldict.copy()
        copied['f0'] = np.array(copied['f0'].copy())

        # pylint: disable=comparison-with-itself
        assert copied == copied
        assert copied == sldict

    def test_equals_tensors(self, sldict_cls):
        sldict = sldict_cls(
            f0=torch.arange(4),
            f1=torch.arange(12).reshape(4, 3),
        )
        copied = sldict.copy()
        copied['f0'] = -copied['f0']

        # pylint: disable=comparison-with-itself
        assert copied == copied
        assert not copied == sldict
        assert copied != sldict

    def test_equals_tensors_deep(self, sldict_cls):
        sldict = sldict_cls(
            f0=torch.arange(4),
            f1=torch.arange(12).reshape(4, 3),
        )
        copied = sldict.copy()
        copied['f0'] = copied['f0'].clone()

        # pylint: disable=comparison-with-itself
        assert copied == copied
        assert copied == sldict

    def test_equals_arrays_tensors_mixed(self, sldict_cls):
        sldict0 = sldict_cls(
            f0=np.arange(4),
            f1=torch.arange(12).reshape(4, 3),
        )
        sldict1 = sldict_cls(
            f0=np.arange(4),
            f1=torch.arange(12).reshape(4, 3),
        )

        assert sldict0 == sldict1

        sldict1['f0'] = torch.arange(4)
        assert sldict0 != sldict1

    def test_equals_different_keys(self, sldict_cls):
        sldict0 = sldict_cls(
            a=np.arange(3),
        )
        sldict1 = sldict_cls(
            a=np.arange(3),
            b=np.arange(3, 6),
        )
        assert sldict0 != sldict1

    def test_subclass_getitem_returns_instance_of_itself(self, sldict_cls):
        class MySliceDict(sldict_cls):
            pass

        sldict = MySliceDict(a=np.zeros(3))
        sliced = sldict[:2]
        assert isinstance(sliced, MySliceDict)


class TestSliceDataset:
    @pytest.fixture(scope='class', params=['numpy', 'torch'])
    def data(self, request):
        X, y = make_classification(100, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)
        if request.param == 'numpy':
            return X, y
        return torch.from_numpy(X), torch.from_numpy(y)

    @pytest.fixture
    def X(self, data):
        return data[0]

    @pytest.fixture
    def y(self, data):
        return data[1]

    @pytest.fixture
    def custom_ds(self, data):
        """Return a custom dataset instance"""
        from skorch.dataset import Dataset
        class MyDataset(Dataset):
            """Simple pytorch dataset that returns 2 values"""
            def __len__(self):
                return len(self.X)

            def __getitem__(self, i):
                Xi = self.X[i]
                yi = self.y[i]
                return self.transform(Xi, yi)

        return MyDataset(*data)

    @pytest.fixture(scope='session')
    def slds_cls(self):
        from skorch.helper import SliceDataset
        return SliceDataset

    @pytest.fixture
    def slds(self, slds_cls, custom_ds):
        return slds_cls(custom_ds)

    @pytest.fixture
    def slds_y(self, slds_cls, custom_ds):
        return slds_cls(custom_ds, idx=1)

    def test_len_and_shape(self, slds, y):
        assert len(slds) == len(y)
        assert slds.shape == (len(y),)

    @pytest.mark.parametrize('sl', [
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(-1, None),
        slice(None, None, -1),
        [0],
        [55],
        [-3],
        [0, 10, 3, -8, 3],
        np.ones(100, dtype=bool),
        # boolean mask array of length 100
        np.array([0, 0, 1, 0] * 25, dtype=bool),
    ])
    def test_len_and_shape_sliced(self, slds, y, sl):
        # torch tensors don't support negative steps, skip test
        if (
                isinstance(sl, slice)
                and (sl == slice(None, None, -1))
                and isinstance(y, torch.Tensor)
        ):
            return

        assert len(slds[sl]) == len(y[sl])
        assert slds[sl].shape == (len(y[sl]),)

    @pytest.mark.parametrize('n', [0, 1])
    def test_slice_non_int_is_slicedataset(self, slds_cls, custom_ds, n):
        slds = slds_cls(custom_ds, idx=n)
        sl = np.arange(7, 55, 3)
        sliced = slds[sl]
        assert isinstance(sliced, slds_cls)
        assert np.allclose(sliced.indices_, sl)

    @pytest.mark.parametrize('n', [0, 1])
    @pytest.mark.parametrize('sl', [0, 55, -3])
    def test_slice(self, slds_cls, custom_ds, X, y, sl, n):
        data = y if n else X
        slds = slds_cls(custom_ds, idx=n)
        sliced = slds[sl]
        x = data[sl]
        assert np.allclose(sliced, x)

    @pytest.mark.parametrize('n', [0, 1])
    @pytest.mark.parametrize('sl0, sl1', [
        ([55], 0),
        (slice(0, 1), 0),
        (slice(-1, None), 0),
        ([55], -1),
        ([0, 10, 3, -8, 3], 1),
        (np.ones(100, dtype=bool), 5),
        # boolean mask array of length 100
        (np.array([0, 0, 1, 0] * 25, dtype=bool), 6),
    ])
    def test_slice_twice(self, slds_cls, custom_ds, X, y, sl0, sl1, n):
        data = X if n == 0 else y
        slds = slds_cls(custom_ds, idx=n)
        sliced = slds[sl0][sl1]
        x = data[sl0][sl1]
        assert np.allclose(sliced, x)

    @pytest.mark.parametrize('n', [0, 1])
    @pytest.mark.parametrize('sl0, sl1, sl2', [
        (slice(0, 50), slice(10, 20), 5),
        ([0, 10, 3, -8, 3], [1, 2, 3], 2),
        (np.ones(100, dtype=bool), np.arange(10, 40), 29),
    ])
    def test_slice_three_times(self, slds_cls, custom_ds, X, y, sl0, sl1, sl2, n):
        data = y if n else X
        slds = slds_cls(custom_ds, idx=n)
        sliced = slds[sl0][sl1][sl2]
        x = data[sl0][sl1][sl2]
        assert np.allclose(sliced, x)

    def test_explicitly_pass_indices_at_init(self, slds_cls, custom_ds, X):
        from skorch.utils import to_numpy
        # test passing indices directy to __init__
        slds = slds_cls(custom_ds, indices=np.arange(10))
        sliced0 = slds[5:]
        sliced1 = sliced0[2]

        # comparison method depends on array type
        if isinstance(sliced1, torch.Tensor):
            sliced0 = to_numpy(sliced0)
            sliced1 = to_numpy(sliced1)

        assert np.allclose(sliced0, X[5:10])
        assert np.allclose(sliced1, X[7])

    def test_access_element_out_of_bounds(self, slds_cls, custom_ds):
        slds = slds_cls(custom_ds, idx=2)
        with pytest.raises(IndexError) as exc:
            # pylint: disable=pointless-statement
            slds[0]

        msg = ("SliceDataset is trying to access element 2 but there are only "
               "2 elements.")
        assert exc.value.args[0] == msg

    def test_fit_with_slds_works(self, slds, y, classifier_module):
        from skorch import NeuralNetClassifier
        net = NeuralNetClassifier(classifier_module)
        net.fit(slds, y)  # does not raise

    def test_fit_with_slds_without_valid_works(self, slds, y, classifier_module):
        from skorch import NeuralNetClassifier
        net = NeuralNetClassifier(classifier_module, train_split=False)
        net.fit(slds, y)  # does not raise

    def test_grid_search_with_slds_works(
            self, slds, y, classifier_module):
        from sklearn.model_selection import GridSearchCV
        from skorch import NeuralNetClassifier
        from skorch.utils import to_numpy

        net = NeuralNetClassifier(
            classifier_module,
            train_split=False,
            verbose=0,
        )
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
        }
        gs = GridSearchCV(
            net, params, refit=False, cv=3, scoring='accuracy', error_score='raise'
        )
        # TODO: after sklearn > 1.6 is released, the to_numpy call should no longer be
        # required and be removed, see:
        # https://github.com/skorch-dev/skorch/pull/1078#discussion_r1887197261
        gs.fit(slds, to_numpy(y))  # does not raise

    def test_grid_search_with_slds_and_internal_split_works(
            self, slds, y, classifier_module):
        from sklearn.model_selection import GridSearchCV
        from skorch import NeuralNetClassifier
        from skorch.utils import to_numpy

        net = NeuralNetClassifier(classifier_module)
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
        }
        gs = GridSearchCV(
            net, params, refit=True, cv=3, scoring='accuracy', error_score='raise'
        )
        # TODO: after sklearn > 1.6 is released, the to_numpy call should no longer be
        # required and be removed, see:
        # https://github.com/skorch-dev/skorch/pull/1078#discussion_r1887197261
        gs.fit(slds, to_numpy(y))  # does not raise

    def test_grid_search_with_slds_X_and_slds_y(
            self, slds, slds_y, classifier_module):
        from sklearn.model_selection import GridSearchCV
        from skorch import NeuralNetClassifier

        net = NeuralNetClassifier(
            classifier_module,
            train_split=False,
            verbose=0,
        )
        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
        }
        gs = GridSearchCV(
            net, params, refit=False, cv=3, scoring='accuracy', error_score='raise'
        )
        gs.fit(slds, slds_y)  # does not raise

    def test_index_with_2d_array_raises(self, slds):
        i = np.arange(4).reshape(2, 2)
        with pytest.raises(IndexError) as exc:
            # pylint: disable=pointless-statement
            slds[i]

        msg = ("SliceDataset only supports slicing with 1 "
               "dimensional arrays, got 2 dimensions instead.")
        assert exc.value.args[0] == msg

    @pytest.mark.parametrize('n', [0, 1])
    def test_slicedataset_to_numpy(self, slds_cls, custom_ds, n):
        from skorch.utils import to_numpy

        slds = slds_cls(custom_ds, idx=n)
        expected = custom_ds.X if n == 0 else custom_ds.y
        result = to_numpy(slds)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('n', [0, 1])
    @pytest.mark.parametrize('dtype', [None, np.float16, np.int32, np.complex64])
    def test_slicedataset_asarray(self, slds_cls, custom_ds, n, dtype):
        torch_to_numpy_dtype_dict = {
            torch.int64: np.int64,
            torch.float32: np.float32,
        }

        slds = slds_cls(custom_ds, idx=n)
        array = np.asarray(slds, dtype=dtype)
        expected = custom_ds.X if n == 0 else custom_ds.y
        assert array.shape == expected.shape

        if dtype is not None:
            assert array.dtype == dtype
        else:
            # if no dtype indicated, use original dtype of the data, or the
            # numpy equivalent if a torch dtype
            expected_dtype = torch_to_numpy_dtype_dict.get(expected.dtype, expected.dtype)
            assert array.dtype == expected_dtype

    @pytest.mark.parametrize('sl', [slice(0, 2), np.arange(3)])
    def test_subclass_getitem_returns_instance_of_itself(self, slds_cls, custom_ds, sl):
        class MySliceDataset(slds_cls):
            pass

        slds = MySliceDataset(custom_ds, idx=0)
        sliced = slds[sl]

        assert isinstance(sliced, MySliceDataset)


class TestPredefinedSplit():

    @pytest.fixture
    def predefined_split(self):
        from skorch.helper import predefined_split
        return predefined_split

    def test_pickle(self, predefined_split, data):
        from skorch.dataset import Dataset

        valid_dataset = Dataset(*data)
        train_split = predefined_split(valid_dataset)

        # does not raise
        pickle.dumps(train_split)


class TestDataFrameTransformer:
    @pytest.fixture
    def transformer_cls(self):
        from skorch.helper import DataFrameTransformer
        return DataFrameTransformer

    @pytest.mark.skipif(not pandas_installed, reason='pandas is not installed')
    @pytest.fixture
    def df(self):
        """DataFrame containing float, int, category types"""
        import pandas as pd

        df = pd.DataFrame({
            'col_floats': [0.1, 0.2, 0.3],
            'col_ints': [11, 11, 10],
            'col_cats': ['a', 'b', 'a'],
        })
        df['col_cats'] = df['col_cats'].astype('category')
        return df

    def test_fit_transform_defaults(self, transformer_cls, df):
        expected = {
            'X': np.asarray([
                [0.1, 11.0],
                [0.2, 11.0],
                [0.3, 10.0],
            ]).astype(np.float32),
            'col_cats': np.asarray([0, 1, 0]),
        }
        Xt = transformer_cls().fit_transform(df)
        assert_dicts_equal(Xt, expected)

    def test_fit_and_transform_defaults(self, transformer_cls, df):
        expected = {
            'X': np.asarray([
                [0.1, 11.0],
                [0.2, 11.0],
                [0.3, 10.0],
            ]).astype(np.float32),
            'col_cats': np.asarray([0, 1, 0]),
        }
        Xt = transformer_cls().fit(df).transform(df)
        assert_dicts_equal(Xt, expected)

    def test_fit_transform_defaults_two_categoricals(
            self, transformer_cls, df):
        expected = {
            'X': np.asarray([
                [0.1, 11.0],
                [0.2, 11.0],
                [0.3, 10.0],
            ]).astype(np.float32),
            'col_cats': np.asarray([0, 1, 0]),
            'col_foo': np.asarray([1, 1, 0]),
        }
        df = df.assign(col_foo=df['col_ints'].astype('category'))
        Xt = transformer_cls().fit_transform(df)
        assert_dicts_equal(Xt, expected)

    def test_fit_transform_int_as_categorical(self, transformer_cls, df):
        expected = {
            'X': np.asarray([0.1, 0.2, 0.3]).astype(np.float32).reshape(-1, 1),
            'col_ints': np.asarray([1, 1, 0]),
            'col_cats': np.asarray([0, 1, 0]),
        }
        Xt = transformer_cls(treat_int_as_categorical=True).fit_transform(df)
        assert_dicts_equal(Xt, expected)

    def test_fit_transform_no_X(self, transformer_cls, df):
        df = df[['col_ints', 'col_cats']]  # no float type present
        expected = {
            'col_ints': np.asarray([1, 1, 0]),
            'col_cats': np.asarray([0, 1, 0]),
        }
        Xt = transformer_cls(treat_int_as_categorical=True).fit_transform(df)
        assert_dicts_equal(Xt, expected)

    @pytest.mark.parametrize('data', [
        np.array([object, object, object]),
        np.array(['foo', 'bar', 'baz']),
    ])
    def test_invalid_dtype_raises(self, transformer_cls, df, data):
        df = df.assign(invalid=data)
        with pytest.raises(TypeError) as exc:
            transformer_cls().fit_transform(df)

        msg = exc.value.args[0]
        expected = ("The following columns have dtypes that cannot be "
                    "interpreted as numerical dtypes: invalid (object)")
        assert msg == expected

    def test_two_invalid_dtypes_raises(self, transformer_cls, df):
        df = df.assign(
            invalid0=np.array([object, object, object]),
            invalid1=np.array(['foo', 'bar', 'baz']),
        )
        with pytest.raises(TypeError) as exc:
            transformer_cls().fit_transform(df)

        msg = exc.value.args[0]
        expected = ("The following columns have dtypes that cannot be "
                    "interpreted as numerical dtypes: invalid0 (object), "
                    "invalid1 (object)")
        assert msg == expected

    @pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
    def test_set_float_dtype(self, transformer_cls, df, dtype):
        Xt = transformer_cls(float_dtype=dtype).fit_transform(df)
        assert Xt['X'].dtype == dtype

    @pytest.mark.parametrize('dtype', [np.int16, np.int32, np.int64])
    def test_set_int_dtype(self, transformer_cls, df, dtype):
        Xt = transformer_cls(
            treat_int_as_categorical=True, int_dtype=dtype).fit_transform(df)
        assert Xt['col_cats'].dtype == dtype
        assert Xt['col_ints'].dtype == dtype

    def test_leave_float_dtype_as_in_df(self, transformer_cls, df):
        # None -> don't cast
        Xt = transformer_cls(float_dtype=None).fit_transform(df)
        assert Xt['X'].dtype == np.float64

    def test_leave_int_dtype_as_in_df(self, transformer_cls, df):
        # None -> don't cast
        # pandas will use the lowest precision int that is capable to
        # encode the categories; since we only have 2 values, that is
        # int8 here
        Xt = transformer_cls(int_dtype=None).fit_transform(df)
        assert Xt['col_cats'].dtype == np.int8

    def test_column_named_X_present(self, transformer_cls, df):
        df = df.assign(X=df['col_cats'])
        with pytest.raises(ValueError) as exc:
            transformer_cls().fit(df)

        msg = exc.value.args[0]
        expected = ("DataFrame contains a column named 'X', which clashes "
                    "with the name chosen for cardinal features; consider "
                    "renaming that column.")
        assert msg == expected

    @pytest.fixture
    def module_cls(self):
        """Simple module with embedding and linear layers"""
        # pylint: disable=missing-docstring
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.reset_params()

            def reset_params(self):
                self.embedding = nn.Embedding(2, 10)
                self.linear = nn.Linear(2, 10)
                self.out = nn.Linear(20, 2)
                self.nonlin = nn.Softmax(dim=-1)

            # pylint: disable=arguments-differ
            def forward(self, X, col_cats):
                X_lin = self.linear(X)
                X_cat = self.embedding(col_cats)
                X_concat = torch.cat((X_lin, X_cat), dim=1)
                return self.nonlin(self.out(X_concat))

        return MyModule

    @pytest.fixture
    def net(self, module_cls):
        from skorch import NeuralNetClassifier
        net = NeuralNetClassifier(
            module_cls,
            train_split=None,
            max_epochs=3,
        )
        return net

    @pytest.fixture
    def pipe(self, transformer_cls, net):
        pipe = Pipeline([
            ('transform', transformer_cls()),
            ('net', net),
        ])
        return pipe

    def test_fit_and_predict_with_pipeline(self, pipe, df):
        y = np.asarray([0, 0, 1])
        pipe.fit(df, y)

        y_proba = pipe.predict_proba(df)
        assert y_proba.shape == (len(df), 2)

        y_pred = pipe.predict(df)
        assert y_pred.shape == (len(df),)

    def test_describe_signature_default_df(self, transformer_cls, df):
        result = transformer_cls().describe_signature(df)
        expected = {
            'X': {"dtype": torch.float32, "input_units": 2},
            'col_cats': {"dtype": torch.int64, "input_units": 2},
        }
        assert result == expected

    def test_describe_signature_non_default_df(self, transformer_cls, df):
        # replace float column with integer having 3 unique units
        df = df.assign(col_floats=[1, 2, 0])

        result = transformer_cls(
            treat_int_as_categorical=True).describe_signature(df)
        expected = {
            'col_floats': {"dtype": torch.int64, "input_units": 3},
            'col_ints': {"dtype": torch.int64, "input_units": 2},
            'col_cats': {"dtype": torch.int64, "input_units": 2},
        }
        assert result == expected

    def test_describe_signature_other_dtypes(self, transformer_cls, df):
        transformer = transformer_cls(
            float_dtype=np.float16,
            int_dtype=np.int32,
        )
        result = transformer.describe_signature(df)
        expected = {
            'X': {"dtype": torch.float16, "input_units": 2},
            'col_cats': {"dtype": torch.int32, "input_units": 2},
        }
        assert result == expected
