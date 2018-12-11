"""Test for helper.py"""
import pickle

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification


class TestSliceDict:
    def assert_dicts_equal(self, d0, d1):
        assert d0.keys() == d1.keys()
        for key in d0.keys():
            assert np.allclose(d0[key], d1[key])

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
        self.assert_dicts_equal(sliced, sldict_cls(**expected))

    def test_slice_list(self, sldict, sldict_cls):
        result = sldict[[0, 2]]
        expected = sldict_cls(
            f0=np.array([0, 2]),
            f1=np.array([[0, 1, 2], [6, 7, 8]]))
        self.assert_dicts_equal(result, expected)

    def test_slice_mask(self, sldict, sldict_cls):
        result = sldict[np.array([1, 0, 1, 0]).astype(bool)]
        expected = sldict_cls(
            f0=np.array([0, 2]),
            f1=np.array([[0, 1, 2], [6, 7, 8]]))
        self.assert_dicts_equal(result, expected)

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
        self.assert_dicts_equal(result, sldict)

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
        gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy',
                          iid=True)
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

        assert copied == copied
        assert not copied == sldict
        assert copied != sldict

    def test_equals_arrays_deep(self, sldict):
        copied = sldict.copy()
        copied['f0'] = np.array(copied['f0'].copy())

        assert copied == copied
        assert copied == sldict

    def test_equals_tensors(self, sldict_cls):
        sldict = sldict_cls(
            f0=torch.arange(4),
            f1=torch.arange(12).reshape(4, 3),
        )
        copied = sldict.copy()
        copied['f0'] = -copied['f0']

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


# TODO: remove in 0.5.0
class TestFilterParameterGroupsRequiresGrad():

    @pytest.fixture
    def filter_requires_grad(self):
        from skorch.helper import filter_requires_grad
        return filter_requires_grad

    def test_all_parameters_requires_gradient(self, filter_requires_grad):
        pgroups = [{
            'params': [torch.zeros(1, requires_grad=True),
                       torch.zeros(1, requires_grad=True)],
            'lr': 0.1
        }, {
            'params': [torch.zeros(1, requires_grad=True)]
        }]

        with pytest.warns(DeprecationWarning):
            filter_pgroups = list(filter_requires_grad(pgroups))
        assert len(filter_pgroups) == 2
        assert len(list(filter_pgroups[0]['params'])) == 2
        assert len(list((filter_pgroups[1]['params']))) == 1

        assert filter_pgroups[0]['lr'] == 0.1

    def test_some_params_requires_gradient(self, filter_requires_grad):
        pgroups = [{
            'params': [
                torch.zeros(1, requires_grad=True),
                torch.zeros(1, requires_grad=False)
            ], 'lr': 0.1
        }, {
            'params': [torch.zeros(1, requires_grad=False)]
        }]

        with pytest.warns(DeprecationWarning):
            filter_pgroups = list(filter_requires_grad(pgroups))
        assert len(filter_pgroups) == 2
        assert len(list(filter_pgroups[0]['params'])) == 1
        assert not list(filter_pgroups[1]['params'])

        assert filter_pgroups[0]['lr'] == 0.1

    def test_does_not_drop_group_when_requires_grad_is_false(
            self, filter_requires_grad):
        pgroups = [{
            'params': [
                torch.zeros(1, requires_grad=False),
                torch.zeros(1, requires_grad=False)
            ], 'lr': 0.1
        }, {
            'params': [torch.zeros(1, requires_grad=False)]
        }]

        with pytest.warns(DeprecationWarning):
            filter_pgroups = list(filter_requires_grad(pgroups))
        assert len(filter_pgroups) == 2
        assert not list(filter_pgroups[0]['params'])
        assert not list(filter_pgroups[1]['params'])

        assert filter_pgroups[0]['lr'] == 0.1


# TODO: remove in 0.5.0
class TestOptimizerParamsRequiresGrad:

    @pytest.fixture
    def filtered_optimizer(self):
        from skorch.helper import filtered_optimizer
        return filtered_optimizer

    @pytest.fixture
    def filter_requires_grad(self):
        from skorch.helper import filter_requires_grad
        return filter_requires_grad

    def test_passes_filtered_cgroups(
            self, filtered_optimizer, filter_requires_grad):
        pgroups = [{
            'params': [torch.zeros(1, requires_grad=True),
                       torch.zeros(1, requires_grad=False)],
            'lr': 0.1
        }, {
            'params': [torch.zeros(1, requires_grad=True)]
        }]

        with pytest.warns(DeprecationWarning):
            opt = filtered_optimizer(torch.optim.SGD, filter_requires_grad)
            filtered_opt = opt(pgroups, lr=0.2)

        assert isinstance(filtered_opt, torch.optim.SGD)
        assert len(list(filtered_opt.param_groups[0]['params'])) == 1
        assert len(list(filtered_opt.param_groups[1]['params'])) == 1

        assert filtered_opt.param_groups[0]['lr'] == 0.1
        assert filtered_opt.param_groups[1]['lr'] == 0.2

    def test_passes_kwargs_to_neuralnet_optimizer(
            self, filtered_optimizer, filter_requires_grad):
        from skorch import NeuralNetClassifier
        from skorch.toy import make_classifier

        module_cls = make_classifier(
            input_units=1,
            num_hidden=0,
            output_units=1,
        )

        with pytest.warns(DeprecationWarning):
            opt = filtered_optimizer(torch.optim.SGD, filter_requires_grad)
            net = NeuralNetClassifier(
                module_cls, optimizer=opt, optimizer__momentum=0.9)
            net.initialize()

        assert isinstance(net.optimizer_, torch.optim.SGD)
        assert len(net.optimizer_.param_groups) == 1
        assert net.optimizer_.param_groups[0]['momentum'] == 0.9

    def test_pickle(self, filtered_optimizer, filter_requires_grad):
        with pytest.warns(DeprecationWarning):
            opt = filtered_optimizer(torch.optim.SGD, filter_requires_grad)
        # Does not raise
        pickle.dumps(opt)


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
