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
