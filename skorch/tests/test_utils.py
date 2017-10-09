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
