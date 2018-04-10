"""Tests for scoring"""

from functools import partial
import itertools
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from skorch.utils import to_numpy


class TestEpochScoring:
    @pytest.fixture(params=[{'use_caching': True}, {'use_caching': False}])
    def scoring_cls(self, request):
        from skorch.callbacks import EpochScoring
        return partial(EpochScoring, **request.param)

    @pytest.fixture
    def caching_scoring_cls(self):
        from skorch.callbacks import EpochScoring
        return partial(EpochScoring, use_caching=True)

    @pytest.fixture(params=[{'use_caching': True}, {'use_caching': False}])
    def mse_scoring(self, request, scoring_cls):
        return scoring_cls(
            'neg_mean_squared_error',
            name='nmse',
            **request.param,
        ).initialize()

    def test_correct_valid_score(
            self, net_cls, module_cls, mse_scoring, train_split, data,
    ):
        net = net_cls(
            module=module_cls,
            callbacks=[mse_scoring],
            train_split=train_split,
            max_epochs=2,
        )
        net.fit(*data)

        expected = -np.mean([(3 - 5) ** 2, (0 - 4) ** 2])
        loss = net.history[:, 'nmse']
        assert np.allclose(loss, expected)

    def test_correct_train_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module=module_cls,
            callbacks=[scoring_cls(
                'neg_mean_squared_error',
                on_train=True,
                name='nmse',
                lower_is_better=False,
            )],
            train_split=train_split,
            max_epochs=2,
        )
        net.fit(*data)

        expected = -np.mean([(0 - -1) ** 2, (2 - 0) ** 2])
        loss = net.history[:, 'nmse']
        assert np.allclose(loss, expected)

    def test_scoring_uses_score_when_none(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None)],
            max_epochs=5,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'score']
        # these values are the hard-coded side_effects from net.score
        expected = [10, 8, 6, 11, 7]
        assert result == expected

    @pytest.mark.parametrize('lower_is_better, expected', [
        (True, [True, True, True, False, False]),
        (False, [True, False, False, True, False]),
        (None, []),
    ])
    def test_best_score_when_lower_is_better(
            self, net_cls, module_cls, scoring_cls, train_split, data,
            lower_is_better, expected,
    ):
        # set scoring to None so that mocked net.score is used
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring=None,
                lower_is_better=lower_is_better)],
            train_split=train_split,
            max_epochs=5,
        )
        net.fit(*data)

        if lower_is_better is not None:
            is_best = net.history[:, 'score_best']
            assert is_best == expected
        else:
            # if lower_is_better==None, don't write score
            with pytest.raises(KeyError):
                # pylint: disable=pointless-statement
                net.history[:, 'score_best']

    def test_no_error_when_no_valid_data(
            self, net_cls, module_cls, mse_scoring, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[mse_scoring],
            max_epochs=3,
            train_split=train_split,
        )
        net.fit(*data)

        net.train_split = None
        # does not raise
        net.partial_fit(*data)

        # only the first 3 epochs wrote scores
        assert len(net.history[:, 'nmse']) == 3

    def test_with_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('accuracy')],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'accuracy']
        assert result == [0, 0]

    def test_with_score_nonexisting_string(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('does-not-exist')],
            max_epochs=2,
            train_split=train_split,
        )
        with pytest.raises(ValueError) as exc:
            net.fit(*data)
        msg = "'does-not-exist' is not a valid scoring value."
        assert exc.value.args[0].startswith(msg)

    def test_with_score_as_custom_func(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'score55']
        assert result == [55, 55]

    def test_with_name_none_returns_score_as_name(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score']

    def test_explicit_name_is_used_in_history(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name='myname')],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'myname']

    def test_with_scoring_str_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring='neg_mean_squared_error', name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'neg_mean_squared_error']

    def test_with_with_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55, name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_with_with_partial_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(partial(score55, foo=0), name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_target_extractor_is_called(
            self, net_cls, module_cls, train_split, scoring_cls, data):
        X, y = data
        extractor = Mock(side_effect=to_numpy)
        scoring = scoring_cls(
            name='nmse',
            scoring='neg_mean_squared_error',
            target_extractor=extractor,
        )
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[scoring], max_epochs=2)
        net.fit(X, y)

        # With caching in use the extractor should be called for
        # each y of a batch. Without caching it should called
        # once per epoch (since we get all data at once).
        if scoring.use_caching:
            assert len(y) // net.batch_size == 4
            assert extractor.call_count == 4
        else:
            assert extractor.call_count == 2

    def test_without_target_data_works(
            self, net_cls, module_cls, scoring_cls, data,
    ):
        score_calls = 0

        def myscore(net, X, y=None):
            nonlocal score_calls
            score_calls += 1
            assert y is None

            # In case we use caching X is a dataset. We need to
            # extract X ourselves.
            if dict(net.callbacks_)['EpochScoring'].use_caching:
                return np.mean(X.X)
            return np.mean(X)

        # pylint: disable=unused-argument
        def mysplit(dataset, y):
            # set y_valid to None
            ds_train = dataset
            ds_valid = type(dataset)(dataset.X, y=None)
            return ds_train, ds_valid

        X, y = data
        net = net_cls(
            module=module_cls,
            callbacks=[scoring_cls(myscore)],
            train_split=mysplit,
            max_epochs=2,
        )
        net.fit(X, y)

        assert score_calls == 2

        expected = np.mean(X)
        loss = net.history[:, 'myscore']
        assert np.allclose(loss, expected)

    def net_input_is_scoring_input(
            self, net_cls, module_cls, scoring_cls, input_data,
            train_split, expected_type, caching,
    ):
        score_calls = 0
        def myscore(net, X, y=None):
            nonlocal score_calls
            score_calls += 1
            assert type(X) == expected_type
            return 0

        max_epochs = 2
        net = net_cls(
            module=module_cls,
            callbacks=[scoring_cls(myscore, use_caching=caching)],
            train_split=train_split,
            max_epochs=max_epochs,
        )
        net.fit(*input_data)
        assert score_calls == max_epochs

    def test_net_input_is_scoring_input(
            self, net_cls, module_cls, scoring_cls, data,
    ):
        # Make sure that whatever data type is put in the network is
        # received at the scoring side as well. For the caching case
        # we only receive datasets.
        import skorch
        from skorch.dataset import CVSplit
        import torch
        import torch.utils.data.dataset

        class MyTorchDataset(torch.utils.data.dataset.TensorDataset):
            def __init__(self, X, y):
                super().__init__(
                    skorch.utils.to_tensor(X.reshape(-1, 1), False),
                    skorch.utils.to_tensor(y, False))

        class MySkorchDataset(skorch.dataset.Dataset):
            pass

        rawsplit = lambda ds, _: (ds, ds)
        cvsplit = CVSplit(2, random_state=0)

        table = [
            # Test a split where type(input) == type(output) is guaranteed
            (data, rawsplit, np.ndarray, False),
            (data, rawsplit, skorch.dataset.Dataset, True),
            ((MyTorchDataset(*data), None), rawsplit, MyTorchDataset, False),
            ((MyTorchDataset(*data), None), rawsplit, MyTorchDataset, True),
            ((MySkorchDataset(*data), None), rawsplit, np.ndarray, False),
            ((MySkorchDataset(*data), None), rawsplit, MySkorchDataset, True),

            # Test a split that splits datasets using torch Subset
            (data, cvsplit, np.ndarray, False),
            (data, cvsplit, skorch.helper.Subset, True),
            ((MyTorchDataset(*data), None), cvsplit, skorch.helper.Subset, False),
            ((MyTorchDataset(*data), None), cvsplit, skorch.helper.Subset, True),
            ((MySkorchDataset(*data), None), cvsplit, np.ndarray, False),
            ((MySkorchDataset(*data), None), cvsplit, skorch.helper.Subset, True),
        ]

        for input_data, train_split, expected_type, caching in table:
            self.net_input_is_scoring_input(
                net_cls,
                module_cls,
                scoring_cls,
                input_data,
                train_split,
                expected_type,
                caching)

    def test_multiple_scorings_share_cache(
            self, net_cls, module_cls, train_split, caching_scoring_cls, data,
    ):
        net = net_cls(
            module=module_cls,
            callbacks=[
                ('a1', caching_scoring_cls('accuracy')),
                ('a2', caching_scoring_cls('accuracy')),
            ],
            train_split=train_split,
            max_epochs=2,
        )

        # on_train_end clears cache, overwrite so we can inspect the contents.
        with patch('skorch.callbacks.scoring.EpochScoring.on_train_end',
                  lambda *x, **y: None):
            net.fit(*data)

        cbs = dict(net.callbacks_)
        assert cbs['a1'].use_caching
        assert cbs['a2'].use_caching
        assert len(cbs['a1'].y_valid_preds_) > 0

        for c1, c2 in zip(cbs['a1'].y_valid_preds_, cbs['a2'].y_valid_preds_):
            assert id(c1) == id(c2)

        for c1, c2 in zip(cbs['a1'].y_valid_trues_, cbs['a2'].y_valid_trues_):
            assert id(c1) == id(c2)

class TestBatchScoring:
    @pytest.fixture
    def scoring_cls(self):
        from skorch.callbacks import BatchScoring
        return BatchScoring

    @pytest.fixture
    def mse_scoring(self, scoring_cls):
        return scoring_cls(
            name='nmse',
            scoring='neg_mean_squared_error',
        ).initialize()

    @pytest.fixture
    def net(self, net_cls, module_cls, train_split, mse_scoring, data):
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[mse_scoring], max_epochs=2)
        return net.fit(*data)

    @pytest.fixture
    def train_loss(self, scoring_cls):
        from skorch.net import train_loss_score
        return scoring_cls(
            train_loss_score,
            name='train_loss',
            on_train=True,
        ).initialize()

    @pytest.fixture
    def valid_loss(self, scoring_cls):
        from skorch.net import valid_loss_score
        return scoring_cls(
            valid_loss_score,
            name='valid_loss',
        ).initialize()

    @pytest.fixture
    def history(self, net):
        return net.history

    def test_correct_train_loss_values(self, history):
        train_losses = history[:, 'train_loss']
        expected = np.mean([(0 - -1) ** 2, (2 - 0) ** 2])
        assert np.allclose(train_losses, expected)

    def test_correct_valid_loss_values(self, history):
        valid_losses = history[:, 'valid_loss']
        expected = np.mean([(3 - 5) ** 2, (0 - 4) ** 2])
        assert np.allclose(valid_losses, expected)

    def test_correct_mse_values_for_batches(self, history):
        nmse = history[:, 'batches', :, 'nmse']
        expected_per_epoch = [-(3 - 5) ** 2, -(0 - 4) ** 2]
        # for the 2 epochs, the loss is the same
        expected = [expected_per_epoch, expected_per_epoch]
        assert np.allclose(nmse, expected)

    def test_missing_batch_size(self, train_loss, history):
        """We skip one batch size entry in history. This batch should
        simply be ignored.

        """
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 20)
        # missing batch size, loss of 20 is ignored

        net = Mock(history=history)
        train_loss.on_epoch_end(net)

        assert history[-1, 'train_loss'] == 10

    def test_average_honors_weights(self, train_loss, history):
        """The batches may have different batch sizes, which is why it
        necessary to honor the batch sizes. Here we use different
        batch sizes to verify this.

        """
        from skorch.history import History

        history = History()
        history.new_epoch()
        history.new_batch()
        history.record_batch('train_loss', 10)
        history.record_batch('train_batch_size', 1)
        history.new_batch()
        history.record_batch('train_loss', 40)
        history.record_batch('train_batch_size', 2)

        net = Mock(history=history)
        train_loss.on_epoch_end(net)

        assert history[0, 'train_loss'] == 30

    @pytest.mark.parametrize('lower_is_better, expected', [
        (True, [True, True, True, False, False]),
        (False, [True, False, False, True, False]),
        (None, []),
    ])
    def test_best_score_when_lower_is_better(
            self, net_cls, module_cls, scoring_cls, train_split, data,
            lower_is_better, expected,
    ):
        # set scoring to None so that mocked net.score is used
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring=None,
                lower_is_better=lower_is_better)],
            train_split=train_split,
            max_epochs=5,
        )
        net.fit(*data)

        if lower_is_better is not None:
            is_best = net.history[:, 'score_best']
            assert is_best == expected
        else:
            # if lower_is_better==None, don't write score
            with pytest.raises(KeyError):
                # pylint: disable=pointless-statement
                net.history[:, 'score_best']

    def test_no_error_when_no_valid_data(
            self, net_cls, module_cls, mse_scoring, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[mse_scoring],
            max_epochs=1,
            train_split=None,
        )
        # does not raise
        net.fit(*data)

    def test_with_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('accuracy')],
            batch_size=1,
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        score_epochs = net.history[:, 'accuracy']
        assert np.allclose(score_epochs, [0, 0])

        score_batches = net.history[:, 'batches', :, 'accuracy']
        assert np.allclose(score_batches, [[0, 0], [0, 0]])

    def test_with_score_nonexisting_string(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls('does-not-exist')],
            max_epochs=2,
            train_split=train_split,
        )
        with pytest.raises(ValueError) as exc:
            net.fit(*data)
        msg = "'does-not-exist' is not a valid scoring value."
        assert exc.value.args[0].startswith(msg)

    def test_with_score_as_custom_func(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        score_epochs = net.history[:, 'score55']
        assert np.allclose(score_epochs, [55, 55])

        score_batches = net.history[:, 'batches', :, 'score55']
        assert np.allclose(score_batches, [[55, 55], [55, 55]])

    def test_with_name_none_returns_score_as_name(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score']

    def test_explicit_name_is_used_in_history(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(scoring=None, name='myname')],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'myname']

    def test_with_scoring_str_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring='neg_mean_squared_error', name=None)],
            max_epochs=1,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'neg_mean_squared_error']

    def test_with_with_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(score55, name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_with_with_partial_custom_func_and_name_none(
            self, net_cls, module_cls, scoring_cls, train_split, data, score55,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(partial(score55, foo=0), name=None)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)
        assert net.history[:, 'score55']

    def test_target_extractor_is_called(
            self, net_cls, module_cls, train_split, scoring_cls, data):
        X, y = data
        extractor = Mock(side_effect=to_numpy)
        scoring = scoring_cls(
            name='nmse',
            scoring='neg_mean_squared_error',
            target_extractor=extractor,
        )
        net = net_cls(
            module_cls, batch_size=1, train_split=train_split,
            callbacks=[scoring], max_epochs=2)
        net.fit(X, y)

        assert extractor.call_count == 2 * 2
