"""Tests for scoring"""

from functools import partial
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, make_scorer
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
        with patch.object(net, 'score', side_effect=[10, 8, 6, 11, 7]):
            net.fit(*data)

        result = net.history[:, 'score']
        expected = [10, 8, 6, 11, 7]
        assert result == expected

    @pytest.mark.parametrize('lower_is_better, expected', [
        (True, [True, True, True, False, False]),
        (False, [True, False, False, True, False]),
    ])
    @pytest.mark.parametrize('initial_epochs', [1, 2, 3, 4])
    def test_scoring_uses_best_score_when_continuing_training(
            self, net_cls, module_cls, scoring_cls, data,
            lower_is_better, expected, tmpdir, initial_epochs,
    ):
        # set scoring to None so that mocked net.score is used
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring=None,
                on_train=True,
                lower_is_better=lower_is_better)],
            max_epochs=initial_epochs,
            # Set train_split to None so that the default 'valid_loss'
            # callback is effectively disabled and does not write to
            # the history. This should not cause problems when trying
            # to load best score for this scorer.
            train_split=None,
        )

        with patch.object(net, 'score', side_effect=[10, 8, 6, 11, 7]):
            net.fit(*data)

            history_fn = tmpdir.mkdir('skorch').join('history.json')
            net.save_params(f_history=str(history_fn))

            net.initialize()
            net.load_params(f_history=str(history_fn))
            net.max_epochs = 5 - initial_epochs
            net.partial_fit(*data)

        is_best = net.history[:, 'score_best']
        assert is_best == expected

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

        with patch.object(net, 'score', side_effect=[10, 8, 6, 11, 7]):
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

    def test_with_make_scorer_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(make_scorer(accuracy_score))],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'accuracy_score']
        assert result == [0, 0]

    def test_with_callable_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(accuracy_score)],
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        result = net.history[:, 'accuracy_score']
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
        def myscore(net, X, y=None):  # pylint: disable=unused-argument
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
        from skorch.dataset import ValidSplit
        import torch.utils.data.dataset
        from torch.utils.data.dataset import Subset

        class MyTorchDataset(torch.utils.data.dataset.TensorDataset):
            def __init__(self, X, y):
                super().__init__(
                    skorch.utils.to_tensor(X.reshape(-1, 1), device='cpu'),
                    skorch.utils.to_tensor(y, device='cpu'))

        class MySkorchDataset(skorch.dataset.Dataset):
            pass

        rawsplit = lambda ds: (ds, ds)
        valid_split = ValidSplit(2)

        def split_ignore_y(ds, y):
            return rawsplit(ds)

        table = [
            # Test a split where type(input) == type(output) is guaranteed
            (data, split_ignore_y, np.ndarray, False),
            (data, split_ignore_y, skorch.dataset.Dataset, True),
            ((MyTorchDataset(*data), None), rawsplit, MyTorchDataset, False),
            ((MyTorchDataset(*data), None), rawsplit, MyTorchDataset, True),
            ((MySkorchDataset(*data), None), rawsplit, np.ndarray, False),
            ((MySkorchDataset(*data), None), rawsplit, MySkorchDataset, True),

            # Test a split that splits datasets using torch Subset
            (data, valid_split, np.ndarray, False),
            (data, valid_split, Subset, True),
            ((MyTorchDataset(*data), None), valid_split, Subset, False),
            ((MyTorchDataset(*data), None), valid_split, Subset, True),
            ((MySkorchDataset(*data), None), valid_split, np.ndarray, False),
            ((MySkorchDataset(*data), None), valid_split, Subset, True),
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
        assert len(cbs['a1'].y_preds_) > 0

        for c1, c2 in zip(cbs['a1'].y_preds_, cbs['a2'].y_preds_):
            assert id(c1) == id(c2)

        for c1, c2 in zip(cbs['a1'].y_trues_, cbs['a2'].y_trues_):
            assert id(c1) == id(c2)

    def test_multiple_scorings_with_dict(
            self, net_cls, module_cls, train_split, scoring_cls, data):
        # This test checks if an exception is raised when a dictionary is passed as scorer.
        net = net_cls(
            module=module_cls,
            callbacks=[
                scoring_cls({'a1': 'accuracy', 'a2': 'accuracy'}),
            ],
            train_split=train_split,
            max_epochs=2,
        )

        msg = "Dict not supported as scorer for multi-metric scoring"
        with pytest.raises(ValueError, match=msg):
                net.fit(*data)

    @pytest.mark.parametrize('use_caching, count', [(False, 1), (True, 0)])
    def test_with_caching_get_iterator_not_called(
            self, net_cls, module_cls, train_split, caching_scoring_cls, data,
            use_caching, count,
    ):
        max_epochs = 3
        net = net_cls(
            module=module_cls,
            callbacks=[
                ('acc', caching_scoring_cls('accuracy', use_caching=use_caching)),
            ],
            train_split=train_split,
            max_epochs=max_epochs,
        )

        get_iterator = net.get_iterator
        net.get_iterator = Mock(side_effect=get_iterator)
        net.fit(*data)

        # expected count should be:
        # max_epochs * (1 (train) + 1 (valid) + 0 or 1 (from scoring,
        # depending on caching))
        count_expected = max_epochs * (1 + 1 + count)
        assert net.get_iterator.call_count == count_expected

    def test_subclassing_epoch_scoring(
            self, classifier_module, classifier_data):
        # This test's purpose is to check that it is possible to
        # easily subclass EpochScoring by overriding on_epoch_end to
        # record 2 scores.
        from skorch import NeuralNetClassifier
        from skorch.callbacks import EpochScoring

        class MyScoring(EpochScoring):
            def on_epoch_end(
                    self,
                    net,
                    dataset_train,
                    dataset_valid,
                    **kwargs):
                _, y_test, y_proba = self.get_test_data(
                    dataset_train, dataset_valid)
                y_pred = np.concatenate(y_proba).argmax(1)

                # record 2 valid scores
                score_0_valid = accuracy_score(y_test, y_pred)
                net.history.record('score_0', score_0_valid)
                score_1_valid = accuracy_score(y_test, y_pred) + 1
                net.history.record('score_1', score_1_valid)

        X, y = classifier_data
        net = NeuralNetClassifier(
            classifier_module,
            callbacks=[MyScoring(scoring=None)],
            max_epochs=1,
        )
        net.fit(X, y)

        row = net.history[-1]
        keys_history = set(row.keys())
        keys_expected = {'score_0', 'score_1'}

        assert keys_expected.issubset(keys_history)
        assert np.isclose(row['score_0'], row['score_1'] - 1)


class TestBatchScoring:
    @pytest.fixture(params=[{'use_caching': True}, {'use_caching': False}])
    def scoring_cls(self, request):
        from skorch.callbacks import BatchScoring
        return partial(BatchScoring, **request.param)

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
        from skorch.utils import train_loss_score
        return scoring_cls(
            train_loss_score,
            name='train_loss',
            on_train=True,
        ).initialize()

    @pytest.fixture
    def valid_loss(self, scoring_cls):
        from skorch.utils import valid_loss_score
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
    ])
    @pytest.mark.parametrize('initial_epochs', [1, 2, 3, 4])
    def test_scoring_uses_best_score_when_continuing_training(
            self, net_cls, module_cls, scoring_cls, data,
            lower_is_better, expected, tmpdir, initial_epochs,
    ):
        # set scoring to None so that mocked net.score is used
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(
                scoring=None,
                on_train=True,
                lower_is_better=lower_is_better)],
            max_epochs=initial_epochs,
            # Set train_split to None so that the default 'valid_loss'
            # callback is effectively disabled and does not write to
            # the history. This should not cause problems when trying
            # to load best score for this scorer.
            train_split=None,
        )

        with patch.object(net, 'score', side_effect=[10, 8, 6, 11, 7]):
            net.fit(*data)

            history_fn = tmpdir.mkdir('skorch').join('history.json')
            net.save_params(f_history=str(history_fn))

            net.max_epochs = 5 - initial_epochs
            net.initialize()
            net.load_params(f_history=str(history_fn))
            net.partial_fit(*data)

        is_best = net.history[:, 'score_best']
        assert is_best == expected

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
        with patch.object(net, 'score', side_effect=[10, 8, 6, 11, 7]):
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

    def test_with_make_scorer_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(make_scorer(accuracy_score))],
            batch_size=1,
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        score_epochs = net.history[:, 'accuracy_score']
        assert np.allclose(score_epochs, [0, 0])

        score_batches = net.history[:, 'batches', :, 'accuracy_score']
        assert np.allclose(score_batches, [[0, 0], [0, 0]])

    def test_with_callable_accuracy_score(
            self, net_cls, module_cls, scoring_cls, train_split, data,
    ):
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(accuracy_score)],
            batch_size=1,
            max_epochs=2,
            train_split=train_split,
        )
        net.fit(*data)

        score_epochs = net.history[:, 'accuracy_score']
        assert np.allclose(score_epochs, [0, 0])

        score_batches = net.history[:, 'batches', :, 'accuracy_score']
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

    def test_without_target_data_works(
            self, net_cls, module_cls, scoring_cls, data,
    ):
        score_calls = 0

        def myscore(net, X, y=None):  # pylint: disable=unused-argument
            nonlocal score_calls
            score_calls += 1
            return X.mean().data.item()

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

    def test_scoring_with_cache_and_fit_interrupt_resets_infer(
            self, net_cls, module_cls, scoring_cls, data, train_split):
        # This test addresses a bug that occurred with caching in
        # scoring when training is interrupted irregularly
        # (e.g. through KeyboardInterrupt). In that case, it is
        # important that the net's infer method is still reset.

        def interrupt_scoring(net, X, y):
            raise KeyboardInterrupt

        X, y = data
        net = net_cls(
            module_cls,
            callbacks=[('interrupt', scoring_cls(interrupt_scoring))],
            train_split=train_split,
        )
        net.fit(X, y)

        y_pred = net.predict(X)
        # We test that we predict as many outputs as we put in. With
        # the bug, the cache would be exhausted early because of the
        # train split, and we would get back less.
        assert len(y_pred) == len(X)


class TestPassthrougScoring:
    @pytest.fixture
    def scoring_cls(self, request):
        from skorch.callbacks import PassthroughScoring
        return PassthroughScoring

    @pytest.fixture
    def train_loss(self, scoring_cls):
        # use train batch size to stand in for batch-level scores
        return scoring_cls(name='train_batch_size', on_train=True)

    @pytest.fixture
    def valid_loss(self, scoring_cls):
        # use valid batch size to stand in for batch-level scores
        return scoring_cls(name='valid_batch_size')

    @pytest.fixture
    def net(self, classifier_module, train_loss, valid_loss, classifier_data):
        from skorch import NeuralNetClassifier
        net = NeuralNetClassifier(
            classifier_module,
            batch_size=10,
            # use train and valid batch size to stand in for
            # batch-level scores
            callbacks=[train_loss, valid_loss],
            max_epochs=2)

        X, y = classifier_data
        n = 75
        # n=75 with a 4/5 train/valid split -> 60/15 samples; with a
        # batch size of 10, that leads to train batch sizes of
        # [10,10,10,10] and valid batch sizes of [10,5]; all labels
        # are set to 0 to ensure that the stratified split is exactly
        # equal to the desired split
        y = np.zeros_like(y)
        return net.fit(X[:n], y[:n])

    @pytest.fixture
    def history(self, net):
        return net.history

    @pytest.fixture
    def history_empty(self):
        from skorch.history import History
        return History()

    def test_correct_train_pass_through_scores(self, history):
        # train: average of [10,10,10,10,10] is 10
        train_scores = history[:, 'train_batch_size']
        assert np.allclose(train_scores, 10.0)

    def test_correct_valid_pass_through_scores(self, history):
        # valid: average of [10,5] with weights also being [10,5] =
        # (10*10 + 5*5)/15
        expected = (10 * 10 + 5 * 5) / 15  # 8.333..
        valid_losses = history[:, 'valid_batch_size']
        assert np.allclose(valid_losses, [expected, expected])

    def test_missing_entry_in_epoch(self, scoring_cls, history_empty):
        """We skip one entry in history_empty. This batch should simply be
        ignored.

        """
        history_empty.new_epoch()
        history_empty.new_batch()
        history_empty.record_batch('score', 10)
        history_empty.record_batch('train_batch_size', 10)

        history_empty.new_batch()
        # this score is ignored since it has no associated batch size
        history_empty.record_batch('score', 20)

        net = Mock(history=history_empty)
        cb = scoring_cls(name='score', on_train=True).initialize()
        cb.on_epoch_end(net)

        train_score = history_empty[-1, 'score']
        assert np.isclose(train_score, 10.0)

    @pytest.mark.parametrize('lower_is_better, expected', [
        (True, [True, True, True, False, False]),
        (False, [True, False, False, True, False]),
        (None, []),
    ])
    def test_lower_is_better_is_honored(
            self, net_cls, module_cls, scoring_cls, train_split, data,
            history_empty, lower_is_better, expected,
    ):
        # results in expected patterns of True and False
        scores = [10, 8, 6, 11, 7]

        cb = scoring_cls(
            name='score',
            lower_is_better=lower_is_better,
        ).initialize()

        net = Mock(history=history_empty)
        for score in scores:
            history_empty.new_epoch()
            history_empty.new_batch()
            history_empty.record_batch('score', score)
            history_empty.record_batch('valid_batch_size', 55)  # doesn't matter
            cb.on_epoch_end(net)

        if lower_is_better is not None:
            is_best = history_empty[:, 'score_best']
            assert is_best == expected
        else:
            # if lower_is_better==None, don't write score
            with pytest.raises(KeyError):
                # pylint: disable=pointless-statement
                history_empty[:, 'score_best']

    def test_no_error_when_no_valid_data(
            self, net_cls, module_cls, scoring_cls, data,
    ):
        # we set the name to 'valid_batch_size' but disable
        # train/valid split -- there should be no error
        net = net_cls(
            module_cls,
            callbacks=[scoring_cls(name='valid_batch_size')],
            max_epochs=1,
            train_split=None,
        )
        # does not raise
        net.fit(*data)
