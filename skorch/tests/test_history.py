"""Tests for history.py."""

import pickle
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pytest
import torch
from torch.distributed import TCPStore
from sklearn.datasets import make_classification


class TestHistory:

    test_epochs = 3
    test_batches = 4

    @pytest.fixture(scope='class', params=['single', 'distributed'])
    def history_cls(self, request):
        # run tests once with default History, once with DistributedHistory
        from skorch.history import DistributedHistory, History
        from skorch._version import Version

        if request.param == 'single':
            return History

        if request.param == 'distributed':
            store = TCPStore("127.0.0.1", port=1234, world_size=1, is_master=True)
            return partial(
                DistributedHistory, store=store, rank=0, world_size=1
            )

        raise ValueError("Incorrect pytest request parameter '{request.param}'")

    @pytest.fixture
    def history(self, history_cls):
        """Return a history filled with epoch and batch data."""
        h = history_cls()
        for num_epoch in range(self.test_epochs):
            h.new_epoch()
            h.record('duration', 1)
            h.record('total_loss', num_epoch + self.test_batches)
            if num_epoch == 2:
                h.record('extra', 42)
            for num_batch in range(self.test_batches):
                h.new_batch()
                h.record_batch('loss', num_epoch + num_batch)
                if num_batch % 2 == 0 and (num_epoch + 1) != self.test_epochs:
                    h.record_batch('extra_batch', 23)
        return h

    @pytest.fixture
    def ref(self, history):
        return history.to_list()

    def test_list_initialization(self, history_cls):
        h = history_cls([1, 2, 3])
        assert len(h) == 3

    def test_history_length(self, history):
        assert len(history) == self.test_epochs
        # we expect to have the extracted batches for each epoch
        assert len(history[:, 'batches']) == self.test_epochs

    def test_history_epoch_column(self, history, ref):
        total_losses = history[:, 'total_loss']
        total_losses_ref = [n['total_loss'] for n in ref]
        assert total_losses == total_losses_ref

    def test_history_epoch_two_columns(self, history, ref):
        duration_with_losses = history[:, ('total_loss', 'duration')]

        total_losses_ref = [n['total_loss'] for n in ref]
        durations_ref = [n['duration'] for n in ref]
        expected = list(zip(total_losses_ref, durations_ref))

        assert duration_with_losses == expected

    def test_history_epoch_two_columns_different_order(self, history, ref):
        duration_with_losses = history[:, ('duration', 'total_loss')]

        total_losses_ref = [n['total_loss'] for n in ref]
        durations_ref = [n['duration'] for n in ref]
        expected = list(zip(durations_ref, total_losses_ref))

        assert duration_with_losses == expected

    def test_history_partial_index(self, history, ref):
        extra = history[:, 'extra']
        assert len(extra) == 1
        # we retrieve 'extra' from a slice, therefore we expect a list as result
        assert extra == [ref[2]['extra']]

    def test_history_partial_and_full_index(self, history, ref):
        total_loss_with_extra = history[:, ('total_loss', 'extra')]

        assert len(total_loss_with_extra) == 1
        assert total_loss_with_extra[0][0] == ref[2]['total_loss']
        assert total_loss_with_extra[0][1] == ref[2]['extra']

    def test_history_partial_join_list(self, history, ref):
        total = history[:, ['total_loss', 'extra', 'batches']]

        # there's only one epoch with the 'extra' key.
        assert len(total) == 1
        assert total[0][0] == ref[2]['total_loss']
        assert total[0][1] == ref[2]['extra']
        assert total[0][2] == ref[2]['batches']

    def test_history_retrieve_single_value(self, history, ref):
        total_loss_0 = history[0, 'total_loss']
        assert total_loss_0 == ref[0]['total_loss']

    def test_history_retrieve_multiple_values(self, history, ref):
        total_loss_0_to_1 = history[0:1, 'total_loss']
        assert total_loss_0_to_1 == [n['total_loss'] for n in ref[0:1]]

    def test_history_non_existing_values(self, history):
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            history[:, 'non-existing']
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            history[0, 'extra']

    def test_history_non_existing_values_batch(self, history):
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            history[:, 'batches', :, 'non-existing']
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            history[:, 'batches', 1, 'extra_batch']

    def test_history_mixed_slicing(self, history, ref):
        losses = history[:, 'batches', 0, 'loss']

        assert len(losses) == self.test_epochs
        assert losses == [epoch['batches'][0]['loss'] for epoch in ref]

        losses = history[0, 'batches', :, 'loss']
        assert losses == [batch['loss'] for batch in ref[0]['batches']]

    def test_history_partial_and_full_index_batches(self, history, ref):
        loss_with_extra = history[:, 'batches', :, ('loss', 'extra_batch')]

        expected_e0 = [(b['loss'], b['extra_batch']) for b in ref[0]['batches']
                       if 'extra_batch' in b]
        expected_e1 = [(b['loss'], b['extra_batch']) for b in ref[1]['batches']
                       if 'extra_batch' in b]

        assert len(loss_with_extra) == self.test_epochs - 1
        assert loss_with_extra[0] == expected_e0
        assert loss_with_extra[1] == expected_e1

    def test_history_partial_batches_batch_key_3rd(self, history, ref):
        extra_batches = history[:, 'batches', :, 'extra_batch']

        expected_e0 = [b['extra_batch'] for b in ref[0]['batches']
                       if 'extra_batch' in b]
        expected_e1 = [b['extra_batch'] for b in ref[1]['batches']
                       if 'extra_batch' in b]

        # In every epoch there are 2 batches with the 'extra_batch'
        # key except for the last epoch. We therefore two results
        # of which one of them is an empty list.
        assert len(extra_batches) == self.test_epochs - 1
        assert extra_batches[0] == expected_e0
        assert extra_batches[1] == expected_e1

    def test_history_partial_batches_batch_key_4th(self, history, ref):
        extra_batches = history[:, 'batches', :, 'extra_batch']

        expected_e0 = [b['extra_batch'] for b in ref[0]['batches']
                       if 'extra_batch' in b]
        expected_e1 = [b['extra_batch'] for b in ref[1]['batches']
                       if 'extra_batch' in b]

        # In every epoch there are 2 batches with the 'extra_batch'
        # key except for the last epoch. We therefore two results
        # of which one of them is an empty list.
        assert len(extra_batches) == self.test_epochs - 1
        assert extra_batches[0] == expected_e0
        assert extra_batches[1] == expected_e1

    def test_history_partial_singular_values(self, history):
        values = history[-1, ('duration', 'total_loss')]
        expected = (history[-1]['duration'], history[-1]['total_loss'])

        # pylint: disable=unidiomatic-typecheck
        assert type(values) == tuple
        assert values == expected

    def test_history_slice_beyond_batches_but_key_not_batches(self, history):
        with pytest.raises(KeyError) as exc:
            # pylint: disable=pointless-statement
            history[:, 'not-batches', 0]

        msg = exc.value.args[0]
        expected = ("History indexing beyond the 2nd level is "
                    "only possible if key 'batches' is used, "
                    "found key 'not-batches'.")
        assert msg == expected

    def test_history_with_invalid_epoch_key(self, history):
        key = slice(None), 'not-batches'
        with pytest.raises(KeyError) as exc:
            # pylint: disable=pointless-statement
            history[key]

        msg = exc.value.args[0]
        expected = "Key 'not-batches' was not found in history."
        assert msg == expected

    def test_history_too_many_indices(self, history):
        with pytest.raises(KeyError) as exc:
            # pylint: disable=pointless-statement
            history[:, 'batches', :, 'train_loss', :]

        msg = exc.value.args[0]
        expected = ("Tried to index history with 5 indices but only "
                    "4 indices are possible.")
        assert msg == expected

    def test_history_save_load_cycle_file_obj(self, history_cls, history, tmpdir):
        if hasattr(history, 'store'):
            # DistributedHistory does not support loading from file
            pytest.skip()

        history_f = tmpdir.mkdir('skorch').join('history.json')

        with open(str(history_f), 'w') as f:
            history.to_file(f)

        with open(str(history_f), 'r') as f:
            new_history = history_cls.from_file(f)

        assert history == new_history

    def test_history_save_load_cycle_file_path(self, history_cls, history, tmpdir):
        if hasattr(history, 'store'):
            # DistributedHistory does not support loading from file
            pytest.skip()

        history_f = tmpdir.mkdir('skorch').join('history.json')

        history.to_file(str(history_f))
        new_history = history_cls.from_file(str(history_f))

        assert history == new_history

    @pytest.mark.parametrize('type_', [list, tuple])
    def test_history_multiple_keys(self, history, type_):
        dur_loss = history[-1, type_(['duration', 'total_loss'])]
        # pylint: disable=unidiomatic-typecheck
        assert type(dur_loss) is type_ and len(dur_loss) == 2

        loss_loss = history[-1, 'batches', -1, type_(['loss', 'loss'])]
        # pylint: disable=unidiomatic-typecheck
        assert type(loss_loss) is type_ and len(loss_loss) == 2

    def test_history_key_in_other_epoch(self, history_cls):
        h = history_cls()
        for has_valid in (True, False):
            h.new_epoch()
            h.new_batch()
            h.record_batch('train_loss', 1)
            if has_valid:
                h.new_batch()
                h.record_batch('valid_loss', 2)

        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            h[-1, 'batches', -1, 'valid_loss']

    def test_history_no_epochs_index(self, history_cls):
        h = history_cls()
        with pytest.raises(IndexError):
            # pylint: disable=pointless-statement
            h[-1, 'batches']

    def test_history_jagged_batches(self, history_cls):
        h = history_cls()
        for num_batch in (1, 2):
            h.new_epoch()
            for _ in range(num_batch):
                h.new_batch()
        # Make sure we can access this batch
        assert h[-1, 'batches', 1] == {}

    @pytest.mark.parametrize('value, check_warn', [
        ([], False),
        (np.array([]), True),
    ])
    def test_history_retrieve_empty_list(self, value, history_cls, check_warn, recwarn):
        h = history_cls()
        if hasattr(h, 'store') and isinstance(value, np.ndarray):
            # DistributedHistory does not support numpy arrays because they
            # cannot be json serialized
            pytest.skip()

        h.new_epoch()
        h.record('foo', value)
        h.new_batch()
        h.record_batch('batch_foo', value)

        # Make sure we can access our object
        out = h[-1, 'foo']
        assert (out is value) or (out == value)
        out = h[-1, 'batches', -1, 'batch_foo']
        assert (out is value) or (out == value)

        # There should be no warning about comparison to an empty ndarray
        if check_warn:
            assert not recwarn.list

    @pytest.mark.parametrize('has_epoch, epoch_slice', [
        (False, slice(None)),
        (True, slice(1, None)),
    ])
    def test_history_no_epochs_key(self, has_epoch, epoch_slice, history_cls):
        h = history_cls()
        if has_epoch:
            h.new_epoch()

        # Expect KeyError since the key was not found in any epochs
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            h[epoch_slice, 'foo']
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            h[epoch_slice, ['foo', 'bar']]

    @pytest.mark.parametrize('has_batch, batch_slice', [
        (False, slice(None)),
        (True, slice(1, None)),
    ])
    def test_history_no_batches_key(self, has_batch, batch_slice, history_cls):
        h = history_cls()
        h.new_epoch()
        if has_batch:
            h.new_batch()

        # Expect KeyError since the key was not found in any batches
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            h[-1, 'batches', batch_slice, 'foo']
        with pytest.raises(KeyError):
            # pylint: disable=pointless-statement
            h[-1, 'batches', batch_slice, ['foo', 'bar']]

    @pytest.mark.parametrize('has_epoch, epoch_slice', [
        (False, slice(None)),
        (True, slice(1, None)),
    ])
    def test_history_no_epochs_batches(self, has_epoch, epoch_slice, history_cls):
        h = history_cls()
        if has_epoch:
            h.new_epoch()

        # Expect a list of zero epochs since 'batches' always exists
        assert h[epoch_slice, 'batches'] == []
        assert h[epoch_slice, 'batches', -1] == []

    def test_pickle(self, history):
        loaded = pickle.loads(pickle.dumps(history))

        # The store cannot be pickled, so it is set to None
        if not hasattr(history, 'store'):
            assert history.to_list() == loaded.to_list()
        else:
            assert loaded.store is None


class TestDistributedHistoryMultiprocessing:
    """Testing DistributedHistory with multiprocessing

    This is not a proper test for how DistributedHistory should be used in the
    wild, which would be in a multi-GPU setting using DDP (possibly through
    accelerate). However, since we cannot test this setting in CI, this is the
    next best approximation.

    Since DDP creates forks of the main process, each one would have its own
    history instance, leading in them diverging. This can be bad, e.g. if the
    history is used to steer the training (think early stopping). This test
    intends to show that the histories do not diverge.

    In a sense, this test actually tests incorrect behavior because the
    different processes all write the exact same content to the history.
    However, if DDP was used, each process would see different batches and thus
    write different records to their histories. So when reading this test,
    imagine that this would be what's truly happening.

    This test is very high level, unlike TestHistory. For instance, there is no
    attempt at testing "weird" read and write patterns, like creating multiple
    empty batches. It is very much assumed that the history is used "correctly",
    as in a normal skorch fit loop.

    """
    @pytest.fixture
    def data(self, classifier_data):
        X, y = classifier_data
        return X[:100], y[:100]

    @staticmethod
    def train(args):
        from skorch import NeuralNetClassifier
        from skorch.history import DistributedHistory

        time.sleep(0.5)  # wait a bit, else broken pipe error may occur

        torch.manual_seed(0)
        rank, nprocs, module, (X, y) = args

        # let's hope the port is free
        store = TCPStore(
            '127.0.0.1', 8890 + nprocs, world_size=nprocs, is_master=(rank == 0)
        )
        dist_history = DistributedHistory(store=store, rank=rank, world_size=nprocs)
        net = NeuralNetClassifier(module, max_epochs=2, history=dist_history)
        net.history = dist_history
        net.fit(X, y)

        time.sleep(0.5)  # wait a bit, else broken pipe error may occur

        return net

    @pytest.mark.parametrize('nprocs', [1, 2, 3, 4])
    def test_distributed_history(self, nprocs):
        from skorch.toy import make_classifier
        from skorch._version import Version

        X, y = make_classification(
            500, 20, n_informative=10, random_state=0, flip_y=0.1
        )
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        torch.set_num_threads(1)
        module = make_classifier(
            input_units=20,
            hidden_units=10,
            num_hidden=2,
            dropout=0.5,
        )

        args = zip(
            range(nprocs), [nprocs] * nprocs, [module] * nprocs, [(X, y)] * nprocs
        )
        # https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        ctx = mp.get_context('spawn')
        with ctx.Pool(nprocs) as pool:
            nets = pool.map(self.train, args)

        # expected entries for training batches:
        # - 500 samples, 400 used for train, 100 for valid
        # - batch size is 128, i.e. 4 batches per epoch for train
        # - the 100 valid samples fit into 1 batch
        # note that when using DDP correctly, since batches are split across the
        # processes, there would *not* be nprocs times more entries.
        train_batches = 4
        valid_batches = 1
        for net in nets:
            h = net.history
            assert len(h[0, 'batches', :, 'train_loss']) == train_batches * nprocs
            assert len(h[1, 'batches', :, 'train_loss']) == train_batches * nprocs
            assert len(h[0, 'batches', :, 'valid_loss']) == valid_batches * nprocs
            assert len(h[1, 'batches', :, 'valid_loss']) == valid_batches * nprocs

        # the batch information is synchronized, so should be identical; the
        # epoch information is *not* synchronized and can diverge, most notably
        # with things like the duration
        for net0, net1 in zip(nets, nets[1:]):
            train_loss0 = net0.history[:, 'batches', :, 'train_loss']
            train_loss1 = net1.history[:, 'batches', :, 'train_loss']
            assert train_loss0 == train_loss1

            valid_loss0 = net0.history[:, 'batches', :, 'valid_loss']
            valid_loss1 = net1.history[:, 'batches', :, 'valid_loss']
            assert valid_loss0 == valid_loss1

            assert net0.history[:, 'dur'] != net1.history[:, 'dur']
