"""Tests for history.py."""

import numpy as np
import pytest

from skorch.history import History


class TestHistory:

    test_epochs = 3
    test_batches = 4

    @pytest.fixture
    def history(self):
        """Return a history filled with epoch and batch data."""
        h = History()
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

    def test_list_initialization(self):
        h = History([1, 2, 3])
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

    def test_history_save_load_cycle_file_obj(self, history, tmpdir):
        history_f = tmpdir.mkdir('skorch').join('history.json')

        with open(str(history_f), 'w') as f:
            history.to_file(f)

        with open(str(history_f), 'r') as f:
            new_history = History.from_file(f)

        assert history == new_history

    def test_history_save_load_cycle_file_path(self, history, tmpdir):
        history_f = tmpdir.mkdir('skorch').join('history.json')

        history.to_file(str(history_f))
        new_history = History.from_file(str(history_f))

        assert history == new_history

    @pytest.mark.parametrize('type_', [list, tuple])
    def test_history_multiple_keys(self, history, type_):
        dur_loss = history[-1, type_(['duration', 'total_loss'])]
        # pylint: disable=unidiomatic-typecheck
        assert type(dur_loss) is type_ and len(dur_loss) == 2

        loss_loss = history[-1, 'batches', -1, type_(['loss', 'loss'])]
        # pylint: disable=unidiomatic-typecheck
        assert type(loss_loss) is type_ and len(loss_loss) == 2

    def test_history_key_in_other_epoch(self):
        h = History()
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

    def test_history_no_epochs_index(self):
        h = History()
        with pytest.raises(IndexError):
            # pylint: disable=pointless-statement
            h[-1, 'batches']

    def test_history_jagged_batches(self):
        h = History()
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
    def test_history_retrieve_empty_list(self, value, check_warn, recwarn):
        h = History()
        h.new_epoch()
        h.record('foo', value)
        h.new_batch()
        h.record_batch('batch_foo', value)

        # Make sure we can access our object
        assert h[-1, 'foo'] is value
        assert h[-1, 'batches', -1, 'batch_foo'] is value

        # There should be no warning about comparison to an empty ndarray
        if check_warn:
            assert not recwarn.list

    @pytest.mark.parametrize('has_epoch, epoch_slice', [
        (False, slice(None)),
        (True, slice(1, None)),
    ])
    def test_history_no_epochs_key(self, has_epoch, epoch_slice):
        h = History()
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
    def test_history_no_batches_key(self, has_batch, batch_slice):
        h = History()
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
    def test_history_no_epochs_batches(self, has_epoch, epoch_slice):
        h = History()
        if has_epoch:
            h.new_epoch()

        # Expect a list of zero epochs since 'batches' always exists
        assert h[epoch_slice, 'batches'] == []
        assert h[epoch_slice, 'batches', -1] == []
