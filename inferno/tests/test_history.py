import pytest

from inferno.net import History


class TestHistory:

    test_epochs = 3
    test_batches = 4

    @pytest.fixture
    def history(self):
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
                if num_batch % 2 == 0:
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

    def test_history_retrieve_single_value(self, history, ref):
        total_loss_0 = history[0, 'total_loss']
        assert total_loss_0 == ref[0]['total_loss']

    def test_history_retrieve_multiple_values(self, history, ref):
        total_loss_0_to_1 = history[0:1, 'total_loss']
        assert total_loss_0_to_1 == [n['total_loss'] for n in ref[0:1]]

    def test_history_non_existing_values(self, history):
        with pytest.raises(KeyError):
            history[:, 'non-existing']
        with pytest.raises(KeyError):
            history[0, 'extra']

    def test_history_non_existing_values_batch(self, history):
        with pytest.raises(KeyError):
            history[:, 'batches', :, 'non-existing']
        with pytest.raises(KeyError):
            history[:, 'batches', 1, 'extra_batch']

    def test_history_mixed_slicing(self, history, ref):
        losses = history[:, 'batches', 0, 'loss']

        assert len(losses) == self.test_epochs
        assert losses == [epoch['batches'][0]['loss'] for epoch in ref]

        losses = history[0, 'batches', :, 'loss']
        assert losses == [batch['loss'] for batch in ref[0]['batches']]

    def test_history_partial_and_full_index_batches(self, history, ref):
        loss_with_extra = history[:, 'batches', :, ('loss', 'extra_batch')]

        expected_e0 = [(b['loss'], b['extra_batch']) for b in ref[0]['batches'] if 'extra_batch' in b]
        expected_e2 = [(b['loss'], b['extra_batch']) for b in ref[2]['batches'] if 'extra_batch' in b]

        assert len(loss_with_extra) == self.test_epochs
        assert loss_with_extra[0] == expected_e0
        assert loss_with_extra[2] == expected_e2

