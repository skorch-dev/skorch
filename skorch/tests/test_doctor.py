import numpy as np
import pytest
import torch
from torch import nn


class TestSkorchDoctor:
    @pytest.fixture(scope='module')
    def module_cls(self):
        class MyModule(nn.Module):
            """Module with predictable parameters"""
            def __init__(self):
                super().__init__()

                self.lin0 = nn.Linear(20, 20)
                nn.init.eye_(self.lin0.weight)
                nn.init.zeros_(self.lin0.bias)

                self.lin1 = nn.Linear(20, 2)
                nn.init.zeros_(self.lin1.weight)
                nn.init.ones_(self.lin1.bias)

                self.softmax = nn.Softmax(dim=-1)

            def forward(self, X):
                X = self.lin0(X)
                X = self.lin1(X)
                return self.softmax(X)

        return MyModule

    @pytest.fixture(scope='module')
    def custom_split(self):
        class Split():
            """Deterministically split train and valid into 80%/20%"""
            def __call__(self, dataset, y=None, groups=None):
                n = int(len(dataset) * 0.8)
                dataset_train = torch.utils.data.Subset(dataset, np.arange(n))
                dataset_valid = torch.utils.data.Subset(dataset, np.arange(n, len(dataset)))
                return dataset_train, dataset_valid

        return Split()

    @pytest.fixture(scope='module')
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def doctor_cls(self):
        from skorch.helper import SkorchDoctor
        return SkorchDoctor

    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        X, y = classifier_data
        # a small amount of data is enough
        return X[:50], y[:50]

    @pytest.fixture(scope='module')
    def doctor(self, module_cls, net_cls, doctor_cls, data, custom_split):
        net = net_cls(module_cls, max_epochs=3, batch_size=32, train_split=custom_split)
        doctor = doctor_cls(net)
        doctor.fit(*data)
        return doctor

    def test_activation_logs_general_content(self, doctor):
        logs = doctor.activation_logs_
        assert len(logs) == 2
        assert set(logs.keys()) == {'module', 'criterion'}

        # nothing locked for criterion, 3 epochs
        assert logs['criterion'] == [[], [], []]

        lm = logs['module']
        # 3 epochs, 2 batches per epoch
        assert len(lm) == 3
        assert [len(batch) for batch in lm] == [2, 2, 2]

        # each batch has layers lin0, lin1, softmax
        for epoch in lm:
            for batch in epoch:
                assert set(batch.keys()) == {'lin0', 'lin1', 'softmax'}

    def test_activation_logs_values(self, doctor, data):
         # 80% of 50 samples is 40, batch size 32 => 32 + 8 samples per batch
        batch_sizes = [[len(b['lin1']) for b in batch] for batch in lm]
        assert batch_sizes == [[32, 8], [32, 8], [32, 8]]

        X, _ = data
        # for the very first batch, before any update, we actually know the values
        batch = lm[0][0]
        lin0_0 = batch['lin0']
        # since it is the identity function, batches should equal the data
        np.testing.assert_array_almost_equal(lin0_0, X[:32])

        lin1_0 = batch['lin1']
        # since weights are 0 and bias is 1, all values should be 1
        np.testing.assert_array_almost_equal(lin1_0, 1.0)

        sm_0 = batch['softmax']
        # since all inputs are equal, probabilities should be uniform
        np.testing.assert_array_almost_equal(sm_0, 0.5)
