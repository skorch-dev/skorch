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
    def net_cls(self):
        from skorch import NeuralNetClassifier
        return NeuralNetClassifier

    @pytest.fixture(scope='module')
    def doctor_cls(self):
        from skorch.helper import SkorchDoctor
        return SkorchDoctor

    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope='module')
    def doctor(self, module_cls, net_cls, doctor_cls, data):
        net = net_cls(module_cls, max_epochs=3, batch_size=32)
        doctor = doctor_cls(net)
        X, y = data
        doctor.fit(X[:50], y[:50])
        return doctor

    def test_activation_logs(self, doctor):
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

        # 80 of 50 samples is 40, batch size 32 => 32 + 8 samples per batch
        batch_sizes = [[len(b['lin1']) for b in batch] for batch in lm]
        assert batch_sizes == [[32, 8], [32, 8], [32, 8]]
