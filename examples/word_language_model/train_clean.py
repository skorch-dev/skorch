import inferno
from sklearn.model_selection import GridSearchCV

from .model import RNNModel

class Trainer(inferno.NeuralNet):

    def __init__(self, *args, **kwargs):
        criterion = torch.nn.CrossEntropyLoss()

        super(Net, self).__init__(criterion=criterion, *args, **kwargs)

    def train_step(self, X, y):
        self.model_.train()



            loss = self.get_loss(y_pred, targets)

corpus = data.Corpus()

batch_size = 20

train_loader = TensorDataset(
        data_tensor=corpus.train[:batch_size],
        target_tensor=corpus.train[batch_size:]
)


model = RNNModel()
trainer = Trainer(model)

params = [
    {'lr': 0.01,},
    {'lr': 0.001 },
]

GridSearchCV(trainer, params)


