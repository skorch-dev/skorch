import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.autograd import Variable

import skorch


class Net(skorch.NeuralNet):

    def __init__(
        self, criterion=torch.nn.CrossEntropyLoss, clip=0.25, lr=20, ntokens=10000, *args, **kwargs
    ):
        self.clip = clip
        self.ntokens = ntokens
        super(Net, self).__init__(criterion=criterion, lr=lr, *args, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, Variable):
            return torch.tensor(h.data, device=h.device)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def on_epoch_begin(self, *args, **kwargs):
        super().on_epoch_begin(*args, **kwargs)

        # As an optimization to save tensor allocation for each
        # batch we initialize the hidden state only once per epoch.
        # This optimization was taken from the original example.
        self.hidden = self.module_.init_hidden(self.batch_size)

    def train_step(self, X, y):
        self.module_.train()

        # Repackage shared hidden state so that the previous batch
        # does not influence the current one.
        self.hidden = self.repackage_hidden(self.hidden)
        self.module_.zero_grad()

        output, self.hidden = self.module_(X, self.hidden)
        y_pred = output.view(-1, self.ntokens)

        loss = self.get_loss(y_pred, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.module_.parameters(), self.clip)
        for p in self.module_.parameters():
            p.data.add_(-self.lr, p.grad.data)
        return {'loss': loss, 'y_pred': y_pred}

    def validation_step(self, X, y):
        self.module_.eval()

        hidden = self.module_.init_hidden(self.batch_size)
        output, _ = self.module_(X, hidden)
        output_flat = output.view(-1, self.ntokens)

        return {'loss': self.get_loss(output_flat, y), 'y_pred': output_flat}

    def evaluation_step(self, X, **kwargs):
        self.module_.eval()

        X = skorch.utils.to_tensor(X, device=self.device)
        hidden = self.module_.init_hidden(self.batch_size)
        output, _ = self.module_(X, hidden)

        return output.view(-1, self.ntokens)

    def predict(self, X):
        return np.argmax(super().predict(X), -1)

    def sample(self, input, temperature=1.0, hidden=None):
        hidden = self.module_.init_hidden(1) if hidden is None else hidden
        output, hidden = self.module_(input, hidden)
        probas = output.squeeze().data.div(temperature).exp()
        sample = torch.multinomial(probas, 1)[-1]
        if probas.dim() > 1:
            sample = sample[0]
        return sample, self.repackage_hidden(hidden)

    def sample_n(self, num_words, input, temperature=1.0, hidden=None):
        preds = [None] * num_words
        for i in range(num_words):
            preds[i], hidden = self.sample(input, hidden=hidden)
            input = skorch.utils.to_tensor(torch.LongTensor([[preds[i]]]), device=self.device)
        return preds, hidden

    def score(self, X, y=None):
        ds = self.get_dataset(X)
        target_iterator = self.get_iterator(ds, training=False)

        y_true = np.concatenate([skorch.utils.to_numpy(y) for _, y in target_iterator])
        y_pred = self.predict(X)

        return f1_score(y_true, y_pred, average='micro')
