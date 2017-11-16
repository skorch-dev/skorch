import skorch
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score


class Net(skorch.NeuralNet):

    def __init__(
            self,
            criterion=torch.nn.CrossEntropyLoss,
            clip=0.25,
            lr=20,
            ntokens=10000,
            *args,
            **kwargs
    ):
        self.clip = clip
        self.ntokens = ntokens
        super(Net, self).__init__(criterion=criterion, lr=lr, *args, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, Variable):
            v = Variable(h.data)
            return v.cuda() if self.use_cuda else v
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

        torch.nn.utils.clip_grad_norm(self.module_.parameters(), self.clip)
        for p in self.module_.parameters():
            p.data.add_(-self.lr, p.grad.data)
        return loss

    def validation_step(self, X, y):
        self.module_.eval()

        hidden = self.module_.init_hidden(self.batch_size)
        output, _ = self.module_(X, hidden)
        output_flat = output.view(-1, self.ntokens)

        return self.get_loss(output_flat, y)

    def evaluation_step(self, X, **kwargs):
        self.module_.eval()

        X = skorch.utils.to_var(X, use_cuda=self.use_cuda)
        hidden = self.module_.init_hidden(self.batch_size)
        output, _ = self.module_(X, hidden)

        return output.view(-1, self.ntokens)

    def sample(self, input, temperature=1., hidden=None):
        hidden = self.module_.init_hidden(1) if hidden is None else hidden
        output, hidden = self.module_(input, hidden)
        probas = output.squeeze().data.div(temperature).exp()
        sample = torch.multinomial(probas, 1)[-1]
        if probas.dim() > 1:
            sample = sample[0]
        return sample, self.repackage_hidden(hidden)

    def sample_n(self, num_words, input, temperature=1., hidden=None):
        preds = [None] * num_words
        for i in range(num_words):
            preds[i], hidden = self.sample(input, hidden=hidden)
            input = skorch.utils.to_var(torch.LongTensor([[preds[i]]]),
                                        use_cuda=self.use_cuda)
        return preds, hidden

    def score(self, X, y=None):
        y_probas = []
        y_target = []

        # We collect the predictions batch-wise and store them on the host
        # side as this data can be quite big and the GPU might run into
        # memory issues. We do not calculate F1 on the batches as this
        # would introduce an error to the score.

        ds = self.get_dataset(X)
        target_iterator = self.get_iterator(ds, train=False)
        pred_iterator = self.forward_iter(X)

        for (_, y_true), y_pred in zip(target_iterator, pred_iterator):
            y_pred_cls = skorch.utils.to_numpy(y_pred).argmax(-1)

            y_probas.append(y_pred_cls)
            y_target.append(skorch.utils.to_numpy(y_true))

        y_probas = np.concatenate(y_probas)
        y_target = np.concatenate(y_target)

        return f1_score(y_probas, y_target, average='micro')
