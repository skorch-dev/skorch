import inferno
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score


class Learner(inferno.NeuralNet):

    def __init__(self,
                 criterion=torch.nn.CrossEntropyLoss,
                 clip=0.25,
                 lr=20,
                 ntokens=10000,
                 *args, **kwargs):
        self.clip = clip
        self.ntokens = ntokens
        super(Learner, self).__init__(criterion=criterion, lr=lr, *args, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, Variable):
            v = Variable(h.data)
            return v.cuda() if self.use_cuda else v
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def on_train_begin(self, net, *args, **kwargs):
        self.hidden = self.module_.init_hidden(self.batch_size)
        if self.use_cuda:
            self.module_.cuda()

    def train_step(self, X, y, _):
        self.module_.train()

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

        output, self.hidden = self.module_(X, self.hidden)
        output_flat = output.view(-1, self.ntokens)

        return self.get_loss(output_flat, y)

    def evaluation_step(self, X, **kwargs):
        self.module_.eval()

        hidden = self.module_.init_hidden(X.size(1))
        output, hidden = self.module_(X, hidden)

        word_weights = output.view(-1, self.ntokens)
        word_idx = torch.multinomial(word_weights, 1)

        return word_idx

    def score(self, X, y=None):
        # TODO: we cannot use predict() directly as the y supplied by GridSearchCV
        # is not a "valid" y and only based on the input given to fit() down below.
        # Therefore we have to generate our own batches.
        #pred = self.predict(X)
        #return f1_score(y, pred)

        iterator = self.get_iterator(X, y, train=False)
        y_probas = []
        y_target = []
        for x, y in iterator:
            y_probas.append(self.evaluation_step(x))
            y_target.append(y)
        y_probas = inferno.utils.to_numpy(torch.cat(y_probas, dim=0).squeeze())
        y_target = inferno.utils.to_numpy(torch.cat(y_target, dim=0))
        return f1_score(y_probas, y_target, average='micro')
