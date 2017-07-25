import inferno
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import torch
from torch.autograd import Variable

from model import RNNModel
import data

class Trainer(inferno.NeuralNet):

    def __init__(self,
            criterion=torch.nn.CrossEntropyLoss,
            clip=0.25,
            lr=20,
            *args, **kwargs):
        self.clip = clip
        super(Trainer, self).__init__(criterion=criterion, lr=lr, *args, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            v = Variable(h.data)
            return v.cuda() if self.use_cuda else v
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def on_train_begin(self, net, *args, **kwargs):
        self.hidden = self.module_.init_hidden(self.batch_size)
        self.module_.cuda()

    def train_step(self, X, y, _):
        self.module_.train()

        self.hidden = self.repackage_hidden(self.hidden)
        self.module_.zero_grad()

        output, self.hidden = self.module_(X, self.hidden)
        y_pred = output.view(-1, ntokens)

        loss = self.get_loss(y_pred, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm(self.module_.parameters(), self.clip)
        for p in self.module_.parameters():
            p.data.add_(-self.lr, p.grad.data)

        return loss

    def validation_step(self, X, y):
        self.module_.eval()

        output, self.hidden = self.module_(X, self.hidden)
        output_flat = output.view(-1, ntokens)

        return self.get_loss(output_flat, y)

    def evaluation_step(self, X):
        self.module_.eval()

        hidden = self.module_.init_hidden(X.size(1))
        output, hidden = self.module_(X, hidden)

        word_weights = output.view(-1, ntokens)
        word_idx = torch.multinomial(word_weights, 1)

        return word_idx

    def forward(self, X, training_behavior=False):
        self.module_.train(training_behavior)

        iterator = self.get_iterator(X, train=training_behavior)
        y_probas = []
        for x in iterator:
            x = inferno.utils.to_var(x, use_cuda=self.use_cuda)
            y_probas.append(self.evaluation_step(x))
        return torch.cat(y_probas, dim=0)

    def score(self, X, y):
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



class LRAnnealing(inferno.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1]['valid_loss_best']:
            net.lr /= 4.0


corpus = data.Corpus('./data/penn')
ntokens = len(corpus.dictionary)
bptt = 10
batch_size = 20
use_cuda = True

class Loader:
    def __init__(self, source, bptt=10, batch_size=20, evaluation=False):
        # FIXME: this is kind of stupid, we supply TensorDatasets to the loader
        # except in forward (=> therefore in predict()) we don't (we just
        # supply it with what we get).
        if type(source) == torch.utils.data.TensorDataset:
            source = source.data_tensor
            self.prediction = False
        else:
            self.prediction = True

        self.batches = self.batchify(source, batch_size)
        self.evaluation = evaluation
        self.bptt = bptt
        self.batch_size = batch_size

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if use_cuda:
            data = data.cuda()
        return data

    def get_batch(self, i):
        seq_len = min(self.bptt, len(self.batches) - 1 - i)
        data = Variable(self.batches[i:i+seq_len], volatile=self.evaluation)

        if self.prediction:
            return data
        else:
            target = Variable(self.batches[i+1:i+1+seq_len].view(-1))
            return data, target

    def __iter__(self):
        for batch, i in enumerate(range(0, self.batches.size(0) - 1, self.bptt)):
            yield self.get_batch(i)


trainer = Trainer(
        module=RNNModel,
        iterator_train=Loader,
        iterator_test=Loader,
        batch_size=batch_size,
        use_cuda=use_cuda,
        callbacks=[LRAnnealing()],
        module__rnn_type='LSTM',
        module__ntoken=ntokens,
        module__ninp=200,
        module__nhid=200,
        module__nlayers=2,
        iterator_test__evaluation=True)

params = [
    {
        'lr': [20],
        'iterator_train__bptt': [5, 10],
    },
]

pl = GridSearchCV(trainer, params)
pl.fit(corpus.train[:1000], corpus.train[:1000])

