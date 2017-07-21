import inferno
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
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def on_train_begin(self, net, *args, **kwargs):
        self.hidden = self.module_.init_hidden(self.batch_size)
        #if self.use_cuda:
        #    [n.cuda() for n in self.hidden]

    def train_step(self, X, y, _):
        self.module_.train()

        self.hidden = self.repackage_hidden(self.hidden)

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

    def score(self, X, y):
        import pdb; pdb.set_trace()

corpus = data.Corpus('./data/penn')
ntokens = len(corpus.dictionary)
bptt = 10
batch_size = 20
use_cuda = False # FIXME

class Loader:
    def __init__(self, source, bptt=10, batch_size=20, evaluation=False):
        self.batches = self.batchify(source.data_tensor, batch_size)
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
        module__rnn_type='LSTM',
        module__ntoken=ntokens,
        module__ninp=200,
        module__nhid=200,
        module__nlayers=2,
        iterator_test__evaluation=True)

params = [
    {
        'lr': [0.01,],
        'iterator_train__bptt': [5, 10],
    },
]

pl = GridSearchCV(trainer, params)
pl.fit(corpus.train[:1000], corpus.train[:1000])

