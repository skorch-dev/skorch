import inferno
from sklearn.model_selection import GridSearchCV

from model import RNNModel
import data
import trainer

class LRAnnealing(inferno.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1]['valid_loss_best']:
            net.lr /= 4.0

corpus = data.Corpus('./data/penn')
ntokens = len(corpus.dictionary)
bptt = 10
batch_size = 20
use_cuda = True

trainer = trainer.Trainer(
    module=RNNModel,
    iterator_train=data.Loader,
    iterator_test=data.Loader,
    batch_size=batch_size,
    use_cuda=use_cuda,
    callbacks=[LRAnnealing()],
    module__rnn_type='LSTM',
    module__ntoken=ntokens,
    module__ninp=200,
    module__nhid=200,
    module__nlayers=2,
    iterator_test__evaluation=True,
    iterator_train__use_cuda=use_cuda,
    iterator_test__use_cuda=use_cuda)

params = [
    {
        'lr': [20],
        'iterator_train__bptt': [5, 10],
    },
]

pl = GridSearchCV(trainer, params)
pl.fit(corpus.train[:1000], corpus.train[:1000])
