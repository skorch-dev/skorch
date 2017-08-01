import argparse

import inferno
import torch
from sklearn.model_selection import GridSearchCV

import data
from model import RNNModel
from learner import Learner

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# TODO: set seed

class LRAnnealing(inferno.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1]['valid_loss_best']:
            net.lr /= 4.0

class Checkpointing(inferno.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if net.history[-1]['valid_loss_best']:
            with open(args.save, 'wb') as f:
                torch.save(net.module_, f)

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

learner = Learner(
    module=RNNModel,
    iterator_train=data.Loader,
    iterator_test=data.Loader,
    batch_size=args.batch_size,
    use_cuda=args.cuda,
    callbacks=[LRAnnealing(), Checkpointing()],
    module__rnn_type='LSTM',
    module__ntoken=ntokens,
    module__ninp=200,
    module__nhid=200,
    module__nlayers=2,
    iterator_train__use_cuda=args.cuda,
    iterator_train__bptt=args.bptt,
    iterator_test__evaluation=True,
    iterator_test__use_cuda=args.cuda,
    iterator_test__bptt=args.bptt)

# FIXME: iterator_test does not use corpus.valid as dataset

params = [
    {
        'lr': [10,20,30],
    },
]

pl = GridSearchCV(learner, params)
pl.fit(corpus.train[:1000])

print("Results of grid search:")
print("Best parameter configuration:", pl.best_params_)
print("Achieved score:", pl.best_score_)
