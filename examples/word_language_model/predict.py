import argparse

import numpy as np
import torch
from torch.autograd import Variable

import data
import model
import net

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')

args = parser.parse_args()

torch.manual_seed(args.seed)

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

net = net.Net(
    module=model.RNNModel,
    batch_size=1,
    use_cuda=args.cuda,
    module__rnn_type='LSTM',
    module__ntoken=ntokens,
    module__ninp=200,
    module__nhid=200,
    module__nlayers=2)
net.initialize()
net.load_params(args.checkpoint)


input_data = np.array([[
    corpus.dictionary.word2idx['fish'],
    corpus.dictionary.word2idx['sees'],
    corpus.dictionary.word2idx['man'],
]])

# Convert to batch last, i.e. (b, t) -> (t, b).
input_data = input_data.T

p = net.predict_proba(input_data)
print(p)

widx = net.predict(input_data)
print(widx, [corpus.dictionary.idx2word[n] for n in widx])
