import argparse

import torch
from torch.autograd import Variable

import data
import model
import learner

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')

args = parser.parse_args()

# TODO: set seed

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

learner = learner.Learner(
    module=model.RNNModel,
    batch_size=1,
    use_cuda=args.cuda,
    module__rnn_type='LSTM',
    module__ntoken=ntokens,
    module__ninp=200,
    module__nhid=200,
    module__nlayers=2,
    iterator_test__evaluation=True,
    iterator_test__use_cuda=args.cuda,
    iterator_test__bptt=args.bptt)

learner.initialize()

if not args.cuda:
    learner.module_ = torch.load(args.checkpoint, map_location=lambda storage, location: 'cpu')
else:
    learner.module_ = torch.load(args.checkpoint)

hidden = learner.module_.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input = input.cuda()

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden = learner.module_(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
