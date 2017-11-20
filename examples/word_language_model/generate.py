import argparse

import skorch
import torch
from torch.autograd import Variable

import data
import model
import net

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

hidden = None
input = skorch.utils.to_var(torch.rand(1, 1).mul(ntokens).long(),
                            use_cuda=args.cuda)

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        word_idx, hidden = net.sample(
                input=input,
                temperature=args.temperature,
                hidden=hidden)
        input = skorch.utils.to_var(
                torch.LongTensor([[word_idx]]),
                use_cuda=args.cuda)

        word = corpus.dictionary.idx2word[word_idx]
        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
