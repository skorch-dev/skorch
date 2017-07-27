import os

import torch
from torch.autograd import Variable


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class Loader:
    def __init__(self, source, use_cuda=False, bptt=10, batch_size=20, evaluation=False):
        # FIXME: this is kind of stupid, we supply TensorDatasets to the loader
        # except in forward (=> therefore in predict()) we don't (we just
        # supply it with what we get).
        if isinstance(source, torch.utils.data.TensorDataset):
            source = source.data_tensor
            self.prediction = False
        else:
            self.prediction = True

        self.evaluation = evaluation
        self.bptt = bptt
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.batches = self.batchify(source, batch_size)

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if self.use_cuda:
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
        for i in range(0, self.batches.size(0) - 1, self.bptt):
            yield self.get_batch(i)


