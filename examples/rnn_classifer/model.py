import os
from urllib.request import urlretrieve
import tarfile

from dstoolbox.transformers import Padder2d
from dstoolbox.transformers import TextFeaturizer
import numpy as np
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
import torch
from torch import nn
from palladium.interfaces import DatasetLoader as IDatasetLoader

F = nn.functional

DATA_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
DATA_FN = DATA_URL.rsplit('/', 1)[1]

np.random.seed(0)


def download():
    if not os.path.exists('aclImdb'):
        # unzip data if it does not exist
        if not os.path.exists(DATA_FN):
            urlretrieve(DATA_URL, DATA_FN)
        with tarfile.open(DATA_FN, 'r:gz') as f:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                # See:
                # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
                # on why plain tar.extractall can be dangerous
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise OSError("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
            
            safe_extract(f)


class DatasetLoader(IDatasetLoader):
    def __init__(self, path='aclImdb/train/'):
        self.path = path

    def __call__(self):
        download()
        dataset = load_files(self.path, categories=['pos', 'neg'])
        X, y = dataset['data'], dataset['target']
        X = np.asarray([x.decode() for x in X])  # decode from bytes
        return X, y


class RNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        rec_layer_type='lstm',
        num_units=128,
        num_layers=2,
        dropout=0,
        vocab_size=1000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rec_layer_type = rec_layer_type.lower()
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout

        self.emb = nn.Embedding(
            vocab_size + 1, embedding_dim=self.embedding_dim)

        rec_layer = {'lstm': nn.LSTM, 'gru': nn.GRU}[self.rec_layer_type]
        # We have to make sure that the recurrent layer is batch_first,
        # since sklearn assumes the batch dimension to be the first
        self.rec = rec_layer(
            self.embedding_dim, self.num_units,
            num_layers=num_layers, batch_first=True,
            )

        self.output = nn.Linear(self.num_units, 2)

    def forward(self, X):
        embeddings = self.emb(X)
        # from the recurrent layer, only take the activities from the
        # last sequence step
        if self.rec_layer_type == 'gru':
            _, rec_out = self.rec(embeddings)
        else:
            _, (rec_out, _) = self.rec(embeddings)
        rec_out = rec_out[-1]  # take output of last RNN layer
        drop = F.dropout(rec_out, p=self.dropout)
        # Remember that the final non-linearity should be softmax, so
        # that our predict_proba method outputs actual probabilities!
        out = F.softmax(self.output(drop), dim=-1)
        return out


def create_pipeline(
    vocab_size=1000,
    max_len=50,
    use_cuda=False,
    **kwargs
):
    return Pipeline([
        ('to_idx', TextFeaturizer(max_features=vocab_size)),
        ('pad', Padder2d(max_len=max_len, pad_value=vocab_size, dtype=int)),
        ('net', NeuralNetClassifier(
            RNNClassifier,
            device=('cuda' if use_cuda else 'cpu'),
            max_epochs=5,
            lr=0.01,
            optimizer=torch.optim.RMSprop,
            module__vocab_size=vocab_size,
            **kwargs,
        ))
    ])
