============
REST Service
============

In this section we'll take the RNN sentiment classifer from the
example `Predicting sentiment on the IMDB dataset
<https://github.com/skorch-dev/skorch/blob/master/examples/rnn_classifer/RNN_sentiment_classification.ipynb>`_
and use it to demonstrate how to easily expose your PyTorch module on
the web using skorch and another library called `Palladium
<https://github.com/ottogroup/palladium>`_.

With Palladium, you define the Palladium dataset, the model, and
Palladium provides the framework to fit, test, and serve your model on
the web.  Palladium comes with its own documentation and a `tutorial
<http://palladium.readthedocs.io/en/latest/user/tutorial.html>`_,
which you may want to check out to learn more about what you can do
with it.

The way to make the dataset and model known to Palladium is through
its configuration file.  Here's the part of the configuration that
defines the dataset and model:

.. code:: python

    {
        'dataset_loader_train': {
            '__factory__': 'model.DatasetLoader',
            'path': 'aclImdb/train/',
        },

        'dataset_loader_test': {
            '__factory__': 'model.DatasetLoader',
            'path': 'aclImdb/test/',
        },

        'model': {
            '__factory__': 'model.create_pipeline',
            'use_cuda': True,
        },

        'model_persister': {
            '__factory__': 'palladium.persistence.File',
            'path': 'rnn-model-{version}',
        },

        'scoring': 'accuracy',
    }

You can save this configuration as ``palladium-config.py``.

The ``dataset_loader_train`` and ``dataset_loader_test`` entries
define where the data comes from.  They refer to a Python class
defined inside the ``model`` module.  Let's create a file and call it
``model.py``, put it in the same directory as the configuration file.
We'll start off with defining the dataset loader:

.. code:: python

    import os
    from urllib.request import urlretrieve
    import tarfile

    import numpy as np
    from palladium.interfaces import DatasetLoader as IDatasetLoader
    from sklearn.datasets import load_files

    DATA_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    DATA_FN = DATA_URL.rsplit('/', 1)[1]


    def download():
        if not os.path.exists('aclImdb'):
            # unzip data if it does not exist
            if not os.path.exists(DATA_FN):
                urlretrieve(DATA_URL, DATA_FN)
            with tarfile.open(DATA_FN, 'r:gz') as f:
                f.extractall()


    class DatasetLoader(IDatasetLoader):
        def __init__(self, path='aclImdb/train/'):
            self.path = path

        def __call__(self):
            download()
            dataset = load_files(self.path, categories=['pos', 'neg'])
            X, y = dataset['data'], dataset['target']
            X = np.asarray([x.decode() for x in X])  # decode from bytes
            return X, y


The most interesting bit here is that our Palladium ``DatasetLoader``
defines a ``__call__`` method that will return the data and the target
(X and y).  Easy.  Note that in the configuration file, we refer to
our ``DatasetLoader`` twice, once for the training set and once for
the test set.

Our configuration also refers to a function ``create_pipeline`` which
we'll create next:

.. code:: python

    from dstoolbox.transformers import Padder2d
    from dstoolbox.transformers import TextFeaturizer
    from sklearn.pipeline import Pipeline
    from skorch import NeuralNetClassifier
    import torch


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


You've noticed that this function's job is to create the model and
return it.  Here, we're defining a pipeline that wraps skorch's
``NeuralNetClassifier``, which in turn is a wrapper around our PyTorch
module, as it's defined in the `predicting sentiment tutorial
<https://github.com/skorch-dev/skorch/blob/master/examples/rnn_classifer/RNN_sentiment_classification.ipynb>`_.
We'll also add the RNNClassifier to ``model.py``:

.. code:: python

    from torch import nn
    F = nn.functional


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


You can find the full contents of the ``model.py`` file in the
``skorch/examples/rnn_classifer`` folder of skorch's source code.

Now with dataset and model in place, it's time to try Palladium out.
You can install Palladium and another dependency we use with ``pip
install palladium dstoolbox``.

From within the directory that contains ``model.py`` and
``palladium-config.py`` now run the following command::

  PALLADIUM_CONFIG=palladium-config.py pld-fit --evaluate

You should see output similar to this::

  INFO:palladium:Loading data...
  INFO:palladium:Loading data done in 0.607 sec.
  INFO:palladium:Fitting model...
    epoch    train_loss    valid_acc    valid_loss     dur
  -------  ------------  -----------  ------------  ------
        1        0.7679       0.5008        0.7617  3.1300
        2        0.6385       0.7100        0.5840  3.1247
        3        0.5430       0.7438        0.5518  3.1317
        4        0.4736       0.7480        0.5424  3.1373
        5        0.4253       0.7448        0.5832  3.1433
  INFO:palladium:Fitting model done in 29.060 sec.
  DEBUG:palladium:Evaluating model on train set...
  INFO:palladium:Train score: 0.83068
  DEBUG:palladium:Evaluating model on train set done in 6.743 sec.
  DEBUG:palladium:Evaluating model on test set...
  INFO:palladium:Test score:  0.75428
  DEBUG:palladium:Evaluating model on test set done in 6.476 sec.
  INFO:palladium:Writing model...
  INFO:palladium:Writing model done in 0.694 sec.
  INFO:palladium:Wrote model with version 1.

Congratulations, you've trained your first model with Palladium!  Note
that in the output you see a train score (accuracy) of 0.83 and a test
score of about 0.75.  These refer to how well your model did on the
training set (defined by ``dataset_loader_train`` in the
configuration) and on the test set (``dataset_loader_test``).

You're ready to now serve the model on the web.  Add this piece of
configuration to the ``palladium-config.py`` configuration file (and
make sure it lives within the outermost brackets:

.. code:: python

    {
        # ...

        'predict_service': {
            '__factory__': 'palladium.server.PredictService',
            'mapping': [
                ('text', 'str'),
            ],
            'predict_proba': True,
            'unwrap_sample': True,
        },

        # ...
    }

With this piece of information inside the configuration, we're ready
to launch the web server using::

  PALLADIUM_CONFIG=palladium-config.py pld-devserver

You can now try out the web service at this address:
http://localhost:5000/predict?text=this+movie+was+brilliant

You should see a JSON string returned that looks something like this:

.. code:: json

    {
        "metadata": {"error_code": 0, "status": "OK"},
        "result": [0.326442807912827, 0.673557221889496],
    }

The ``result`` entry has the probabilities.  Our model assigns 67%
probability to the sentence "this movie was brilliant" to be positive.
By the way, the skorch tutorial itself has tips on how to improve this
model.

The take away is Palladium helps you reduce the boilerplate code
that's needed to get your machine learning project started.  Palladium
has routines to fit, test, and serve models so you don't have to worry
about that, and you can concentrate on the actual machine learning
part.  Configuration and code are separated with Palladium, which
helps organize your experiments and work on ideas in parallel.  Check
out the `Palladium documentation <https://palladium.readthedocs.io>`_
for more.
