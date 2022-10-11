"""Run integration tests with Hugging Face platform

See accompanying test-hf-integration.yml GH workflow file.

Specifically, this tests:

- loading a pre-trained tokenizer
- loading a pre-trained transformer model
- saving model (pickle and pytorch) on the model hub

Trigger:

- trigger the action manually at https://github.com/skorch-dev/skorch/actions >
  HF integration tests > run workflow
- the action runs automatically according to the cron schedule defined in the GH
  action

Initial setup (only needs to be done once):

- create a user on https://huggingface.co
- create the organization 'skorch-tests'
- add HF_TOKEN as a secret token to the GitHub repo (unfortunately, this has to
  be a user token, the orga token doesn't have sufficent permissions); now it is
  available in the GH action as ${{ secrets.HF_TOKEN }}

This script will automatically create a private repo on HF for testing and
delete it at the end of the tests.

To run the script locally, you need a skorch Python environment that
additionally contains the Hugging Face dependencies:

$ python -m pip install transformers tokenizers huggingface_hub

Then run:

$ HF_TOKEN=<TOKEN> python scripts/hf-integration-tests.py

"""

import os
import pickle
from contextlib import contextmanager

import numpy as np
import torch
from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from transformers import AutoModelForSequenceClassification

from skorch import NeuralNetClassifier
from skorch.hf import HuggingfacePretrainedTokenizer
from skorch.callbacks import TrainEndCheckpoint
from skorch.hf import HfHubStorage


# Choose a tokenizer and BERT model that work together
TOKENIZER = "distilbert-base-uncased"
PRETRAINED_MODEL = "distilbert-base-uncased"

# bert model hyper-parameters
OPTMIZER = torch.optim.AdamW
LR = 5e-5
MAX_EPOCHS = 2
CRITERION = nn.CrossEntropyLoss
BATCH_SIZE = 8
N_SAMPLES = 100

# device
DEVICE = 'cpu'

# token with permissions to write to the orga on HF Hub
TOKEN = os.environ['HF_TOKEN']
REPO_NAME = 'skorch-tests/test-skorch-hf-ci'
MODEL_NAME = 'skorch-model.pkl'
WEIGHTS_NAME = 'weights.pt'


#######################################
# TESTING TRANSFORMERS AND TOKENIZERS #
#######################################

class BertModule(nn.Module):
    def __init__(self, name, num_labels):
        super().__init__()
        self.name = name
        self.num_labels = num_labels

        self.reset_weights()

    def reset_weights(self):
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels
        )

    def forward(self, **kwargs):
        pred = self.bert(**kwargs)
        return pred.logits


def load_20newsgroup_small():
    dataset = fetch_20newsgroups()
    y = dataset.target
    mask_0_or_1 = (y == 0) | (y == 1)
    X = np.asarray(dataset.data)[mask_0_or_1][:N_SAMPLES]
    y = dataset.target[mask_0_or_1][:N_SAMPLES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )
    return X_train, X_test, y_train, y_test


def get_transformer_model():
    return Pipeline([
        ('tokenizer', HuggingfacePretrainedTokenizer(TOKENIZER)),
        ('net', NeuralNetClassifier(
            BertModule,
            module__name=PRETRAINED_MODEL,
            module__num_labels=2,
            optimizer=OPTMIZER,
            lr=LR,
            max_epochs=MAX_EPOCHS,
            criterion=CRITERION,
            batch_size=BATCH_SIZE,
            iterator_train__shuffle=True,
            device=DEVICE,
        )),
    ])


def test_tokenizers_transfomers():
    print("Testing tokenizers and transfomers started")
    torch.manual_seed(0)
    np.random.seed(0)
    X_train, X_test, y_train, y_test = load_20newsgroup_small()

    pipeline = get_transformer_model()
    pipeline.fit(X_train, y_train)

    with torch.inference_mode():
        y_pred = pipeline.predict(X_test)

    assert accuracy_score(y_test, y_pred) > 0.7
    print("Testing tokenizers and transfomers completed")


########################
# TESTING HF MODEL HUB #
########################

class ClassifierModule(nn.Module):
    def __init__(
            self,
            num_units=30,
            nonlin=nn.ReLU(),
            dropout=0.5,
    ):
        super().__init__()
        self.num_units = num_units
        self.nonlin = nonlin
        self.dropout = dropout

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


def get_classification_data():
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


@contextmanager
def temporary_hf_repo():
    print(f"Creating repo '{REPO_NAME}' on HF Hub")
    create_repo(
        REPO_NAME,
        private=True,
        token=TOKEN,
        exist_ok=True,
    )
    try:
        yield
    finally:
        HfApi().delete_repo(REPO_NAME, token=TOKEN)
        print(f"Deleted repo '{REPO_NAME}' from HF Hub")


def test_hf_model_hub():
    print("Testing HF model hub started")
    torch.manual_seed(0)
    np.random.seed(0)
    X_train, X_test, y_train, y_test = get_classification_data()

    hf_api = HfApi()
    hub_pickle_storer = HfHubStorage(
        hf_api,
        path_in_repo=MODEL_NAME,
        repo_id=REPO_NAME,
        token=TOKEN,
        verbose=1,
    )
    hub_params_storer = HfHubStorage(
        hf_api,
        path_in_repo=WEIGHTS_NAME,
        repo_id=REPO_NAME,
        token=TOKEN,
        verbose=1,
        local_storage='my-model-weights.pt',
    )
    checkpoint = TrainEndCheckpoint(
        f_pickle=hub_pickle_storer,
        f_params=hub_params_storer,
        f_optimizer=None,
        f_criterion=None,
        f_history=None,
    )

    net = NeuralNetClassifier(
        ClassifierModule,
        lr=0.1,
        device=DEVICE,
        iterator_train__shuffle=True,
        callbacks=[checkpoint],
    )

    net.fit(X_train, y_train)
    assert accuracy_score(y_test, net.predict(X_test)) > 0.7
    print(hub_pickle_storer.latest_url_)

    path = hf_hub_download(REPO_NAME, MODEL_NAME, use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        net_loaded = pickle.load(f)

    with torch.inference_mode():
        assert np.allclose(net.predict_proba(X_test), net_loaded.predict_proba(X_test))

    path = hf_hub_download(REPO_NAME, WEIGHTS_NAME, use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        weights_loaded = torch.load(f)

    weights = dict(net.module_.named_parameters())
    assert weights.keys() == weights_loaded.keys()
    for key in weights.keys():
        torch.testing.assert_close(weights[key], weights[key])

    print("Testing HF model hub completed")


def main():
    exceptions = []

    try:
        with temporary_hf_repo():
            test_hf_model_hub()
    except Exception as exc:
        exceptions.append(exc)

    try:
        test_tokenizers_transfomers()
    except Exception as exc:
        exceptions.append(exc)

    if exceptions:
        print("encountered the following exceptions:")
        for exc in exceptions:
            print(exc)
        raise RuntimeError(f"{len(exceptions)} of 2 tests failed")
    print("All tests succeeded")


if __name__ == '__main__':
    main()
