"""Fine tune an image model for image classification on the beans dataset

https://huggingface.co/datasets/beans

By default, use the pretrained 'vit-base-patch32-224-in21k' model by Google:

https://huggingface.co/google/vit-base-patch32-224-in21k

"""

from functools import partial
import pickle

import fire
import numpy as np
import torch
from datasets import load_dataset
from skorch.helper import parse_args
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar, LRScheduler
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import ViTFeatureExtractor, ViTForImageClassification


DEFAULTS = {
    'feature_extractor__model_name': 'google/vit-base-patch32-224-in21k',
    'net__module__model_name': 'google/vit-base-patch32-224-in21k',
    'net__criterion': nn.CrossEntropyLoss,
    'net__batch_size': 16,
    'net__optimizer': torch.optim.AdamW,
    'net__lr': 2e-4,
    'net__optimizer__weight_decay': 0.0,
    'net__iterator_train__shuffle': True,
    'net__train_split': False,
    'net__max_epochs': 4,
}


def get_data():
    ds = load_dataset('beans')

    X_train = ds['train']['image']
    y_train = np.array(ds['train']['labels'])

    X_valid = ds['validation']['image']
    y_valid = np.array(ds['validation']['labels'])

    return X_train, X_valid, y_train, y_valid


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Image feature extractor

    Parameters
    ----------
    model_name : str (default='google/vit-base-patch32-224-in21k')
      Name of the feature extractor on Hugging Face Hub.

    device : str (default='cuda')
      Computation device, typically 'cuda' or 'cpu'.

    """
    def __init__(
            self,
            model_name='google/vit-base-patch32-224-in21k',
            device='cuda',
    ):
        self.model_name = model_name
        self.device = device

    def fit(self, X, y=None, **fit_params):
        self.extractor_ = ViTFeatureExtractor.from_pretrained(
            self.model_name, device=self.device,
        )
        return self

    def transform(self, X):
        return self.extractor_(X, return_tensors='pt')['pixel_values']


class VitModule(nn.Module):
    """Vision transformer module

    Parameters
    ----------
    model_name : str (default='google/vit-base-patch32-224-in21k')
      Name of the feature extractor on Hugging Face Hub.

    num_classes : int (default=3)
      Number of target classes to classify.

    """
    def __init__(
            self,
            model_name='google/vit-base-patch32-224-in21k',
            num_classes=3,
    ):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

    def forward(self, X):
        X = self.model(X)
        return X.logits


def lr_lambda(current_step: int, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    )


def get_model(num_classes, lr_lambda):
    pipe = Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('net', NeuralNetClassifier(
            VitModule,
            callbacks=[
                LRScheduler(LambdaLR, lr_lambda=lr_lambda),
                ProgressBar(),
            ],
            module__num_classes=num_classes,
        )),
    ])
    return pipe


def save_model(pipe, output_file, trim=True):
    if trim:
        print("Trimming net, cannot be trained further, only use for prediction")
        pipe.steps[-1][1].trim_for_prediction()

    with open(output_file, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"Successfully saved model in {output_file}")


def train(
        seed=1234,
        device='cuda',
        output_file=None,
        # max epochs need to be known beforehand for lr scheduler, so set it explicitly
        **kwargs
):
    parsed = parse_args(kwargs, defaults=DEFAULTS)
    if kwargs.get('help'):
        # don't need to run expensive steps below
        parsed(get_model(num_classes=3, lr_lambda=None))
        return

    torch.manual_seed(seed)
    # set the same device for all pipeline steps
    kwargs['net__device'] = kwargs['feature_extractor__device'] = device

    X_train, X_valid, y_train, y_valid = get_data()
    num_classes = len(set(y_train))
    max_epochs = kwargs.get('net__max_epochs', DEFAULTS['net__max_epochs'])
    lr_lambda_schedule = partial(
        lr_lambda, num_warmup_steps=0.0, num_training_steps=max_epochs
    )
    pipe = parsed(get_model(num_classes=num_classes, lr_lambda=lr_lambda_schedule))

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_valid)
    print(f"Accuracy on validation dataset is {accuracy_score(y_valid, y_pred):.3f}")

    if output_file:
        save_model(pipe, output_file, trim=True)


if __name__ == '__main__':
    fire.Fire({'net': train})
