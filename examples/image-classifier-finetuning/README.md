# Fine tune an image model for image classification on the beans dataset

## Description

This is a showcase of a script that uses a pretrained vision transformer model to finetune it on an image classification task.

The dataset and model are provided by Hugging Face. With some light wrapping, they can be used with skorch, and thanks to skorch's CLI helper function, the command line interface comes almost free. There is no need to write any argument parsers or help text for the arguments, check it out!

## Installation

On top of all the packages you'd normally install for using skorch, you also need numpydoc and Google Fire:

```bash
python -m pip install fire numpydoc datasets
```

## Dataset

[Beans dataset](https://huggingface.co/datasets/beans)

## Model

By default, use the pretrained 'vit-base-patch32-224-in21k' model by Google:

[Vision Transformer (base-sized model)](https://huggingface.co/google/vit-base-patch32-224-in21k)

## Usage

### Help

```bash
# general help
python train.py net -- --help
# model specific help
python train.py net --help
```

Notice how all the arguments are added automatically. So e.g., even though we never specified that the `verbose` argument on `NeuralNetClassifier` should be exposed, we can still set it to `False` using `--net__verbose=False`. The same is true for all other parameters. On top of that, as long as there is a corresponding docstring (using numpydoc format), the help for the argument will be automatically parsed from the docstring and shown to the user.

### Training

```bash
# train default model
python train.py net
# train with some non-defaults
python train.py net --net__max_epochs=10 --net__batch_size=32 --device=cpu --net__verbose=False --output_file=mymodel.pkl
```

## Notebook

The same example is also shown in [this notebook](https://nbviewer.jupyter.org/github/skorch-dev/skorch/blob/master/notebooks/Hugging_Face_Finetuning.ipynb).
