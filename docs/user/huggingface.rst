========================
Hugging Face Integration
========================

skorch integrates with some libraries from the `Hugging Face
<https://huggingface.co/>`_ ecosystem. Take a look at the sections below to
learn more.

Accelerate
----------

The :class:`.AccelerateMixin` class can be used to add support for huggingface
accelerate_ to skorch. E.g., this allows you to use mixed precision training
(AMP), multi-GPU training, raining with a TPU, or gradient accumulation. For the
time being, this feature should be considered experimental.

To use this feature, create a new subclass of the neural net class you want to
use and inherit from the mixin class. E.g., if you want to use a
:class:`.NeuralNet`, it would look like this:

.. code:: python

    from skorch import NeuralNet
    from skorch.hf import AccelerateMixin

    class AcceleratedNet(AccelerateMixin, NeuralNet):
        """NeuralNet with accelerate support"""

The same would work for :class:`.NeuralNetClassifier`,
:class:`.NeuralNetRegressor`, etc. Then pass an instance of Accelerator_ with
the desired parameters and you're good to go:

.. code:: python

    from accelerate import Accelerator

    accelerator = Accelerator(...)
    net = AcceleratedNet(
        MyModule,
        accelerator=accelerator,
    )
    net.fit(X, y)

accelerate_ recommends to leave the device handling to the Accelerator_, which
is why ``device`` defautls to ``None`` (thus telling skorch not to change the
device).

To install accelerate_, run the following command inside your Python environment:

.. code:: bash

      python -m pip install accelerate

.. note::

    Under the hood, accelerate uses :class:`~torch.cuda.amp.GradScaler`,
    which does not support passing the training step as a closure.
    Therefore, if your optimizer requires that (e.g.
    :class:`torch.optim.LBFGS`), you cannot use accelerate.


Tokenizers
----------

skorch also provides sklearn-like transformers that work with Hugging Face
`tokenizers <https://huggingface.co/docs/tokenizers/index>`_. The ``transform``
methods of these transformers return data in a dict-like data structure, which
makes them easy to use in conjunction with skorch's :class:`.NeuralNet`. Below
is an example of how to use a pretrained tokenizer with the help of
:class:`skorch.hf.HuggingfacePretrainedTokenizer`:

.. code:: python

    from skorch.hf import HuggingfacePretrainedTokenizer
    # pass the model name to be downloaded
    hf_tokenizer = HuggingfacePretrainedTokenizer('bert-base-uncased')
    data = ['hello there', 'this is a text']
    hf_tokenizer.fit(data)  # only loads the model
    hf_tokenizer.transform(data)

    # use hyper params from pretrained tokenizer to fit on own data
    hf_tokenizer = HuggingfacePretrainedTokenizer(
        'bert-base-uncased', train=True, vocab_size=12345)
    data = ...
    hf_tokenizer.fit(data)  # fits new tokenizer on data
    hf_tokenizer.transform(data)

We also :class:`skorch.hf.HuggingfaceTokenizer` if you don't want to use a
pretrained tokenizer but instead want to train your own tokenizer with
fine-grained control over each component, like which tokenization method to use.

Of course, since both transformers are scikit-learn compatible, you can use them
in a grid search.

Transformers
------------

The Hugging Face `transformers
<https://huggingface.co/docs/transformers/index>`_ library gives you access to
many pretrained deep learning models. There is no special skorch integration for
those, since they're just normal models and can thus be used without further
adjustments (as long as they're PyTorch models).

If you want to see how using ``transformers`` with skorch could look like in
practice, take a look at the `Hugging Face fine-tuning notebook
<https://nbviewer.org/github/skorch-dev/skorch/blob/master/notebooks/Hugging_Face_Finetuning.ipynb>`_.

.. _accelerate: https://github.com/huggingface/accelerate
.. _Accelerator: https://huggingface.co/docs/accelerate/accelerator.html
