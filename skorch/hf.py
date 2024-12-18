"""Classes to work with Hugging Face ecosystem (https://huggingface.co/)

E.g. transformers or tokenizers

This module should be treated as a leaf node in the dependency tree, i.e. no
other skorch modules should depend on these classes or import from here. Even
so, don't import any Hugging Face libraries on the root level because skorch
should not depend on them.

"""

import io
import os
import pathlib
from copy import deepcopy
from operator import itemgetter

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from skorch.callbacks import LRScheduler
from skorch.dataset import unpack_data
from skorch.utils import check_is_fitted, params_for


class _HuggingfaceTokenizerBase(TransformerMixin, BaseEstimator):
    """Base class for yet to train and pretrained tokenizers

    Implements the ``vocabulary_`` attribute and the methods
    ``get_feature_names``, ``transform``, and ``inverse_transform``.

    Subclasses should implement the ``fit``  method.

    """
    @property
    def vocabulary_(self):
        if not hasattr(self, 'fast_tokenizer_'):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute 'vocabulary_', "
                f"did you fit it first?"
            )
        return self.fast_tokenizer_.vocab

    # pylint: disable=unused-argument
    def get_feature_names_out(self, input_features=None):
        """Array mapping from feature integer indices to feature name.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.

        """
        # Note: Same implementation as sklearn's CountVectorizer
        return np.asarray(
            [t for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))],
            dtype=object,
        )

    def __sklearn_is_fitted__(self):
        # method is explained here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        return hasattr(self, 'fast_tokenizer_')

    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError

    def transform(self, X):
        """Transform the given data

        Parameters
        ----------
        X : iterable of str
          A list/array of strings or an iterable which generates either strings.

        Returns
        -------
        Xt : transformers.tokenization_utils_base.BatchEncoding
          A Hugging Face ``BatchEncoding`` instance. This is basically a
          dictionary containing the ids of the tokens and some additional fields
          (depending on the parameters), e.g. the attention mask. For this
          reason, the output is not well suited to be used with normal sklearn
          models, but it works with Hugging Face transformers and with skorch
          nets.

        """
        check_is_fitted(self)

        # from sklearn, triggers a parameter validation
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )
        X = list(X)  # transformers tokenizer does not accept arrays

        verbose = bool(self.verbose)

        # When using tensors/arrays, truncate/pad to max length, otherwise don't
        return_tensors = None
        truncation = False
        padding = False
        if self.return_tensors is not None:
            return_tensors = self.return_tensors
            padding = 'max_length'
            truncation = True

        Xt = self.fast_tokenizer_(
            X,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_token_type_ids=self.return_token_type_ids,
            return_length=self.return_length,
            return_attention_mask=self.return_attention_mask,
            verbose=verbose,
        )
        return Xt

    def inverse_transform(self, X):
        """Decode encodings back into strings

        Be aware that depending on the tokenizer used, the tokenization can lead
        to loss of information (e.g. words outside the vocabulary). In that
        case, the inverse transformation might not restore the original string
        completely.

        X : transformers.tokenization_utils_base.BatchEncoding
          The transformed object obtained from calling ``.transform``.

        Returns
        -------
        Xt : list of str
          The decoded text.

        """
        check_is_fitted(self, ['fast_tokenizer_'])

        Xt = self.fast_tokenizer_.batch_decode(
            X['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return np.asarray(Xt)

    def tokenize(self, X, **kwargs):
        """Convenience method to use the trained tokenizer for tokenization

        The input text is not encoded into integers, instead the strings are
        kept.

        Use this method if you're mainly interested in splitting the text into
        tokens using the trained Hugging Face tokenizer.

        Parameters
        ----------
        X : iterable of str
          A list/array of strings or an iterable which generates either strings.

        kwargs : dict
          Additional arguments, passed directly to the ``decode`` method of the
          tokenizer, e.g. ``skip_special_tokens=True``.

        Returns
        -------
        Xt : np.ndarray
          2d array containing, in each row, an array of strings corresponding to
          the tokenized input text.

        """
        check_is_fitted(self, ['fast_tokenizer_'])

        encoded = self.transform(X)
        tokenizer = self.fast_tokenizer_
        Xt = []
        for token_ids in encoded['input_ids']:
            tokens = [tokenizer.decode(token_id, **kwargs) for token_id in token_ids]
            Xt.append(tokens)
        return np.asarray(Xt)


class HuggingfaceTokenizer(_HuggingfaceTokenizerBase):
    """Wraps a Hugging Face tokenizer to work as an sklearn transformer

    From the `tokenizers docs
    <https://huggingface.co/docs/tokenizers/python/latest/index.html>`_:

    ::

        🤗 Tokenizers provides an implementation of today’s most used
        tokenizers, with a focus on performance and versatility.

    Use of Hugging Face tokenizers for training on custom data using an sklearn
    compatible API.

    Examples
    --------
    >>> # train a BERT tokenizer from scratch
    >>> from tokenizers import Tokenizer
    >>> from tokenizers.models import WordPiece
    >>> from tokenizers import normalizers
    >>> from tokenizers.normalizers import Lowercase, NFD, StripAccents
    >>> from tokenizers.pre_tokenizers import Whitespace
    >>> from tokenizers.processors import TemplateProcessing
    >>> bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    >>> normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    >>> pre_tokenizer = Whitespace()
    >>> post_processor = TemplateProcessing(
    ...    single="[CLS] $A [SEP]",
    ...    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    ...    special_tokens=[
    ...        ("[CLS]", 1),
    ...        ("[SEP]", 2),
    ...    ],
    ... )
    >>> from skorch.hf import HuggingfaceTokenizer
    >>> hf_tokenizer = HuggingfaceTokenizer(
    ...     tokenizer=bert_tokenizer,
    ...     pre_tokenizer=pre_tokenizer,
    ...     post_processor=post_processor,
    ...     trainer__vocab_size=30522,
    ...     trainer__special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    ... )
    >>> data = ['hello there', 'this is a text']
    >>> hf_tokenizer.fit(data)
    >>> hf_tokenizer.transform(data)

    In general, you can pass both initialized objects and uninitialized objects
    as parameters:

    .. code:: python

        # initialized
        HuggingfaceTokenizer(tokenizer=Tokenizer(model=WordPiece()))
        # uninitialized
        HuggingfaceTokenizer(tokenizer=Tokenizer, model=WordPiece)

    Both approaches work equally well and allow you to, for instance, grid
    search on the tokenizer parameters. However, it is recommended *not* to pass
    an initialized trainer. This is because the trainer will then be saved as an
    attribute on the object, which can be wasteful. Instead, it is best to leave
    the default ``trainer='auto'``, which results in the trainer being derived
    from the model.

    .. note::

        If you want to train the ``HuggingfaceTokenizer`` in parallel (e.g.
        during a grid search), you should probably set the environment variable
        ``TOKENIZERS_PARALLELISM=false``. Otherwise, you may experience slow
        downs or deadlocks.

    Parameters
    ----------
    tokenizer : tokenizers.Tokenizer
      The tokenizer to train.

    model : tokenizers.models.Model
      The model represents the actual tokenization algorithm, e.g. ``BPE``.

    trainer : tokenizers.trainers.Trainer or 'auto' (default='auto')
      Class responsible for training the tokenizer. If 'auto', the correct
      trainer will be inferred from the used model using
      ``model.get_trainer()``.

    normalizer : tokenizers.normalizers.Normalizer or None (default=None)
      Optional normalizer, e.g. for casting the text to lowercase.

    pre_tokenizer : tokenizers.pre_tokenizers.PreTokenizer or None (default=None)
      Optional pre-tokenization, e.g. splitting on space.

    post_processor : tokenizers.processors.PostProcessor
      Optional post-processor, mostly used to add special tokens for BERT etc.

    max_length : int (default=256)
      Maximum number of tokens used per sequence.

    return_tensors : one of None, str, 'pt', 'np', 'tf' (default='pt')
      What type of result values to return. By default, return a padded and
      truncated (to ``max_length``) PyTorch Tensor. Similarly, 'np' results in a
      padded and truncated numpy array. Tensorflow tensors are not supported
      officially supported but should also work. If None or str, return a list
      of lists instead. These lists are not padded or truncated, thus each row
      may have different numbers of elements.

    return_attention_mask : bool (default=True)
      Whether to return the attention mask.

    return_token_type_ids : bool (default=False)
      Whether to return the token type ids.

    return_length : bool (default=False)
      Whether to return the length of the encoded inputs.

    pad_token : str (default='[PAD]')
      A special token used to make arrays of tokens the same size for batching
      purpose. Will then be ignored by attention mechanisms.

    verbose : int (default=0)
      Whether the tokenizer should print more information and warnings.

    Attributes
    ----------
    vocabulary_ : dict
      A mapping of terms to feature indices.

    fast_tokenizer_ : transformers.PreTrainedTokenizerFast
      If you want to extract the Hugging Face tokenizer to use it without skorch,
      use this attribute.

    .. _tokenizers: https://huggingface.co/docs/tokenizers/python/latest/index.html

    """
    prefixes_ = [
        'model', 'normalizer', 'post_processor', 'pre_tokenizer', 'tokenizer', 'trainer'
    ]

    def __init__(
            self,
            tokenizer,
            model=None,
            trainer='auto',
            normalizer=None,
            pre_tokenizer=None,
            post_processor=None,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_length=False,
            pad_token='[PAD]',
            verbose=0,
            **kwargs,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.trainer = trainer
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.post_processor = post_processor
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.return_length = return_length
        self.pad_token = pad_token
        self.verbose = verbose

        self._params_to_validate = set(kwargs.keys())
        vars(self).update(kwargs)

    def _validate_params(self):
        """Check argument names passed at initialization.

        Raises
        ------
        ValueError
          Raises a ValueError if one or more arguments don't seem to
          match or are malformed.

        """
        # This whole method is taken from NeuralNet

        # check for wrong arguments
        unexpected_kwargs = []
        missing_dunder_kwargs = []
        for key in sorted(self._params_to_validate):
            if key.endswith('_'):
                continue

            # see https://github.com/skorch-dev/skorch/pull/590 for
            # why this must be sorted
            for prefix in sorted(self.prefixes_, key=lambda s: (-len(s), s)):
                if key == prefix:
                    break
                if key.startswith(prefix):
                    if not key.startswith(prefix + '__'):
                        missing_dunder_kwargs.append((prefix, key))
                    break
            else:  # no break means key didn't match a prefix
                unexpected_kwargs.append(key)

        msgs = []
        if unexpected_kwargs:
            tmpl = ("__init__() got unexpected argument(s) {}. "
                    "Either you made a typo, or you added new arguments "
                    "in a subclass; if that is the case, the subclass "
                    "should deal with the new arguments explicitly.")
            msg = tmpl.format(', '.join(sorted(unexpected_kwargs)))
            msgs.append(msg)

        for prefix, key in sorted(missing_dunder_kwargs, key=lambda tup: tup[1]):
            tmpl = "Got an unexpected argument {}, did you mean {}?"
            suffix = key[len(prefix):].lstrip('_')
            suggestion = prefix + '__' + suffix
            msgs.append(tmpl.format(key, suggestion))

        if msgs:
            full_msg = '\n'.join(msgs)
            raise ValueError(full_msg)

    def initialized_instance(self, instance_or_cls, kwargs):
        """Return an instance initialized with the given parameters

        This is a helper method that deals with several possibilities for a
        component that might need to be initialized:

        * It is already an instance that's good to go
        * It is an instance but it needs to be re-initialized
        * It's not an instance and needs to be initialized

        For the majority of use cases, this comes down to just comes down to
        just initializing the class with its arguments.

        Parameters
        ----------
        instance_or_cls
          The instance or class or callable to be initialized.

        kwargs : dict
          The keyword arguments to initialize the instance or class. Can be an
          empty dict.

        Returns
        -------
        instance
          The initialized component.

        """
        # This whole method is taken from NeuralNet
        if instance_or_cls is None:
            return None

        is_init = not isinstance(instance_or_cls, type)
        if is_init and not kwargs:
            return instance_or_cls
        if is_init:
            if self.verbose:
                print(f"Re-initializing {instance_or_cls}")
            return type(instance_or_cls)(**kwargs)
        return instance_or_cls(**kwargs)

    def get_params_for(self, prefix):
        """Collect and return init parameters for an attribute."""
        return params_for(prefix, self.__dict__)

    def initialize_model(self):
        kwargs = self.get_params_for('model')
        model = self.model
        if model is None:
            model = getattr(self, 'tokenizer__model', None)
        if model is None:
            # no model defined, should already be set on tokenizer
            return model
        return self.initialized_instance(model, kwargs)

    def initialize_tokenizer(self, model):
        kwargs = self.get_params_for('tokenizer')
        if model is not None:
            kwargs['model'] = model
        tokenizer = self.initialized_instance(self.tokenizer, kwargs)
        return deepcopy(tokenizer)

    def initialize_normalizer(self):
        kwargs = self.get_params_for('normalizer')
        return self.initialized_instance(self.normalizer, kwargs)

    def initialize_pre_tokenizer(self):
        kwargs = self.get_params_for('pre_tokenizer')
        return self.initialized_instance(self.pre_tokenizer, kwargs)

    def initialize_post_processor(self):
        kwargs = self.get_params_for('post_processor')
        return self.initialized_instance(self.post_processor, kwargs)

    def _get_tokenizer_model(self, tokenizer):
        return tokenizer.model

    def initialize_trainer(self):
        """Initialize the trainer

        Infer the trainer type from the model if necessary.

        """
        kwargs = self.get_params_for('trainer')
        trainer = self.trainer
        if trainer is None:
            # The 'trainer' attribute cannot be pickled. To still allow
            # pickling, we set it to None, since it's not actually required from
            # transforming. If the user tries to train, however, we need a
            # trainer. Thus, raise a helpful error message.
            # This might get fixed in a future release of tokenizers
            # https://github.com/huggingface/tokenizers/issues/941
            msg = (
                f"Tried to fit {self.__class__.__name__} but trainer is None; either "
                "you passed the wrong value during initialization or you loaded this "
                "transformer with pickle, which deletes the trainer; if so, please "
                "set the trainer again, e.g. 'tokenizer.trainer = mytrainer'"
            )
            raise TypeError(msg)

        if trainer == 'auto':
            trainer = self.tokenizer_.model.get_trainer()

        return self.initialized_instance(trainer, kwargs)

    def initialize(self):
        """Initialize the individual tokenizer components"""
        self._validate_params()

        model = self.initialize_model()
        tokenizer = self.initialize_tokenizer(model)
        normalizer = self.initialize_normalizer()
        pre_tokenizer = self.initialize_pre_tokenizer()
        post_processor = self.initialize_post_processor()

        if normalizer is not None:
            tokenizer.normalizer = normalizer
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer
        if post_processor is not None:
            tokenizer.post_processor = post_processor
        self.tokenizer_ = tokenizer

        return self

    def fit(self, X, y=None, **fit_params):
        """Train the tokenizer on given data

        Parameters
        ----------
        X : iterable of str
          A list/array of strings or an iterable which generates either strings.

        y : None
          This parameter is ignored.

        fit_params : dict
          This parameter is ignored.

        Returns
        -------
        self : HuggingfaceTokenizer
          The fitted instance of the tokenizer.

        """
        from transformers import PreTrainedTokenizerFast

        # from sklearn, triggers a parameter validation
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )
        X = list(X)    # transformers tokenizer does not accept arrays

        self.initialize()

        trainer = self.initialize_trainer()
        self.tokenizer_.train_from_iterator(X, trainer)
        self.tokenizer_.add_special_tokens([self.pad_token])
        self.fast_tokenizer_ = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer_,
            pad_token=self.pad_token,
        )
        self.fixed_vocabulary_ = False
        return self

    def __getstate__(self):
        # The 'trainer' attribute cannot be pickled. To still allow pickling, we
        # set it to None, since it's not actually required for transforming.
        # This might get fixed in a future release of tokenizers
        # https://github.com/huggingface/tokenizers/issues/941
        state = super().__getstate__()
        if state['trainer'] != 'auto':
            state['trainer'] = None
        return state

    def get_params(self, deep=False):
        params = super().get_params(deep=deep)
        if deep:
            for key in self._params_to_validate:
                # We cannot assume that the attribute is already set because
                # sklearn's set_params calls get_params first.
                if hasattr(self, key):
                    params[key] = getattr(self, key)
        return params

    def set_params(self, **kwargs):
        """Set the parameters of this class.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self

        """
        # similar to NeuralNet.set_params
        normal_params, special_params = {}, {}

        for key, val in kwargs.items():
            if any(key.startswith(prefix) for prefix in self.prefixes_):
                special_params[key] = val
                self._params_to_validate.add(key)
            elif '__' in key:
                special_params[key] = val
                self._params_to_validate.add(key)
            else:
                normal_params[key] = val

        BaseEstimator.set_params(self, **normal_params)

        for key, val in special_params.items():
            if key.endswith('_'):
                raise ValueError(
                    "Something went wrong here. Please open an issue on "
                    "https://github.com/skorch-dev/skorch/issues detailing what "
                    "caused this error.")
            setattr(self, key, val)

        # If the transformer is not initialized or there are no special params,
        # we can exit as this point, because the special_params have been set as
        # attributes and will be applied by initialize() at a later point in
        # time.
        if not hasattr(self, 'tokenizer_') or not special_params:
            return self

        # if transformer is initialized, checking kwargs is possible
        self._validate_params()

        # Re-initializing of tokenizer necessary
        self.initialize()
        if self.verbose:
            print(
                f"{self.__class__.__name__} was re-initialized, please fit it (again)"
            )
        return self


class HuggingfacePretrainedTokenizer(_HuggingfaceTokenizerBase):
    """Wraps a pretrained Huggingface tokenizer to work as an sklearn
    transformer

    From the `tokenizers docs
    <https://huggingface.co/docs/tokenizers/python/latest/index.html>`_:

    ::

        🤗 Tokenizers provides an implementation of today’s most used
        tokenizers, with a focus on performance and versatility.

    Use pretrained Hugging Face tokenizers in an sklearn compatible transformer.

    Examples
    --------
    >>> from skorch.hf import HuggingfacePretrainedTokenizer
    >>> # pass the model name to be downloaded
    >>> hf_tokenizer = HuggingfacePretrainedTokenizer('bert-base-uncased')
    >>> data = ['hello there', 'this is a text']
    >>> hf_tokenizer.fit(data)  # only loads the model
    >>> hf_tokenizer.transform(data)

    >>> # pass pretrained tokenizer as object
    >>> my_tokenizer = ...
    >>> hf_tokenizer = HuggingfacePretrainedTokenizer(my_tokenizer)
    >>> hf_tokenizer.fit(data)
    >>> hf_tokenizer.transform(data)

    >>> # use hyper params from pretrained tokenizer to fit on own data
    >>> hf_tokenizer = HuggingfacePretrainedTokenizer(
    ...     'bert-base-uncased', train=True, vocab_size=12345)
    >>> data = ...
    >>> hf_tokenizer.fit(data)  # fits new tokenizer on data
    >>> hf_tokenizer.transform(data)

    Parameters
    ----------
    tokenizer : str or os.PathLike or transformers.PreTrainedTokenizerFast
      If a string, the model id of a predefined tokenizer hosted inside a model
      repo on huggingface.co. Valid model ids can be located at the root-level,
      like bert-base-uncased, or namespaced under a user or organization name,
      like dbmdz/bert-base-german-cased. If a path, A path to a directory
      containing vocabulary files required by the tokenizer, e.g.,
      ./my_model_directory/. Else, should be an instantiated
      ``PreTrainedTokenizerFast``.

    train : bool (default=False)
      Whether to use the pre-trained tokenizer directly as is or to retrain it
      on your data. If you just want to use the pre-trained tokenizer without
      further modification, leave this parameter as False. However, if you want
      to fit the tokenizer on your own data (completely from scratch, forgetting
      what it has learned previously), set this argument to True. The latter
      option is useful if you want to use the same hyper-parameters as the
      pre-trained tokenizer but want the vocabulary to be fitted to your
      dataset. The vocabulary size of this new tokenizer can be set explicitly
      by passing the ``vocab_size`` argument.

    max_length : int (default=256)
      Maximum number of tokens used per sequence.

    return_tensors : one of None, str, 'pt', 'np', 'tf' (default='pt')
      What type of result values to return. By default, return a padded and
      truncated (to ``max_length``) PyTorch Tensor. Similarly, 'np' results in a
      padded and truncated numpy array. Tensorflow tensors are not supported
      officially supported but should also work. If None or str, return a list
      of lists instead. These lists are not padded or truncated, thus each row
      may have different numbers of elements.

    return_attention_mask : bool (default=True)
      Whether to return the attention mask.

    return_token_type_ids : bool (default=False)
      Whether to return the token type ids.

    return_length : bool (default=False)
      Whether to return the length of the encoded inputs.

    pad_token : str (default='[PAD]')
      A special token used to make arrays of tokens the same size for batching
      purpose. Will then be ignored by attention mechanisms.

    vocab_size : int or None (default=None)
      Change this parameter only if you use ``train=True``. In that case, this
      parameter will determine the vocabulary size of the newly trained
      tokenizer. If you set ``train=True`` but leave this parameter as None, the
      same vocabulary size as the one from the initial toknizer will be used.

    verbose : int (default=0)
      Whether the tokenizer should print more information and warnings.

    Attributes
    ----------
    vocabulary_ : dict
      A mapping of terms to feature indices.

    fast_tokenizer_ : transformers.PreTrainedTokenizerFast
      If you want to extract the Hugging Face tokenizer to use it without skorch,
      use this attribute.

    .. _tokenizers: https://huggingface.co/docs/tokenizers/python/latest/index.html

    """

    def __init__(
            self,
            tokenizer,
            train=False,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_length=False,
            verbose=0,
            vocab_size=None,
    ):
        self.tokenizer = tokenizer
        self.train = train
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.return_length = return_length
        self.vocab_size = vocab_size
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        """Load the pretrained tokenizer

        Parameters
        ----------
        X : iterable of str
          This parameter is ignored.

        y : None
          This parameter is ignored.

        fit_params : dict
          This parameter is ignored.

        Returns
        -------
        self : HuggingfacePretrainedTokenizer
          The fitted instance of the tokenizer.

        """
        from transformers import AutoTokenizer

        # from sklearn, triggers a parameter validation
        # even though X is not used, we leave this check in for consistency
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        if not self.train and (self.vocab_size is not None):
            raise ValueError("Setting vocab_size has no effect if train=False")

        if isinstance(self.tokenizer, (str, os.PathLike)):
            self.fast_tokenizer_ = AutoTokenizer.from_pretrained(
                self.tokenizer
            )
        else:
            self.fast_tokenizer_ = self.tokenizer

        if not self.train:
            self.fixed_vocabulary_ = True
        else:
            X = list(X)  # transformers tokenizer does not accept arrays
            vocab_size = (
                self.fast_tokenizer_.vocab_size if self.vocab_size is None
                else self.vocab_size
            )
            self.fast_tokenizer_ = self.fast_tokenizer_.train_new_from_iterator(
                X, vocab_size=vocab_size
            )
            self.fixed_vocabulary_ = False

        return self


class AccelerateMixin:
    """Mixin class to add support for Hugging Face accelerate

    This is an *experimental* feature.

    Use this mixin class with one of the neural net classes (e.g. ``NeuralNet``,
    ``NeuralNetClassifier``, or ``NeuralNetRegressor``) and pass an instance of
    ``Accelerator`` for mixed precision, multi-GPU, or TPU training.

    Install the accelerate library using:

    .. code-block::

      python -m pip install accelerate

    skorch does not itself provide any facilities to enable these training
    features. A lot of them can still be implemented by the user with a little
    bit of extra work but it can be a daunting task. That is why this helper
    class was added: Using this mixin in conjunction with the accelerate library
    should cover a lot of common use cases.

    .. note::

        Under the hood, accelerate uses :class:`~torch.cuda.amp.GradScaler`,
        which does not support passing the training step as a closure.
        Therefore, if your optimizer requires that (e.g.
        :class:`torch.optim.LBFGS`), you cannot use accelerate.

    .. warning::

        Since accelerate is still quite young and backwards compatiblity
        breaking features might be added, we treat its integration as an
        experimental feature. When accelerate's API stabilizes, we will consider
        adding it to skorch proper.

        Also, models accelerated this way cannot be pickled. If you need to save
        and load the net, either use :py:meth:`skorch.net.NeuralNet.save_params`
        and :py:meth:`skorch.net.NeuralNet.load_params` or don't use
        ``accelerate``.

    Examples
    --------
    >>> from skorch import NeuralNetClassifier
    >>> from skorch.hf import AccelerateMixin
    >>> from accelerate import Accelerator
    >>>
    >>> class AcceleratedNet(AccelerateMixin, NeuralNetClassifier):
    ...     '''NeuralNetClassifier with accelerate support'''
    >>>
    >>> accelerator = Accelerator(...)
    >>> # you may pass gradient_accumulation_steps to enable grad accumulation
    >>> net = AcceleratedNet(MyModule,  accelerator=accelerator)
    >>> net.fit(X, y)

    The same approach works with all the other skorch net classes.

    Parameters
    ----------
    accelerator : accelerate.Accelerator
      In addition to the usual parameters, pass an instance of
      ``accelerate.Accelerator`` with the desired settings.

    device : str, torch.device, or None (default=None)
      The compute device to be used. When using accelerate, it is recommended to
      leave device handling to accelerate. Therefore, it is best to leave this
      argument to be None, which means that skorch does not set the device.

    unwrap_after_train : bool (default=True)
      By default, with this option being ``True``, the module(s) and criterion
      are automatically "unwrapped" after training. This means that their
      initial state -- from before they were prepared by the ``accelerator`` --
      is restored. This is necessary to pickle the net.

      There are circumstances where you might want to disable this behavior. For
      instance, when you want to further train the model with AMP enabled (using
      ``net.partial_fit`` or ``warm_start=True``). Also, unwrapping the modules
      means that the advantage of using mixed precision is lost during
      inference. In those cases, if you don't need to pickle the net, you should
      set ``unwrap_after_train=False``.

    callbacks__print_log__sink : 'auto' or callable
      If 'auto', uses the ``print`` function of the accelerator, if it has one.
      This avoids printing the same output multiple times when training
      concurrently on multiple machines. If the accelerator does not have a
      ``print`` function, use Python's ``print`` function instead.

    """
    def __init__(
            self,
            *args,
            accelerator,
            device=None,
            unwrap_after_train=True,
            callbacks__print_log__sink='auto',
            **kwargs
    ):
        super().__init__(
            *args,
            device=device,
            callbacks__print_log__sink=callbacks__print_log__sink,
            **kwargs
        )
        self.accelerator = accelerator
        self.unwrap_after_train = unwrap_after_train
        self._wrapped_with_accelerator = False

    def _validate_params(self):
        super()._validate_params()

        if self.accelerator.device_placement and (self.device is not None):
            raise ValueError(
                "When device placement is performed by the accelerator, set device=None"
            )

    def _initialize_accelerator(self):
        """Prepare everything for use with accelerate"""
        if self._wrapped_with_accelerator:
            return self

        with self._current_init_context('criterion'):
            for name in self._criteria:
                criterion = getattr(self, name + '_')
                if isinstance(criterion, torch.nn.Module):
                    setattr(self, name + '_', self.accelerator.prepare(criterion))

        with self._current_init_context('module'):
            for name in self._modules:
                module = getattr(self, name + '_')
                if isinstance(module, torch.nn.Module):
                    setattr(self, name + '_', self.accelerator.prepare(module))

        with self._current_init_context('optimizer'):
            for name in self._optimizers:
                optimizer = getattr(self, name + '_')
                if isinstance(optimizer, torch.optim.Optimizer):
                    setattr(self, name + '_', self.accelerator.prepare(optimizer))

        for _, callback in self.callbacks_:
            if isinstance(callback, LRScheduler):
                callback.policy_ = self.accelerator.prepare(callback.policy_)

        self._wrapped_with_accelerator = True
        return self

    def initialize(self):
        """Initializes all of its components and returns self."""
        # this should be the same as the parent class, except for the one marked
        # line
        self.check_training_readiness()

        self._initialize_virtual_params()
        self._initialize_callbacks()
        self._initialize_module()
        self._initialize_criterion()
        self._initialize_optimizer()
        self._initialize_history()
        self._initialize_accelerator()  # <= added

        self._validate_params()

        self.initialized_ = True
        return self

    def _initialize_callbacks(self):
        if self.callbacks__print_log__sink == 'auto':
            print_func = getattr(self.accelerator, 'print', print)
            self.callbacks__print_log__sink = print_func
        super()._initialize_callbacks()
        return self

    def train_step(self, batch, **fit_params):
        # Call training step within the accelerator context manager
        with self.accelerator.accumulate(self.module_):
            # Why are we passing only module_ here, even though there might be
            # other modules as well? First of all, there is no possibility to
            # pass multiple modules. Second, the module_ is only used to
            # determine if Distributed Data Parallel is being used, not for
            # anything else. Therefore, passing module_ should be sufficient
            # most of the time.
            return super().train_step(batch, **fit_params)

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        with self.accelerator.autocast():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            self.accelerator.backward(loss)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def get_iterator(self, *args, **kwargs):
        iterator = super().get_iterator(*args, **kwargs)
        iterator = self.accelerator.prepare(iterator)
        return iterator

    def _step_optimizer(self, step_fn):
        # We cannot step_fn as a 'closure' to .step because GradScaler doesn't
        # suppor it:
        # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step
        # Therefore, we need to call step_fn explicitly and step without
        # argument.
        step_fn()
        for name in self._optimizers:
            optimizer = getattr(self, name + '_')
            optimizer.step()

    def _unwrap_accelerator(self):
        if not self._wrapped_with_accelerator:
            return

        for name in self._modules + self._criteria:
            module = getattr(self, name + '_')
            if isinstance(module, torch.nn.Module):
                orig = self.accelerator.unwrap_model(module, keep_fp32_wrapper=False)
                setattr(self, name + '_', orig)
        self._wrapped_with_accelerator = False

    # pylint: disable=unused-argument
    def on_train_end(self, net, X=None, y=None, **kwargs):
        self.accelerator.wait_for_everyone()
        super().on_train_end(net, X=X, y=y, **kwargs)
        if self.unwrap_after_train:
            self._unwrap_accelerator()
        return self

    def evaluation_step(self, batch, training=False):
        # More context:
        # https://github.com/skorch-dev/skorch/issues/944
        # https://huggingface.co/docs/accelerate/quicktour#distributed-evaluation
        output = super().evaluation_step(batch, training=training)
        y_pred = self.accelerator.gather_for_metrics(output)
        return y_pred

    # pylint: disable=missing-function-docstring
    def save_params(self, *args, **kwargs):
        # has to be called even if not main process, or else there is a dead lock
        self.accelerator.wait_for_everyone()

        if not self._wrapped_with_accelerator:
            if self.accelerator.is_main_process:
                super().save_params(*args, **kwargs)
        else:
            # A potential issue with using accelerate is that a model that has
            # been prepared with accelerate is wrapped, so that the keys of the
            # state dict have an additional prefix, "module.". Therefore, when
            # the model is unwrapped when saving and wrapped when loading, or
            # vice versa, there will be a mismatch in the state dict keys. To
            # prevent this, always unwrap before saving. During loading, in case
            # the model is wrapped, this would result in an error, but we take
            # care of unwrapping the model in that case during loading.
            self._unwrap_accelerator()
            try:
                # note: although saving is only done on the main process,
                # unwrapping+wrapping has to be done on all processes, or else
                # there is an error, not sure why
                if self.accelerator.is_main_process:
                    super().save_params(*args, **kwargs)
            finally:
                self._initialize_accelerator()

    # pylint: disable=missing-function-docstring
    def load_params(self, *args, **kwargs):
        self.accelerator.wait_for_everyone()
        prev_device = self.device
        if self.device is None:
            self.device = 'cpu'

        try:
            if not self._wrapped_with_accelerator:
                super().load_params(*args, **kwargs)
            else:
                # A potential issue with using accelerate is that a model that
                # has been prepared with accelerate is wrapped, so that the keys
                # of the state dict have an additional prefix, "module.".
                # Therefore, when the model is unwrapped when saving and wrapped
                # when loading, or vice versa, there will be a mismatch in the
                # state dict keys. Here, we always unwrap the model first before
                # loading (1st case). This would still result in an error in the
                # 2nd case, but we take care of unwrapping the model in that
                # case during saving.
                self._unwrap_accelerator()
                try:
                    super().load_params(*args, **kwargs)
                finally:
                    self._initialize_accelerator()
        finally:
            # ensure that the device remains unchanged in case it was None
            # before calling load_params
            self.device = prev_device


class HfHubStorage:
    """Helper class that allows writing data to the Hugging Face Hub.

    Use this, for instance, in combination with checkpoint callbacks such as
    :class:`skorch.callbacks.training.TrainEndCheckpoint` or
    :class:`skorch.callbacks.training.Checkpoint` to upload the trained model
    directly to the Hugging Face Hub instead of storing it locally.

    To use this, it is necessary to install the `Hugging Face Hub library
    <https://huggingface.co/docs/huggingface_hub/index>`__.

    .. code:: bash

        python -m pip install huggingface_hub

    Note that writes to the Hub are synchronous. Therefore, if the time it takes
    to upload the data is long compared to training the model, there can be a
    signficant slowdown. It is best to use this with
    :class:`skorch.callbacks.training.TrainEndCheckpoint`, as that checkpoint
    only uploads the data once, at the end of training. Also, using this writer
    with :class:`skorch.callbacks.training.LoadInitState` is not supported for
    now because the Hub API does not support model loading yet.

    Parameters
    ----------
    hf_api : instance of huggingface_hub.HfApi
      Pass an instantiated ``huggingface_hub.HfApi`` object here.

    path_in_repo : str
      The name that the file should have in the repo, e.g. ``my-model.pkl``. If
      you want each upload to have a different file name, instead of overwriting
      the file, use a templated name, e.g. ``my-model-{}.pkl``. Then your files
      will be called ``my-model-1.pkl``, ``my-model-2.pkl``, etc. If there are
      already files by this name in the repository, they will be overwritten.

    repo_id : str
      The repository to which the file will be uploaded, for example:
      ``"username/reponame"``.

    verbose : int (default=0)
      Control the level of verbosity.

    local_storage : str, pathlib.Path or None (default=None)
      Indicate temporary storage of the parameters. By default, they are stored
      in-memory. By passing a string or Path to this parameter, you can instead
      store the parameters at the indicated location. There is no automatic
      cleanup, so if you don't need the file on disk, put it into a temp folder.

    sink : callable (default=print)
      The target that the verbose information is sent to. By default, the output
      is printed to stdout, but the sink could also be a logger or
      :func:`~skorch.utils.noop`.

    kwargs : dict
      The remaining arguments are the same as for ``HfApi.upload_file`` (see
      https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_file).

    Attributes
    ----------
    latest_url_ : str
      Stores the latest URL that the file has been uploaded to.

    Examples
    --------
    >>> from huggingface_hub import create_repo, HfApi
    >>> model_name = 'my-skorch-model.pkl'
    >>> params_name = 'my-torch-params.pt'
    >>> repo_name = 'my-user/my-repo'
    >>> token = 'my-secret-token'
    >>> # you can create a new repo like this:
    >>> create_repo(repo_name, token=token, exist_ok=True)
    >>> hf_api = HfApi()
    >>> hub_pickle_writer = HfHubStorage(
    ...     hf_api,
    ...     path_in_repo=model_name,
    ...     repo_id=repo_name,
    ...     token=token,
    ...     verbose=1,
    ... )
    >>> hub_params_writer = HfHubStorage(
    ...     hf_api,
    ...     path_in_repo=params_name,
    ...     repo_id=repo_name,
    ...     token=token,
    ...     verbose=1,
    ... )
    >>> checkpoints = [
    ...     TrainEndCheckpoint(f_pickle=hub_pickle_writer),
    ...     TrainEndCheckpoint(f_params=hub_params_writer),
    ... ]
    >>> net = NeuralNet(..., checkpoints=checkpoints)
    >>> net.fit(X, y)
    >>> # prints:
    >>> # Uploaded model to https://huggingface.co/my-user/my-repo/blob/main/my-skorch-model.pkl
    >>> # Uploaded model to https://huggingface.co/my-user/my-repo/blob/main/my-torch-params.pt
    ...
    >>> # later...
    >>> import pickle
    >>> from huggingface_hub import hf_hub_download
    >>> path = hf_hub_download(repo_name, model_name, use_auth_token=token)
    >>> with open(path, 'rb') as f:
    >>>     net_loaded = pickle.load(f)

    """
    def __init__(
            self,
            hf_api,
            path_in_repo,
            repo_id,
            local_storage=None,
            verbose=0,
            sink=print,
            **kwargs
    ):
        self.hf_api = hf_api
        self.path_in_repo = path_in_repo
        self.repo_id = repo_id
        self.local_storage = local_storage
        self.verbose = verbose
        self.sink = sink
        self.kwargs = kwargs

        self.latest_url_ = None
        self._buffer = None
        self._call_count = 0
        self._needs_flush = False

    def _get_buffer(self):
        if self.local_storage is None:
            return io.BytesIO()

        return open(self.local_storage, 'wb')

    def write(self, content):
        """Upload the file to the Hugging Face Hub"""
        if self._buffer is None:
            self._buffer = self._get_buffer()
        self._buffer.write(content)
        self._needs_flush = True

    def flush(self):
        """Flush buffered file"""
        if not self._needs_flush:
            # This is to prevent double-flushing. Some PyTorch versions create
            # two contexts, resulting in __exit__, and thus flush, being called
            # twice
            return

        if isinstance(self.local_storage, (str, pathlib.Path)):
            self._buffer.close()
            path_or_fileobj = self._buffer.name
        else:
            self._buffer.seek(0)
            path_or_fileobj = self._buffer

        path_in_repo = self.path_in_repo.format(self._call_count)
        return_url = self.hf_api.upload_file(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            **self.kwargs
        )
        if hasattr(return_url, 'commit_url'):
            # starting from huggingface_hub, the return type is now a CommitInfo
            # object instead of a string
            return_url = return_url.commit_url
        self._buffer = None
        self._needs_flush = False

        self.latest_url_ = return_url
        self._call_count += 1
        if self.verbose:
            self.sink(f"Uploaded file to {return_url}")

    # pylint: disable=unused-argument
    def close(self, *args):
        self.flush()

    def seek(self, offset, whence=0):
        raise NotImplementedError("Seek is not (yet) implemented")

    def tell(self):
        raise NotImplementedError("Tell is not (yet) implemented")

    def read(self):
        raise NotImplementedError("Read is not (yet) implemented")
