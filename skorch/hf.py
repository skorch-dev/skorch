"""Classes to work with huggingface libraries

E.g. transformers or tokenizers

This module should be treated as a leaf node in the dependency tree, i.e. no
other skorch modules should depend on these classes or import from here. Even
so, don't import any Huggingface libraries on the root level because skorch
should not depend on them.

"""

import os
from copy import deepcopy
from operator import itemgetter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from skorch.utils import check_is_fitted, params_for


class _HuggingfaceTokenizerBase(BaseEstimator, TransformerMixin):
    """Base class for yet to train and pretrained tokenizers

    Implements the ``vocabulary_`` attribute and the methods
    ``get_feature_names``, ``transform``, and ``inverse_transform``.

    Subclasses should implement the ``fit``  method.

    """
    @property
    def vocabulary_(self):
        if not hasattr(self, 'fast_tokenizer_'):
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute 'vocabulary_', did you fit it first?"
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
          A huggingface ``BatchEncoding`` instance. This is basically a
          dictionary containing the ids of the tokens and some additional fields
          (depending on the parameters), e.g. the attention mask. For this
          reason, the output is not well suited to be used with normal sklearn
          models, but it works with huggingface transformers and with skorch
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
    """Wraps a Huggingface tokenizer to work as an sklearn transformer

    From the tokenizers_ docs:

    ::

        🤗 Tokenizers provides an implementation of today’s most used
        tokenizers, with a focus on performance and versatility.

    Use of Huggingface tokenizers_ for training on custom data using an sklearn
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
      If you want to extract the Huggingface tokenizer to use it without skorch,
      use this attribute.

    .. _tokenizers: https://huggingface.co/docs/tokenizers/python/latest/index.html

    """
    import transformers as _transformers
    import tokenizers as _tokenizers

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

        self._kwargs = kwargs
        vars(self).update(kwargs)

    def _check_kwargs(self, kwargs):
        """Check argument names passed at initialization.

        Raises
        ------
        TypeError
          Raises a TypeError if one or more arguments don't seem to
          match or are malformed.

        Returns
        -------
        kwargs: dict
          Return the passed keyword arguments.

        """
        # This whole method is taken from NeuralNet

        # check for wrong arguments
        unexpected_kwargs = []
        missing_dunder_kwargs = []
        for key in kwargs:
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
            raise TypeError(full_msg)

        return kwargs

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
        self._check_kwargs(self._kwargs)

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
        self.fast_tokenizer_ = self._transformers.PreTrainedTokenizerFast(
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
        params.update(self._kwargs)
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
                self._kwargs[key] = val
            elif '__' in key:
                special_params[key] = val
                self._kwargs[key] = val
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
        self._check_kwargs(self._kwargs)

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

    From the tokenizers_ docs:

    ::

        🤗 Tokenizers provides an implementation of today’s most used
        tokenizers, with a focus on performance and versatility.

    Use pretrained Huggingface tokenizers_ in an sklearn compatible transformer.

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
    >>> hf_tokenizer.fit(data)  # only loads the model
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
      If you want to extract the Huggingface tokenizer to use it without skorch,
      use this attribute.

    .. _tokenizers: https://huggingface.co/docs/tokenizers/python/latest/index.html

    """

    import transformers as _transformers

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
        # from sklearn, triggers a parameter validation
        # even though X is not used, we leave this check in for consistency
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        if not self.train and (self.vocab_size is not None):
            raise ValueError("Setting vocab_size has no effect if train=False")

        if isinstance(self.tokenizer, (str, os.PathLike)):
            self.fast_tokenizer_ = self._transformers.AutoTokenizer.from_pretrained(
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
