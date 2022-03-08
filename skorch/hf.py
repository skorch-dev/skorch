"""Classes to work with huggingface libraries

E.g. transformers or tokenizers

This module should be treated as a leaf node in the dependency tree, i.e. no
other skorch modules should depend on these classes or import from here. Even
so, don't import any Huggingface libraries on the root level because skorch
should not depend on them.

"""

from copy import deepcopy
from operator import itemgetter

from sklearn.base import BaseEstimator, TransformerMixin

from skorch.utils import check_is_fitted


class _HuggingfaceTokenizerBase(BaseEstimator, TransformerMixin):
    """Base class for yet to train and pretrained tokenizers

    Implements the ``vocabulary_`` attribute and the methods
    ``get_feature_names``, ``transform``, and ``inverse_transform``.

    Subclasses should implement the ``fit``  method.

    """
    @property
    def vocabulary_(self):
        check_is_fitted(self, ['fast_tokenizer_'])
        return self.fast_tokenizer_.vocab

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name.

        Note: Same implementation as sklearn's CountVectorizer

        Returns
        -------
        feature_names : list
            A list of feature names.

        """
        return [t for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))]

    def transform(self, X):
        """Transform the given data

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
        # from sklearn, triggers a parameter validation
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        verbose = bool(getattr(self, 'verbose'))

        # When using tensors/arrays, truncate/pad to max length, otherwise don't
        return_tensors = None
        truncation = False
        padding = False
        if self.return_tensors not in (str, None):
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
        Xt = []
        for x in X['input_ids']:
            Xt.append(
                self.fast_tokenizer_.decode(
                    x, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            )
        return Xt

    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError


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
    >>> from tokenizers.trainers import WordPieceTrainer
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
    >>> trainer = WordPieceTrainer(
    ...     vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    ... )
    >>> from skorch.hf import HuggingfaceTokenizer
    >>> hf_tokenizer = HuggingfaceTokenizer(
    ...     tokenizer=bert_tokenizer,
    ...     trainer=trainer,
    ...     pre_tokenizer=pre_tokenizer,
    ...     post_processor=post_processor,
    ... )
    >>> data = ['hello there', 'this is a text']
    >>> hf_tokenizer.fit(data)
    >>> hf_tokenizer.transform(data)

    Parameters
    ----------
    tokenizer : tokenizers.Tokenizer
      The tokenizer to train.

    trainer : tokenizers.trainers.Trainer
      Class responsible for training the tokenizer.

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

    def __init__(
            self,
            tokenizer,
            trainer,
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
    ):
        self.tokenizer = tokenizer
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

    def fit(self, X, y=None, **fit_params):
        """Train the tokenizer on given data

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

        tokenizer = deepcopy(self.tokenizer)
        if self.normalizer:
            tokenizer.normalizer = self.normalizer
        if self.pre_tokenizer:
            tokenizer.pre_tokenizer = self.pre_tokenizer
        if self.post_processor:
            tokenizer.post_processor = self.post_processor

        if self.trainer is None:
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

        X = list(X)
        tokenizer.train_from_iterator(X, self.trainer)
        tokenizer.add_special_tokens([self.pad_token])
        self.tokenizer_ = tokenizer
        self.fast_tokenizer_ = self._transformers.PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token=self.pad_token,
        )
        self.fixed_vocabulary_ = False
        return self

    def __getstate__(self):
        # The 'trainer' attribute cannot be pickled. To still allow pickling, we
        # set it to None, since it's not actually required from transforming.
        # This might get fixed in a future release of tokenizers
        # https://github.com/huggingface/tokenizers/issues/941
        state = super().__getstate__()
        state['trainer'] = None
        return state


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
    >>> hf_tokenizer = HuggingfacePretrainedTokenizer('bert-base-uncased')
    >>> data = ['hello there', 'this is a text']
    >>> hf_tokenizer.fit(data)
    >>> hf_tokenizer.transform(data)

    Parameters
    ----------
    pretrained_model_name_or_path : str or os.PathLike

      If a string, the model id of a predefined tokenizer hosted inside a model
      repo on huggingface.co. Valid model ids can be located at the root-level,
      like bert-base-uncased, or namespaced under a user or organization name,
      like dbmdz/bert-base-german-cased. If a path, A path to a directory
      containing vocabulary files required by the tokenizer, e.g.,
      ./my_model_directory/.

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

    def __init__(
            self,
            pretrained_model_name_or_path,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_length=False,
            verbose=0,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.return_length = return_length
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        """Load the pretrained tokenizer

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

        self.fast_tokenizer_ = self._transformers.AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.fixed_vocabulary_ = False
        return self
