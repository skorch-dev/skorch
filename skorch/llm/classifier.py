"""Using LLMs from transformers for zero/few-shot learning

Open tasks:

- Have a "fast/greedy" option for predict - it is not necessary to calculate
  probabilities for all classes up until the last token. When class A has
  probability p_A and class B has p_B(t) < p_A at token t, then no matter what
  the probability for later tokens, it cannot surpass p_A anymore.

- A small use case where the classifiers are used as a transformer in a bigger
  pipeline, e.g. to extract structured knowledge from a text ("Does this
  product description contain the size of the item?")

- a way to format the text/labels/few-shot samples before they're
  string-interpolated, maybe Jinja2?

- Test if this works with a more diverse range of LLMs

- Enable multi-label classification. Would probably require sigmoid instead of
  softmax and a (empirically determined?) threshold.

- Check if it is possible to enable caching for encoder-decoder LLMs like
  flan-t5.

"""

import importlib
import warnings
from string import Formatter

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from transformers import AutoConfig, AutoTokenizer, LogitsProcessor

from skorch.exceptions import LowProbabilityError
from skorch.llm.prompts import DEFAULT_PROMPT_FEW_SHOT, DEFAULT_PROMPT_ZERO_SHOT, DELIM


def _check_format_string(text, kwargs):
    """Check if string can be exactly formatted with kwargs

    This is necessary because even though calling .format(...) on a string will
    raise if a key is missing, it ignores extra keys, and we want to error in
    this case.

    Gives a warning if the text contains placeholders that are not in kwargs, or
    if kwargs contains keys that are not in the text.

    Parameters
    ----------
    text : str
      The string to format.

    kwargs : dict
      The values to use for formatting.

    """
    formatter = Formatter()
    keys = {key for _, key, _, _ in formatter.parse(text) if key is not None}
    keys_expected = set(kwargs.keys())
    num_keys = len(keys_expected)

    if keys == keys_expected:
        return

    keys_missing = keys_expected - keys
    keys_extra = keys - keys_expected
    msg = (
        f"The prompt may not be correct, it expects {num_keys} "
        "placeholders: " + ", ".join(f"'{key}'" for key in sorted(keys_expected))
    )
    if keys_missing:
        msg += ", missing keys: "
        msg += ", ".join(f"'{key}'" for key in sorted(keys_missing))
    if keys_extra:
        msg += ", extra keys: "
        msg += ", ".join(f"'{key}'" for key in sorted(keys_extra))
    warnings.warn(msg)


def _load_model_and_tokenizer(
        name, device, architectures=('Generation', 'LMHead', 'CausalLM')
):
    """Load a transformers model based only on its name

    This is a bit tricky, because we usually require the task as well to load
    the correct model. However, asking users to pass name and task is not very
    user friendly because it makes, for instance, defining grid search
    parameters more cumbersome if two parameters need to be changed together.

    To solve this, we basically guess the correct architecture based on its
    name.

    Parameters
    ----------
    name : str
      The name of the transformers model to load, e.g. ``google/flan-t5-small``.

    device : str or torch.device
      The device to use for the model, e.g. ``'cpu'`` or
      ``torch.device('cuda:0')``.

    architectures : tuple of str (default=('Generation', 'LMHead', 'CausalLM'))
      The architectures allowed to be loaded. An architecture is chosen if one
      of the strings is a substring of the actual architecture name.

    Returns
    -------
    model : torch.nn.Module
      The pre-trained transformers model.

    tokenizer
      The tokenizer for the model.

    Raises
    ------
    ValueError
      If the model architecture cannot be inferred from the name, raise a
      ``ValueError``.

    """
    config = AutoConfig.from_pretrained(name)

    architecture = None
    for arch in config.architectures:
        if any(substr in arch for substr in architectures):
            architecture = arch
            break

    if architecture is None:
        raise ValueError(
            f"Could not identify architecture for model '{name}', try loading "
            "model and tokenizer directly using the corresponding 'Auto' classes "
            "from transformers and pass them to the classifier"
        )

    transformers_module = importlib.import_module("transformers")
    cls = getattr(transformers_module, architecture, None)
    if cls is None:
        raise ValueError(
            f"Could not find a class '{architecture}' in transformers, try loading "
            "model and tokenizer directly using the corresponding 'Auto' classes "
            "from transformers and pass them to the classifier"
        )

    model = cls.from_pretrained(name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def _extend_inputs(inputs, extra):
    """Extend input arguments with an extra column"""
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    extra = torch.atleast_2d(
        torch.tensor(extra, dtype=torch.long, device=input_ids.device)
    )
    inputs_extended = inputs.copy()

    inputs_extended['input_ids'] = torch.cat((input_ids, extra), dim=-1)
    inputs_extended['attention_mask'] = torch.cat(
        (attention_mask, torch.ones_like(extra)), dim=-1
    )
    return inputs_extended


class _LogitsRecorder(LogitsProcessor):
    """Helper class to record logits and force the given label token ids"""
    def __init__(self, label_ids, tokenizer):
        self.recorded_scores = []
        self.label_ids = label_ids
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        idx = len(self.recorded_scores)
        # we pull the logits to CPU because they are not used as input,
        # therefore there is no device mismatch and we save a bit of GPU memory
        self.recorded_scores.append(scores[0].clone().cpu())
        mask = torch.ones(scores.size(), dtype=torch.bool)
        mask[0, self.label_ids[idx]] = False
        scores[mask] = -float('inf')
        return scores


class _CacheModelWrapper:
    """Helper class that caches model generations

    For label ids, if one token sequence is [1, 2, 3] and the next token
    sequence is [1, 2, 4], for the 2nd sequence, the generation will retrieve
    the cached logits for [1, 2] and only generate [4].

    Set use_caching=False to disable it, e.g. for debugging.

    """
    def __init__(self, model, tokenizer, use_caching=True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_caching = use_caching
        self.cache = {}
        self._total_calls = 0
        self._uncached_calls = 0

    def clear(self):
        self.cache.clear()

    def _make_key(self, kwargs):
        input_ids = kwargs['input_ids']
        input_id = input_ids[0].tolist()
        key = str(input_id)
        return key

    def get_cache(self, kwargs):
        if not self.use_caching:
            return

        key = self._make_key(kwargs)
        val = self.cache.get(key)
        return val

    def set_cache(self, kwargs, label_id, scores):
        if not self.use_caching:
            return

        key = self._make_key(kwargs)
        # store 1st element
        self.cache[key] = scores[0]

        # note that label_id i corresponds to score i+1
        # this is because the first score is for the input w/o label_id (only
        # the prompt); for this reason, the two sequences are offset by +1
        input_id = kwargs['input_ids'][0].tolist()
        for lid, score in zip(label_id, scores[1:]):
            input_id.append(lid)
            key = str(input_id)
            self.cache[key] = score

    def generate_logits(self, *, label_id, **kwargs):
        self._total_calls += 1  # mainly for debugging

        recorded_logits = []
        logits_cached = self.get_cache(kwargs)
        while logits_cached is not None:
            if not label_id or label_id[0] == self.tokenizer.eos_token_id:
                # don't extend with eos_token -- it is already there at the end,
                # we don't need it twice
                break

            recorded_logits.append(logits_cached)
            kwargs = _extend_inputs(kwargs, label_id[:1])
            label_id = label_id[1:]
            logits_cached = self.get_cache(kwargs)

        if not label_id:
            # the whole generation was cached
            return recorded_logits

        if label_id[0] == self.tokenizer.pad_token_id:
            # no need to generate on pad tokens
            return recorded_logits

        self._uncached_calls += 1  # mainly for debugging
        recorder = _LogitsRecorder(
            label_ids=label_id,
            tokenizer=self.tokenizer,
        )
        self.model.generate(
            logits_processor=[recorder],
            # TODO: should this be the max len of all labels?
            max_new_tokens=len(label_id),
            **kwargs
        )
        self.set_cache(kwargs, label_id, recorder.recorded_scores)
        return recorded_logits + recorder.recorded_scores[:]


class _LlmBase(ClassifierMixin, BaseEstimator):
    """Base class for LLM models

    This class handles a few of the checks, as well as the whole prediction
    machinery.

    Required attributes are:

    - model_name
    - model
    - tokenizer
    - prompt
    - probas_sum_to_1
    - device
    - error_low_prob
    - threshold_low_prob
    - use_caching

    """
    def check_X_y(self, X, y, **fit_params):
        raise NotImplementedError

    def check_prompt(self, prompt):
        raise NotImplementedError

    def get_prompt(self, text):
        raise NotImplementedError

    def fit(self, X, y, **fit_params):
        # note: should call _fit
        raise NotImplementedError

    def check_classes(self, y):
        return np.unique(y)

    def check_is_fitted(self):
        required_attrs = ['model_', 'tokenizer_', 'prompt_', 'classes_', 'label_ids_']
        check_is_fitted(self, required_attrs)

    @property
    def device_(self):
        # Use whatever device the model is on, not self.device. If a user
        # initializes with a model that is on GPU, we don't want to require them
        # to adjust the device argument, which would be annoying
        return self.model_.device

    def check_args(self):
        # users should either pass the model name, or the model and tokenizer,
        # but not both
        cls_name = self.__class__.__name__
        msg = (
            f"{cls_name} needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        if self.model_name is not None:
            if (self.model is not None) or (self.tokenizer is not None):
                raise ValueError(msg)
        else:
            if (self.model is None) or (self.tokenizer is None):
                raise ValueError(msg)

        possible_values_error_low_prob = ['ignore', 'raise', 'warn', 'return_none']
        if self.error_low_prob not in possible_values_error_low_prob:
            raise ValueError(
                "error_low_prob must be one of "
                f"{', '.join(possible_values_error_low_prob)}; "
                f"got {self.error_low_prob} instead"
            )

        if (self.threshold_low_prob < 0) or (self.threshold_low_prob > 1):
            raise ValueError(
                "threshold_low_prob must be between 0 and 1, "
                f"got {self.threshold_low_prob} instead"
            )

    def _is_encoder_decoder(self, model):
        return hasattr(model, 'get_encoder')

    def _fit(self, X, y, **fit_params):
        """Prepare everything to enable predictions."""
        self.check_args()
        self.check_X_y(X, y)
        if self.model_name is not None:
            self.model_, self.tokenizer_ = _load_model_and_tokenizer(
                self.model_name, device=self.device
            )
        else:
            self.model_, self.tokenizer_ = self.model, self.tokenizer

        if self._is_encoder_decoder(self.model_) and self.use_caching:
            # Explanation: When we have a prompt [1, 2, 3] and label [4, 5], and
            # if the model is an encoder-decoder architecture (seq2seq), then
            # [1, 2, 3] is encoded and [4, 5] are generated by the decoder. If
            # we wanted to cache, we would store the result (among others) for
            # [1, 2, 3, 4]. However, encoding [1, 2, 3, 4] and generating [5] is
            # not the same operation as encoding [1, 2, 3] and generating [4,
            # 5]. Granted, the logits could be very close, but we cannot be sure
            # and it will depend on the model. Therefore, for the time being, we
            # don't allow caching for encoder-decoder models.
            raise ValueError(
                "Caching is not supported for encoder-decoder models, "
                "initialize the model with use_caching=False."
            )

        self.classes_ = self.check_classes(y)
        self.prompt_ = self.check_prompt(self.prompt)
        classes = [str(c) for c in self.classes_]
        self.label_ids_ = self.tokenizer_(classes)['input_ids']
        self.cached_model_ = _CacheModelWrapper(
            self.model_, self.tokenizer_, use_caching=self.use_caching
        )
        return self

    def _predict_one(self, text):
        """Make a prediction for a single sample

        The returned probabilites are *not normalized* yet.

        Raises a ``LowProbabilityError`` if the total probability of all labels
        is 0, or, assuming ``error_low_prob`` is ``'raise'``, when it is below
        ``threshold_low_prob``. This check is done here because we want to raise
        eagerly instead of going through all samples and then raise.

        """
        inputs = self.tokenizer_(text, return_tensors='pt').to(self.device_)

        probas_all_labels = []
        for label_id in self.label_ids_:
            logits = self.cached_model_.generate_logits(label_id=label_id, **inputs)
            logits = torch.vstack(logits)
            probas = torch.nn.functional.softmax(logits, dim=-1)

            # in case we deal with BFloats which are not supported on CPU
            probas = probas.to(torch.float)

            n = len(logits)
            label_id = label_id[:n]  # don't need EOS, PAD, etc.
            probas_at_idx = probas[torch.arange(n), label_id]
            # multiplying the probabilities for token A, B, C, ... This is
            # because we are interested in P(A) AND P(B|A) AND P(C|A,B), etc.
            # p(A) * P(B|A) =
            # p(A) * p(AnB) / p(A) =
            # p(A) * p(A) * p(B) / p(A) =
            # p(A) * p(B)
            # thus the product of individual probalities is correct
            probas_all_labels.append(torch.prod(probas_at_idx).item())

        prob_sum = sum(probas_all_labels)
        if prob_sum == 0.0:
            raise LowProbabilityError("The sum of all probabilities is zero.")

        probs_are_low = prob_sum < self.threshold_low_prob
        if probs_are_low and (self.error_low_prob == 'raise'):
            raise LowProbabilityError(
                f"The sum of all probabilities is {prob_sum:.3f}, "
                f"which is below the minimum threshold of {self.threshold_low_prob:.3f}"
            )

        y_prob = np.array(probas_all_labels).astype(np.float64)
        return y_prob

    def _predict_proba(self, X):
        """Return the unnormalized y_proba

        Warns if the total probability for a sample is below the threshold and
        ``error_low_prob`` is ``'warn'``.

        """
        self.check_is_fitted()
        y_proba = []
        for xi in X:
            text = self.get_prompt(xi)
            proba = self._predict_one(text)
            y_proba.append(proba)
        y_proba = np.vstack(y_proba)

        if self.error_low_prob == 'warn':
            total_low_probas = (y_proba.sum(1) < self.threshold_low_prob).sum()
            if total_low_probas:
                warnings.warn(
                    f"Found {total_low_probas} samples to have a total probability "
                    f"below the threshold of {self.threshold_low_prob:.3f}."
                )

        return y_proba

    def predict_proba(self, X):
        """Return the probabilities predicted by the LLM.

        Predictions will be forced to be one of the labels the model learned
        during ``fit``. Each column in ``y_proba`` corresponds to one class. The
        order of the classes can be checked in the ``.classes_`` attribute. In
        general, it is alphabetic. So the first column is the probability for
        the first class in ``.classes_``, the second column is the probability
        for the second class in ``.classes_``, etc.

        If ``error_low_prob`` is set to ``'warn'``, then a warning is given if,
        for at least one sample, the sum of the probabilities for all classes is
        below the threshold set in ``threshold_low_prob``. If ``error_low_prob``
        is set to ``'raise'``, then an error is raised instead.

        Parameters
        ----------
        X : input data
          Typically, this is a list/array of strings. E.g., this could be a list
          of reviews and the target is the sentiment. Technically, however, this
          can also contain numerical or categorical data, although it is
          unlikely that the LLM will generate good predictions for those.

        Returns
        -------
        y_proba : numpy ndarray
          The probabilities for each class. If ``probas_sum_to_1`` is set to
          ``False``, then the sum of the probabilities for each sample will not
          add up to 1

        """
        # y_proba not normalized here
        y_proba = self._predict_proba(X)

        if self.probas_sum_to_1:
            # normalizing here is okay because we already checked earlier that
            # the sum is not 0
            y_proba /= y_proba.sum(1, keepdims=True)

        return y_proba

    def predict(self, X):
        """Return the classes predicted by the LLM.

        Predictions will be forced to be one of the labels the model learned
        during ``fit`` (see the ``classes_`` attribute). The model will never
        predict a different label.

        If ``error_low_prob`` is set to ``'warn'``, then a warning is given if,
        for at least one sample, the sum of the probabilities for all classes is
        below the threshold set in ``threshold_low_prob``. If ``error_low_prob``
        is set to ``'raise'``, then an error is raised instead.

        If ``error_low_prob`` is set to ``'return_none'``, then the predicted
        class will be replaced by ``None`` if the sum of the probabilities is
        below the threshold set in ``threshold_low_prob``.

        Parameters
        ----------
        X : input data
          Typically, this is a list/array of strings. E.g., this could be a list
          of reviews and the target is the sentiment. Technically, however, this
          can also contain numerical or categorical data, although it is
          unlikely that the LLM will generate good predictions for those.

        Returns
        -------
        y_pred : numpy ndarray
          The label for each class.

        """
        # y_proba not normalized but it's not neeeded here
        y_proba = self._predict_proba(X)
        pred_ids = y_proba.argmax(1)
        y_pred = self.classes_[pred_ids]

        if self.error_low_prob == 'return_none':
            y_pred = y_pred.astype(object)  # prevents None from being cast to str
            mask_low_probas = y_proba.sum(1) < self.threshold_low_prob
            y_pred[mask_low_probas] = None
        return y_pred

    def clear_model_cache(self):
        """Clear the cache of the model

        Call this to free some memory.

        """
        if self.use_caching and hasattr(self, 'cached_model_'):
            # caching is used and model cache has been initialized
            self.cached_model_.clear()

    def __repr__(self):
        # The issue is that the repr for the transformer model can be quite huge
        # and the repr for the tokenizer quite ugly. We solve this by
        # temporarily replacing them by their class names. We should, however,
        # not completely rewrite the repr code because sklearn does some special
        # magic with the repr. This is why super().__repr__() should be used.
        if (self.model is None) and (self.tokenizer is None):
            # the user initialized the class with model name, the repr should be
            # fine
            return super().__repr__()

        model_prev = self.model
        tokenizer_prev = self.tokenizer
        rep = None

        # we are very defensive, making sure to restore the temporarily made
        # change in case of an error
        try:
            self.model = self.model.__class__.__name__
            self.tokenizer = self.tokenizer.__class__.__name__
            rep = super().__repr__()
        finally:
            self.model = model_prev
            self.tokenizer = tokenizer_prev

        if rep is None:
            # if the repr could not be generated, fall back to using the ugly
            # repr
            rep = super().__repr__()

        return rep


class ZeroShotClassifier(_LlmBase):
    """Zero-shot classification using a Large Language Model (LLM).

    This class allows you to use an LLM from Hugging Face transformers for
    zero-shot classification. There is no training during the ``fit`` call,
    instead, the LLM will be prompted to predict the labels for each sample.

    Parameters
    ----------
    model_name : str or None (default=None)
      The name of the model to use. This is the same name as used on Hugging
      Face Hub. For example, to use GPT2, pass ``'gpt2'``, to use the small
      flan-t5 model, pass ``'google/flan-t5-small'``. If the ``model_name``
      parameter is passed, don't pass ``model`` or ``tokenizer`` parameters.

    model : torch.nn.Module or None (default=None)
      The model to use. This should be a PyTorch text generation model from
      Hugging Face Hub or a model with the same API. Most notably, the model
      should have a ``generate`` method. If you pass the ``model``, you should
      also pass the ``tokenizer``, but you shall not pass the ``model_name``.

      Passing the model explicitly instead of the ``model_name`` can have a few
      advantages. Most notably, this allows you to modify the model, e.g.
      changing its config or how the model is loaded. For instance, some models
      can only be loaded with the option ``trust_remote_code=True``. If using
      the ``model_name`` argument, the default settings will be used instead.
      Passing the model explicitly also allows you to use custom models that are
      not uploaded to Hugging Face Hub.

    tokenizer (default=None)
      A tokenizer that is compatible with the model. Typically, this is loaded
      using the ``AutoTokenizer.from_pretrained`` method provided by Hugging
      Face transformers. If you pass the ``tokenizer``, you should also pass the
      ``model``, but you should not pass the ``model_name``.

    prompt : str or None (default=None)

      The prompt to use. This is the text that will be passed to the model to
      generate the prediction. If no prompt is passed, a default prompt will be
      used. The prompt should be a Python string with two placeholders, one
      called ``text`` and one called ``labels``. The ``text`` placeholder will
      replaced by the contents from ``X`` and the ``labels`` placeholder will be
      replaced by the unique labels taken from ``y``.

      An example prompt could be something like this:

          "Classify this text: {text}. Possible labels are {labels}". Your
          response: "

      All general tips for good prompt crafting apply here as well. Be aware
      that if the prompt is too long, it will exceed the context size of the
      model.

    probas_sum_to_1 : bool (default=True)
      If ``True``, then the probabilities for each sample will be normalized to
      sum to 1. If ``False``, the probabilities will not be normalized.

      In general, without normalization, the probabilities will not sum to 1
      because the LLM can generate any token, not just the labels. Since the
      model is restricted to only generate the available labels, there will be
      some probability mass that is unaccounted for. You could consider the
      missing probability mass to be an implicit 'other' class.

      In general, you should set this parameter to ``True`` because the default
      assumption is that probabilities sum to 1. However, setting this to
      ``False`` can be useful for debugging purposes, as it allows you to see
      how much probability the LLM assigns to different tokens. If the total
      probabilities are very low, it could be a sign that the LLM is not
      powerful enough or that the prompt is not well crafted.

    device : str or torch.device (default='cpu')
      The device to use. In general, using a GPU or other accelerated hardware
      is advised if runtime performance is critical.

      Note that if the ``model`` parameter is passed explicitly, the device of
      that model takes precedence over the value of ``device``.

    error_low_prob : {'ignore', 'warn', 'raise', 'return_none'} (default='ignore')
      Controls what should happen if the sum of the probabilities for a sample
      is below a given threshold. When encountering low probabilities, the
      options are to do one of the following:

        - ``'ignore'``: do nothing
        - ``'warn'``: issue a warning
        - ``'raise'``: raise an error
        - ``'return_none'``: return ``None`` as the prediction when calling
          ``.predict``

      The threshold is controlled by the ``threshold_low_prob`` parameter.

    threshold_low_prob : float (default=0.0)
      The threshold for the sum of the probabilities below which they are
      considered to be too low. The consequences of low probabilities are
      controlled by the ``error_low_prob`` parameter.

    use_caching : bool (default=True)
      If ``True``, the predictions for each sample will be cached, as well as
      the intermediate result for each generated token. This can speed up
      predictions when some samples are duplicated, or when labels have a long
      common prefix. An example of the latter would be if a label is called
      "intent.support.email" and another label is called "intent.support.phone",
      then the tokens for the common prefix "intent.support." are reused for
      both labels, as their probabilities are identical.

      Note that caching is currently not supported for encoder-decoder
      architectures such as flan-t5. If you want to use such an architecture,
      turn caching off.

      If you see any issues you might suspect are caused by caching, turn this
      option off, see if it helps, and report the issue on the skorch GitHub
      page.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
      A list of class labels known to the classifier. This attribute can be used
      to identify which column in the probabilties returned by ``predict_proba``
      corresponds to which class.

    """
    def __init__(
            self,
            model_name=None,
            *,
            model=None,
            tokenizer=None,
            prompt=None,
            probas_sum_to_1=True,
            device='cpu',
            error_low_prob='ignore',
            threshold_low_prob=0.0,
            use_caching=True,
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.probas_sum_to_1 = probas_sum_to_1
        self.device = device
        self.error_low_prob = error_low_prob
        self.threshold_low_prob = threshold_low_prob
        self.use_caching = use_caching

    def check_prompt(self, prompt):
        """Check if the prompt is well formed.

        If no prompt is provided, return the default prompt.

        Raises
        ------
        ValueError
          When the prompt is not well formed.

        """
        if prompt is None:
            prompt = DEFAULT_PROMPT_ZERO_SHOT


        kwargs = {
            'text': "some text",
            'labels': ["foo", "bar"],
        }
        _check_format_string(prompt, kwargs)
        return prompt

    def get_prompt(self, text):
        """Return the prompt for the given sample."""
        self.check_is_fitted()
        return self.prompt_.format(text=text, labels=self.classes_.tolist())

    def check_X_y(self, X, y, **fit_params):
        """Check that input data is well-behaved."""
        # X can be None but not y
        if y is None:
            raise ValueError(
                "y cannot be None, as it is used to infer the existing classes"
            )

        if not isinstance(y[0], str):
            # don't raise an error, as, hypothetically, the LLM could also
            # predict encoded targets, but it's not advisable
            warnings.warn(
                "y should contain the name of the labels as strings, e.g. "
                "'positive' and 'negative', don't pass label-encoded targets"
            )

    def fit(self, X, y, **fit_params):
        """Prepare everything to enable predictions.

        There is no actual fitting going on here, as the LLM is used as is.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
          The input data. For zero-shot classification, this can be ``None``.

        y : array-like of shape (n_samples,)
          The target classes. Ensure that each class that the LLM should be able
          to predict is present at least once. Classes that are not present
          during the ``fit`` call will never be predicted.

        **fit_params : dict
          Additional fitting parameters. This is mostly a placeholder for
          sklearn-compatibility, as there is no actual fitting process.

        Returns
        -------
        self
          The fitted estimator.

        """
        return self._fit(X, y, **fit_params)


class FewShotClassifier(_LlmBase):
    """Few-shot classification using a Large Language Model (LLM).

    This class allows you to use an LLM from Hugging Face transformers for
    few-shot classification. There is no training during the ``fit`` call,
    instead, the LLM will be prompted to predict the labels for each sample.

    Parameters
    ----------
    model_name : str or None (default=None)
      The name of the model to use. This is the same name as used on Hugging
      Face Hub. For example, to use GPT2, pass ``'gpt2'``, to use the small
      flan-t5 model, pass ``'google/flan-t5-small'``. If the ``model_name``
      parameter is passed, don't pass ``model`` or ``tokenizer`` parameters.

    model : torch.nn.Module or None (default=None)
      The model to use. This should be a PyTorch text generation model from
      Hugging Face Hub or a model with the same API. Most notably, the model
      should have a ``generate`` method. If you pass the ``model``, you should
      also pass the ``tokenizer``, but you shall not pass the ``model_name``.

      Passing the model explicitly instead of the ``model_name`` can have a few
      advantages. Most notably, this allows you to modify the model, e.g.
      changing its config or how the model is loaded. For instance, some models
      can only be loaded with the option ``trust_remote_code=True``. If using
      the ``model_name`` argument, the default settings will be used instead.
      Passing the model explicitly also allows you to use custom models that are
      not uploaded to Hugging Face Hub.

    tokenizer (default=None)
      A tokenizer that is compatible with the model. Typically, this is loaded
      using the ``AutoTokenizer.from_pretrained`` method provided by Hugging
      Face transformers. If you pass the ``tokenizer``, you should also pass the
      ``model``, but you should not pass the ``model_name``.

    prompt : str or None (default=None)

      The prompt to use. This is the text that will be passed to the model to
      generate the prediction. If no prompt is passed, a default prompt will be
      used. The prompt should be a Python string with three placeholders, one
      called ``text``, one called ``labels``, and one called ``examples``. The
      ``text`` placeholder will replaced by the contents from ``X`` passed to
      during inference and the ``labels`` placeholder will be replaced by the
      unique labels taken from ``y``. The examples will be taken from the ``X``
      and ``y`` seen during ``fit``.

      An example prompt could be something like this:

          "Classify this text: {text}. Possible labels are: {labels}. Here are
          some examples: {examples}. Your response: ".

      All general tips for good prompt crafting apply here as
      well. Be aware that if the prompt is too long, it will exceed the context
      size of the model.

    probas_sum_to_1 : bool (default=True)
      If ``True``, then the probabilities for each sample will be normalized to
      sum to 1. If ``False``, the probabilities will not be normalized.

      In general, without normalization, the probabilities will not sum to 1
      because the LLM can generate any token, not just the labels. Since the
      model is restricted to only generate the available labels, there will be
      some probability mass that is unaccounted for. You could consider the
      missing probability mass to be an implicit 'other' class.

      In general, you should set this parameter to ``True`` because the default
      assumption is that probabilities sum to 1. However, setting this to
      ``False`` can be useful for debugging purposes, as it allows you to see
      how much probability the LLM assigns to different tokens. If the total
      probabilities are very low, it could be a sign that the LLM is not
      powerful enough or that the prompt is not well crafted.

    max_samples: int (default=5)
      The number of samples to use for few-shot learning. The few-shot samples
      are taken from the ``X`` and ``y`` passed to ``fit``.

      This number should be large enough for the LLM to generalize, but not too
      large so as to exceed the context window size. More samples will also
      lower prediction speed.

    device : str or torch.device (default='cpu')
      The device to use. In general, using a GPU or other accelerated hardware
      is advised if runtime performance is critical.

      Note that if the ``model`` parameter is passed explicitly, the device of
      that model takes precedence over the value of ``device``.

    error_low_prob : {'ignore', 'warn', 'raise', 'return_none'} (default='ignore')
      Controls what should happen if the sum of the probabilities for a sample
      is below a given threshold. When encountering low probabilities, the
      options are to do one of the following:

        - ``'ignore'``: do nothing
        - ``'warn'``: issue a warning
        - ``'raise'``: raise an error
        - ``'return_none'``: return ``None`` as the prediction when calling
          ``.predict``

      The threshold is controlled by the ``threshold_low_prob`` parameter.

    threshold_low_prob : float (default=0.0)
      The threshold for the sum of the probabilities below which they are
      considered to be too low. The consequences of low probabilities are
      controlled by the ``error_low_prob`` parameter.

    use_caching : bool (default=True)
      If ``True``, the predictions for each sample will be cached, as well as
      the intermediate result for each generated token. This can speed up
      predictions when some samples are duplicated, or when labels have a long
      common prefix. An example of the latter would be if a label is called
      "intent.support.email" and another label is called "intent.support.phone",
      then the tokens for the common prefix "intent.support." are reused for
      both labels, as their probabilities are identical.

      Note that caching is currently not supported for encoder-decoder
      architectures such as flan-t5. If you want to use such an architecture,
      turn caching off.

      If you see any issues you might suspect are caused by caching, turn this
      option off, see if it helps, and report the issue on the skorch GitHub
      page.

    random_state : int, RandomState instance or None (default=None)
      The choice of examples that are picked for few-shot learning is random. To
      fix the random seed, use this argument.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
      A list of class labels known to the classifier. This attribute can be used
      to identify which column in the probabilties returned by ``predict_proba``
      corresponds to which class.

    """
    def __init__(
            self,
            model_name=None,
            *,
            model=None,
            tokenizer=None,
            prompt=None,
            probas_sum_to_1=True,
            max_samples=5,
            device='cpu',
            error_low_prob='ignore',
            threshold_low_prob=0.0,
            use_caching=True,
            random_state=None,
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.probas_sum_to_1 = probas_sum_to_1
        self.max_samples = max_samples
        self.device = device
        self.error_low_prob = error_low_prob
        self.threshold_low_prob = threshold_low_prob
        self.use_caching = use_caching
        self.random_state = random_state

    def check_prompt(self, prompt):
        """Check if the prompt is well formed.

        If no prompt is provided, return the default prompt.

        Raises
        ------
        ValueError
          When the prompt is not well formed.

        """

        if prompt is None:
            prompt = DEFAULT_PROMPT_FEW_SHOT

        kwargs = {
            'text': "some text",
            'labels': ["foo", "bar"],
            'examples': ["some examples"],
        }
        _check_format_string(prompt, kwargs)
        return prompt

    def get_examples(self, X, y, n_samples):
        """Given input data ``X`` and ``y``, return a subset of ``n_samples``
        for few-shot learning.

        This method aims at providing at least one example for each existing
        class.

        """
        examples = []
        seen_X = set()
        seen_y = set()
        rng = check_random_state(self.random_state)
        indices = rng.permutation(np.arange(len(y)))

        # first batch, fill with one example for each label
        for i in indices:
            if y[i] not in seen_y:
                examples.append((X[i], y[i]))
                seen_X.add(str(X[i]))
                seen_y.add(y[i])
            if len(seen_y) == len(self.classes_):
                # each label is represented
                break
            if len(examples) == n_samples:
                break

        if len(examples) == n_samples:
            return examples

        # second batch, fill with random other examples
        for i in indices:
            if str(X[i]) in seen_X:
                continue

            examples.append((X[i], y[i]))
            if len(examples) == n_samples:
                break

        # return in reverse order so that the label diversity from the 1st batch
        # comes last
        return examples[::-1]


    def fit(self, X, y, **fit_params):
        """Prepare everything to enable predictions.

        There is no actual fitting going on here, as the LLM is used as is. The
        examples used for few-shot learning will be derived from the provided
        input data. The selection mechanism for this is that, for each possible
        label, at least one example is taken from the data (if ``max_samples``
        is large enough).

        To change the way that examples are selected, override the
        ``get_examples`` method.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
          The input data. For zero-shot classification, this can be ``None``.

        y : array-like of shape (n_samples,)
          The target classes. Ensure that each class that the LLM should be able
          to predict is present at least once. Classes that are not present
          during the ``fit`` call will never be predicted.

        **fit_params : dict
          Additional fitting parameters. This is mostly a placeholder for
          sklearn-compatibility, as there is no actual fitting process.

        Returns
        -------
        self
          The fitted estimator.

        """
        self._fit(X, y, **fit_params)
        n_samples = min(self.max_samples, len(y))
        self.examples_ = self.get_examples(X, y, n_samples=n_samples)
        return self

    def get_prompt(self, text):
        """Return the prompt for the given sample."""
        self.check_is_fitted()
        few_shot_examples = []
        for xi, yi in self.examples_:
            few_shot_examples.append(
                f"{DELIM}\n{xi}\n{DELIM}\n\nYour response:\n{yi}\n"
            )
        examples = "\n".join(few_shot_examples)
        return self.prompt_.format(
            text=text, labels=self.classes_.tolist(), examples=examples
        )

    def check_X_y(self, X, y, **fit_params):
        """Check that input data is well-behaved."""
        if (X is None) or (not len(X)):
            raise ValueError("For few-shot learning, pass at least one example")

        if y is None:
            raise ValueError(
                "y cannot be None, as it is used to infer the existing classes"
            )

        if not isinstance(y[0], str):
            # don't raise an error, as, hypothetically, the LLM could also
            # predict encoded targets, but it's not advisable
            warnings.warn(
                "y should contain the name of the labels as strings, e.g. "
                "'positive' and 'negative', don't pass label-encoded targets"
            )

        len_X, len_y = len(X), len(y)
        if len_X != len_y:
            raise ValueError(
                "X and y don't have the same number of samples, found "
                f"{len_X} and {len_y} samples, respectively"
        )
