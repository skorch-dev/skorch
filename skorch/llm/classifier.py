"""Using LLMs from transformers for zero/few shot learning

TODO:

- Determine if it's possible to initialize the model and tokenizer from a single
  argument (no task required), e.g. clf = ZeroShotClassifier(model_name='gpt2').
  The problem is that for the model, it is not obvious which class to use
  (AutoModelForSeq2SeqLM, AutoModelForCausalLM, anything else?). Ideally, this
  should be inferred somehow.

- Have a "fast/greedy" option for predict - it is not necessary to calculate
  probabilities for all classes up until the last token. When class A has
  probability p_A and class B has p_B(t) < p_A at token t, then no matter what
  the probability for later tokens, it cannot surpass p_A anymore.

- A small use case where the classifiers are used as a transformer in a bigger
  pipeline, e.g. to extract structured knowledge from a text ("Does this
  product description contain the size of the item?")

- Probability threshold to identify if the prompt might be totally off. If
  probability is too low:

  - do nothing special (default)
  - predict returns None at those indices
  - give a warning
  - raise an error

- a way to format the text/labels/few shot samples before they're
  string-interpolated, maybe Jinja2?

- Test if this works with a more diverse range of LLMs

"""

from string import Formatter

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessor,
)

# TODO: move prompts to separate module
DEFAULT_PROMPT_ZERO_SHOT = """You are a text classification assistant.

The text to classify:

```
{text}
```

Choose the label among the following possibilities with the highest probability:

{labels}

Only return the label, nothing more.

Your response:
"""

DEFAULT_PROMPT_FEW_SHOT = """You are a text classification assistant.

Choose the label among the following possibilities with the highest probability.
Only return the label, nothing more:

{labels}

Here are a few examples:

{examples}

The text to classify:

```
{text}
```

Your response:
"""


def _check_format_string(text, kwargs):
    """Check if string can be exactly formatted with kwargs

    This is necessary because even though calling .format(...) on a string will
    raise if a key is missing, it ignores extra keys, and we want to error in
    this case.

    Parameters
    ----------
    text : str
      The string to format.

    kwargs : dict
      The values to use for formatting.

    Raises
    ------
    ValueError
      If the text contains placeholders that are not in kwargs, or if kwargs
      contains keys that are not in the text, raises a ``ValueError``.

    """
    formatter = Formatter()
    keys = {key for _, key, _, _ in formatter.parse(text) if key is not None}
    keys_expected = set(kwargs.keys())
    num_keys = len(keys_expected)

    if keys != keys_expected:
        keys_missing = keys_expected - keys
        keys_extra = keys - keys_expected
        msg = (
            f"The prompt is not correct, it should have exactly {num_keys} "
            "placeholders: " + ", ".join(f"'{key}'" for key in sorted(keys_expected))
        )
        if keys_missing:
            msg += ", missing keys: "
            msg += ", ".join(f"'{key}'" for key in sorted(keys_missing))
        if keys_extra:
            msg += ", extra keys: "
            msg += ", ".join(f"'{key}'" for key in sorted(keys_extra))
        raise ValueError(msg)


def _insert_2nd_to_last(tensor, middle, dim=-1):
    # Insert 2nd to last because last token is EOS token
    n = tensor.shape[dim]
    left_side, right_side = tensor.split([n - 1, 1], dim=dim)
    return torch.cat((left_side, middle, right_side), dim=dim)


def _extend_inputs(inputs, extra):
    """Extend input arguments with an extra column"""
    extra = torch.atleast_2d(extra)
    inputs_extended = inputs.copy()
    inputs_extended['input_ids'] = _insert_2nd_to_last(
        inputs_extended['input_ids'], extra
    )
    inputs_extended['attention_mask'] = _insert_2nd_to_last(
        inputs_extended['attention_mask'], torch.ones_like(extra)
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

    If one token sequence is [1, 2, 3] and the next token sequence is [1, 2, 4],
    for the 2nd sequence, the generation will retrieve the cached logits for [1,
    2] and only generate [4].

    Set use_caching=False to disable it, e.g. for debugging.

    """
    def __init__(self, model, tokenizer, use_caching=True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_caching = use_caching
        self.cache = {}
        self._cached_calls = 0
        self._uncached_calls = 0

    def get_cache(self, kwargs):
        if not self.use_caching:
            return None

        input_ids = kwargs['input_ids']
        input_id = input_ids[0].tolist()
        key = str(input_id)
        val = self.cache.get(key)

        # for debugging
        self._cached_calls += val is not None
        self._uncached_calls += val is None

        return val

    def set_cache(self, kwargs, label_id, scores):
        input_ids = kwargs['input_ids']
        input_id = input_ids[0].tolist()

        # store 1st element
        key = str(input_id)
        self.cache[key] = scores[0]

        # note that label_id i corresponds to score i+1
        # this is because the first score is for the input w/o label_id (only
        # the prompt) for this reason, the two sequences are offset by +1
        for lid, score in zip(label_id.tolist(), scores[1:]):
            input_id.insert(-1, lid)
            key = str(input_id)
            self.cache[key] = score

    def generate_logits(self, *, label_id, **kwargs):
        recorded_logits = []

        logits_cached = self.get_cache(kwargs)
        while logits_cached is not None:
            if label_id[:1] == self.tokenizer.eos_token_id:
                # don't extend with eos_token -- it is already there at the end,
                # we don't need it twice
                break

            recorded_logits.append(logits_cached)
            kwargs = _extend_inputs(kwargs, label_id[:1])
            label_id = label_id[1:]
            logits_cached = self.get_cache(kwargs)

        if not len(label_id):
            # the whole generation was cached
            return recorded_logits

        if label_id[0] == self.tokenizer.pad_token_id:
            # no need to generate on pad tokens
            return recorded_logits

        recorder = _LogitsRecorder(
            label_ids=label_id,
            tokenizer=self.tokenizer,
        )
        self.model.generate(
            logits_processor=[recorder],
            max_new_tokens=len(label_id),
            **kwargs
        )
        self.set_cache(kwargs, label_id, recorder.recorded_scores)
        return recorded_logits + recorder.recorded_scores[:]


class _LlmBase(BaseEstimator, ClassifierMixin):
    @classmethod
    def from_auto_model_for_seq2seq_lm(cls, modelname, device=None, **kwargs):
        # TODO: infer the correct type of model, check e.g. here:
        # https://github.com/huggingface/transformers/blob/118e9810687dd713b6be07af79e80eeb1d916908/src/transformers/pipelines/__init__.py#L428
        # if this works, this should be moved to __init__ but keep the
        # possibility to initialize with model and tokenizer directly
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        model = AutoModelForSeq2SeqLM.from_pretrained(modelname)
        if device:
            model = model.to(device)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    @classmethod
    def from_auto_model_for_causal_lm(cls, modelname, device=None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        model = AutoModelForCausalLM.from_pretrained(modelname)
        if device:
            model = model.to(device)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    def check_prompt(self, prompt):
        raise NotImplementedError

    def check_classes(self, y):
        return np.unique(y)

    def check_is_fitted(self):
        # TODO
        pass

    @property
    def device_(self):
        return self.model.device

    def check_X_y(self, X, y, **fit_params):
        raise NotImplementedError

    def fit(self, X, y, **fit_params):
        self.check_X_y(X, y)
        self.classes_ = self.check_classes(y)
        self.prompt_ = self.check_prompt(self.prompt)
        self.label_ids_ = self.tokenizer(
            self.classes_.tolist(),
            return_tensors='pt',
            padding=True,  # TODO check if padding is really needed
        ).to(self.device_)['input_ids']
        self.cached_model_ = _CacheModelWrapper(
            self.model, self.tokenizer, use_caching=self.use_caching
        )
        return self

    def _predict_one(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
        ).to(self.device_)

        probas_all_labels = []
        for label_id in self.label_ids_:
            logits = self.cached_model_.generate_logits(label_id=label_id, **inputs)
            logits = logits[:-1]  # TODO last token is EOS, keep it or not?
            logits = torch.vstack(logits)
            probas = torch.nn.functional.softmax(logits, dim=-1)

            n = len(logits)
            label_id = label_id[:n]  # don't need EOS, PAD, etc.
            probas_at_idx = probas[torch.arange(n), label_id.cpu()]
            # multiplying the probabilities for token A, B, C, ... This is
            # because we are interested in P(A) AND P(B|A) AND P(C|A,B), etc.
            # p(A) * P(B|A) =
            # p(A) * p(AnB) / p(A) =
            # p(A) * p(A) * p(B) / p(A) =
            # p(A) * p(B)
            # thus the product of individual probalities is correct
            probas_all_labels.append(torch.prod(probas_at_idx).item())

        y_prob = np.array(probas_all_labels).astype(np.float64)
        if self.probas_sum_to_1:
            y_prob /= y_prob.sum()

        return y_prob

    def predict_proba(self, X):
        self.check_is_fitted()
        y_proba = []
        for xi in X:
            text = self.get_prompt(xi)
            pred = self._predict_one(text)
            y_proba.append(pred)
        return np.vstack(y_proba)

    def predict(self, X):
        pred_ids = self.predict_proba(X).argmax(1)
        return self.classes_[pred_ids]


class ZeroShotClassifier(_LlmBase):
    """TODO"""
    def __init__(
            self,
            model,
            tokenizer,
            prompt=None,
            probas_sum_to_1=True,
            generate_kwargs=None,
            use_caching=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.probas_sum_to_1 = probas_sum_to_1
        self.generate_kwargs = generate_kwargs
        self.use_caching = use_caching

    def check_prompt(self, prompt):
        if prompt is not None:
            _check_format_string(
                prompt, {'text': "some text", 'labels': ["foo", "bar"]}
            )
            return prompt

        return DEFAULT_PROMPT_ZERO_SHOT

    def get_prompt(self, text):
        self.check_is_fitted()
        return self.prompt_.format(text=text, labels=self.classes_)

    def check_X_y(self, X, y, **fit_params):
        # TODO proper errors
        assert y is not None
        assert not fit_params

    def __repr__(self):
        # TODO self.tokenizer has a very ugly repr, can we replace it?
        return super().__repr__()


class FewShotClassifier(_LlmBase):
    """TODO"""
    def __init__(
            self,
            model,
            tokenizer,
            prompt=None,
            probas_sum_to_1=True,
            max_samples=5,
            generate_kwargs=None,
            use_caching=True,
            random_state=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.probas_sum_to_1 = probas_sum_to_1
        self.max_samples = max_samples
        self.generate_kwargs = generate_kwargs
        self.use_caching = use_caching
        self.random_state = random_state

    def check_prompt(self, prompt):
        if prompt is not None:
            kwargs = {
                'text': "some text",
                'labels': ["foo", "bar"],
                'examples': ["some examples"],
            }
            _check_format_string(prompt, kwargs)

        return DEFAULT_PROMPT_FEW_SHOT

    def _get_examples(self, X, y, n_samples):
        # TODO ensure that each label is present
        # check if stratified shuffle split could be used
        examples = []
        seen_targets = set()
        rng = check_random_state(self.random_state)
        indices = rng.permutation(np.arange(len(y)))

        # first batch, fill with one example for each label
        for i in range(len(y)):
            j = indices[i]
            if y[j] not in seen_targets:
                examples.append((X[j], y[j]))
                seen_targets.add(y[j])
            if len(seen_targets) == len(self.classes_):
                # each target represented
                break

        if len(examples) == n_samples:
            return examples

        # second batch, fill with random other examples
        for i in range(i, len(y)):
            j = indices[i]
            examples.append((X[j], y[j]))
            if len(examples) == n_samples:
                break

        # return in reverse order so that the label diversity from the 1st batch
        # comes last
        return examples[::-1]

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        self.examples_ = self._get_examples(
            X, y, n_samples=min(self.max_samples, len(y))
        )
        return self

    def get_prompt(self, text):
        self.check_is_fitted()
        few_shot_examples = []
        for xi, yi in self.examples_:
            # TODO make the formatting of samples modifiable
            few_shot_examples.append(f"```\n{xi}\n```\n\nYour response:\n{yi}\n")
        examples = "\n".join(few_shot_examples)
        return self.prompt_.format(text=text, labels=self.classes_, examples=examples)

    def check_X_y(self, X, y, **fit_params):
        # TODO proper errors
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert not fit_params

    def __repr__(self):
        # TODO self.tokenizer has a very ugly repr, can we replace it?
        return super().__repr__()
