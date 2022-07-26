"""Tests for hf.py"""

import difflib
import pickle
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def text_similarity(text1, text2):
    """Very simple text similarity function"""
    def process(text):
        text = text.replace(' ', '').replace('##', '').lower().strip()
        return text

    diffs = list(difflib.Differ().compare(process(text1), process(text2)))
    same = sum(diff.startswith(' ') for diff in diffs)
    total = len(diffs)
    return same / total


@contextmanager
def temporary_set_param(obj, key, val):
    """Temporarily set value

    Avoid permanently mutating the object. This way, the object does not need to
    be re-initialized.

    """
    val_before = obj.get_params()[key]
    try:
        obj.set_params(**{key: val})
        yield
    finally:
        obj.set_params(**{key: val_before})


class _HuggingfaceTokenizersBaseTest:
    """Base class for testing huggingface tokenizer transformers

    Should implement a (parametrized) ``tokenizer`` fixture.

    Tests should not call ``fit`` since that can be expensive for pretrained
    tokenizers. Instead, implement these tests on the subclass if necessary.

    """
    @pytest.fixture(scope='module')
    def data(self):
        return [
            "The Zen of Python, by Tim Peters",
            "Beautiful is better than ugly.",
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Sparse is better than dense.",
            "Readability counts.",
            "Special cases aren't special enough to break the rules.",
            "Although practicality beats purity.",
            "Errors should never pass silently.",
            "Unless explicitly silenced.",
            "In the face of ambiguity, refuse the temptation to guess.",
            "There should be one-- and preferably only one --obvious way to do it.",
            "Although that way may not be obvious at first unless you're Dutch.",
            "Now is better than never.",
            "Although never is often better than *right* now.",
            "If the implementation is hard to explain, it's a bad idea.",
            "If the implementation is easy to explain, it may be a good idea.",
            "Namespaces are one honking great idea -- let's do more of those!",
        ]

    def test_transform(self, tokenizer, data):
        Xt = tokenizer.transform(data)
        assert 'input_ids' in Xt
        assert 'attention_mask' in Xt

        for val in Xt.values():
            assert val.shape[0] == len(data)
            assert val.shape[1] == tokenizer.max_length
            assert isinstance(val, torch.Tensor)

    def test_with_numpy_array(self, tokenizer, data):
        # does not raise
        tokenizer.transform(np.array(data))

    def test_inverse_transform(self, tokenizer, data):
        # Inverse transform does not necessarily result in the exact same
        # output; therefore, we test text similarity.
        Xt = tokenizer.transform(data)
        Xt_inv = tokenizer.inverse_transform(Xt)
        cutoff = 0.9
        for x_orig, x_dec, x_other in zip(data, Xt_inv, data[1:]):
            # check that original and inverse transform are similar
            assert text_similarity(x_orig, x_dec) > cutoff
            assert text_similarity(x_orig, x_other) < cutoff

    def test_tokenize(self, tokenizer, data):
        tokens = tokenizer.tokenize(data)
        assert tokens.shape == (len(data), tokenizer.max_length)
        assert isinstance(tokens[0][0], str)

    def test_tokenize_skip_special_tokens(self, tokenizer, data):
        # Check that the tokens don't contain pad tokens. Only works if we
        # actually know the pad token.
        if not hasattr(tokenizer, 'pad_token'):
            return

        tokens = tokenizer.tokenize(data, skip_special_tokens=True)
        last_col = tokens[:, -1]
        assert not (last_col == tokenizer.pad_token).any()

    def test_vocabulary(self, tokenizer):
        assert isinstance(tokenizer.vocabulary_, dict)

        # vocabulary size is not always exactly as indicated
        vocab_size = pytest.approx(len(tokenizer.vocabulary_), abs=10)
        assert vocab_size == tokenizer.fast_tokenizer_.vocab_size

    def test_get_feature_names_out(self, tokenizer):
        feature_names = tokenizer.get_feature_names_out()
        assert isinstance(feature_names, np.ndarray)
        assert isinstance(feature_names[0], str)

    def test_keys_in_output(self, tokenizer, data):
        Xt = tokenizer.transform(data)

        assert len(Xt) == 2
        assert 'input_ids' in Xt
        assert 'attention_mask' in Xt

    def test_return_token_type_ids(self, tokenizer, data):
        with temporary_set_param(tokenizer, 'return_token_type_ids', True):
            Xt = tokenizer.transform(data)

        assert 'token_type_ids' in Xt

    def test_return_length(self, tokenizer, data):
        with temporary_set_param(tokenizer, 'return_length', True):
            Xt = tokenizer.transform(data)

        assert 'length' in Xt

    def test_return_attention_mask(self, tokenizer, data):
        with temporary_set_param(tokenizer, 'return_attention_mask', False):
            Xt = tokenizer.transform(data)

        assert 'attention_mask' not in Xt

    def test_return_lists(self, tokenizer, data):
        with temporary_set_param(tokenizer, 'return_tensors', None):
            Xt = tokenizer.transform(data)

        assert set(Xt) == {'input_ids', 'attention_mask'}
        for val in Xt.values():
            assert isinstance(val, list)
            assert isinstance(val[0], list)

        # input type ids can have different lengths because they're not padded
        # or truncated
        assert len(set(len(row) for row in Xt['input_ids'])) != 1

    def test_numpy_arrays(self, tokenizer, data):
        with temporary_set_param(tokenizer, 'return_tensors', 'np'):
            Xt = tokenizer.transform(data)

        assert 'input_ids' in Xt
        assert 'attention_mask' in Xt

        for val in Xt.values():
            assert val.shape[0] == len(data)
            assert val.shape[1] == tokenizer.max_length
            assert isinstance(val, np.ndarray)

    def test_pickle(self, tokenizer):
        # does not raise
        pickled = pickle.dumps(tokenizer)
        pickle.loads(pickled)

    def test_deepcopy(self, tokenizer):
        deepcopy(tokenizer)  # does not raise

    def test_clone(self, tokenizer):
        clone(tokenizer)  # does not raise


class TestHuggingfaceTokenizerUninitialized(_HuggingfaceTokenizersBaseTest):
    """Test with (mostly) uninitialized instances of tokenizer etc. being
    passed

    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordLevel, WordPiece, Unigram
    from tokenizers import normalizers
    from tokenizers import pre_tokenizers
    from tokenizers.normalizers import Lowercase, NFD, StripAccents
    from tokenizers.pre_tokenizers import CharDelimiterSplit, Digits, Whitespace
    from tokenizers.processors import ByteLevel, TemplateProcessing
    from tokenizers.trainers import BpeTrainer, UnigramTrainer
    from tokenizers.trainers import WordPieceTrainer, WordLevelTrainer

    # Test one of the main tokenizer types: BPE, WordLevel, WordPiece, Unigram.
    # Individual settings like vocab size or choice of pre_tokenizer may not
    # necessarily make sense.
    settings = {
        'setting0': {
            'tokenizer': Tokenizer,
            'model': BPE,
            'model__unk_token': "[UNK]",
            'trainer': BpeTrainer,
            'trainer__vocab_size': 50,
            'trainer__special_tokens': SPECIAL_TOKENS,
            'trainer__show_progress': False,
            'normalizer': None,
            'pre_tokenizer': CharDelimiterSplit,
            'pre_tokenizer__delimiter': ' ',
            'post_processor': ByteLevel,
            'max_length': 100,
        },
        'setting1': {
            'tokenizer': Tokenizer,
            'tokenizer__model': WordLevel,  # model set via tokenizer__model
            'model__unk_token': "[UNK]",
            'trainer': 'auto',  # infer trainer
            'trainer__vocab_size': 100,
            'trainer__special_tokens': SPECIAL_TOKENS,
            'trainer__show_progress': False,
            'normalizer': Lowercase,
            'pre_tokenizer': Whitespace,
            'post_processor': None,
            'max_length': 100,
        },
        'setting2': {
            'tokenizer': Tokenizer,
            'model': WordPiece(unk_token="[UNK]"),  # initialized model passed
            'trainer__vocab_size': 150,
            'trainer__special_tokens': SPECIAL_TOKENS,
            'trainer__show_progress': False,
            # sequences: no kwargs
            'normalizer': normalizers.Sequence([NFD(), Lowercase(), StripAccents()]),
            'pre_tokenizer': pre_tokenizers.Sequence(
                [Whitespace(), Digits(individual_digits=True)]
            ),
            'post_processor': TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
            ),
            'max_length': 200,
        },
        'setting4': {
            'tokenizer': Tokenizer(model=Unigram()),
        },
    }

    @pytest.fixture(params=settings.keys())
    def tokenizer(self, request, data):
        # return one tokenizer per setting
        from skorch.hf import HuggingfaceTokenizer

        return HuggingfaceTokenizer(**self.settings[request.param]).fit(data)

    def test_fixed_vocabulary(self, tokenizer):
        assert tokenizer.fixed_vocabulary_ is False

    def test_get_params(self):
        from skorch.hf import HuggingfaceTokenizer

        tokenizer = HuggingfaceTokenizer(**self.settings['setting0'])
        params = tokenizer.get_params(deep=True)
        assert 'model__dropout' not in params

        tokenizer.set_params(model__dropout=0.123)
        params = tokenizer.get_params(deep=True)
        assert 'model__dropout' in params

    def test_set_params(self, data):
        from skorch.hf import HuggingfaceTokenizer

        tokenizer = HuggingfaceTokenizer(**self.settings['setting0'])
        tokenizer.set_params(
            model__dropout=0.123,
            trainer__vocab_size=123,
            pre_tokenizer__delimiter='*',
            max_length=456,
        )
        tokenizer.fit(data)

        assert tokenizer.tokenizer_.model.dropout == pytest.approx(0.123)
        assert len(tokenizer.vocabulary_) == pytest.approx(123, abs=5)
        assert tokenizer.tokenizer_.pre_tokenizer.delimiter == '*'
        assert tokenizer.max_length == 456


class TestHuggingfaceTokenizerInitialized(_HuggingfaceTokenizersBaseTest):
    """Test with initialized instances of tokenizer etc. being passed"""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordLevel, WordPiece, Unigram
    from tokenizers import normalizers
    from tokenizers import pre_tokenizers
    from tokenizers.normalizers import Lowercase, NFD, StripAccents
    from tokenizers.pre_tokenizers import Digits, Whitespace
    from tokenizers.processors import ByteLevel, TemplateProcessing
    from tokenizers.trainers import BpeTrainer, UnigramTrainer
    from tokenizers.trainers import WordPieceTrainer, WordLevelTrainer

    # Test one of the main tokenizer types: BPE, WordLevel, WordPiece, Unigram.
    # Individual settings like vocab size or choice of pre_tokenizer may not
    # necessarily make sense.
    settings = {
        'setting0': {
            'tokenizer': Tokenizer(BPE(unk_token="[UNK]")),
            'trainer': BpeTrainer(
                vocab_size=50, special_tokens=SPECIAL_TOKENS, show_progress=False
            ),
            'normalizer': None,
            'pre_tokenizer': Whitespace(),
            'post_processor': ByteLevel(),
            'max_length': 100,
        },
        'setting1': {
            'tokenizer': Tokenizer(WordLevel(unk_token="[UNK]")),
            'trainer': WordLevelTrainer(
                vocab_size=100, special_tokens=SPECIAL_TOKENS, show_progress=False
            ),
            'normalizer': Lowercase(),
            'pre_tokenizer': Whitespace(),
            'post_processor': None,
            'max_length': 100,
        },
        'setting2': {
            'tokenizer': Tokenizer(WordPiece(unk_token="[UNK]")),
            'trainer': WordPieceTrainer(
                vocab_size=150, special_tokens=SPECIAL_TOKENS, show_progress=False
            ),
            'normalizer': normalizers.Sequence([NFD(), Lowercase(), StripAccents()]),
            'pre_tokenizer': pre_tokenizers.Sequence(
                [Whitespace(), Digits(individual_digits=True)]
            ),
            'post_processor': TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
            ),
            'max_length': 200,
        },
        'setting4': {
            'tokenizer': Tokenizer(Unigram()),
            'trainer': UnigramTrainer(
                vocab_size=120, special_tokens=SPECIAL_TOKENS, show_progress=False
            ),
            'normalizer': None,
            'pre_tokenizer': None,
            'post_processor': None,
            'max_length': 250,
        },
    }

    @pytest.fixture(params=settings.keys())
    def tokenizer_not_fitted(self, request):
        # return one tokenizer per setting
        from skorch.hf import HuggingfaceTokenizer

        return HuggingfaceTokenizer(**self.settings[request.param])

    @pytest.fixture
    def tokenizer(self, tokenizer_not_fitted, data):
        return tokenizer_not_fitted.fit(data)

    def test_fixed_vocabulary(self, tokenizer):
        assert tokenizer.fixed_vocabulary_ is False

    @pytest.mark.xfail
    def test_clone(self, tokenizer):
        # This might get fixed in a future release of tokenizers
        # https://github.com/huggingface/tokenizers/issues/941
        clone(tokenizer)  # does not raise

    def test_pickle_and_fit(self, tokenizer, data):
        # This might get fixed in a future release of tokenizers
        # https://github.com/huggingface/tokenizers/issues/941
        pickled = pickle.dumps(tokenizer)
        loaded = pickle.loads(pickled)
        msg = "Tried to fit HuggingfaceTokenizer but trainer is None"
        with pytest.raises(TypeError, match=msg):
            loaded.fit(data)

    def test_pad_token(self, tokenizer, data):
        pad_token = "=FOO="
        tokenizer.set_params(pad_token=pad_token)
        tokenizer.fit(data)
        Xt = tokenizer.transform(['hello there'])
        pad_token_id = Xt['input_ids'][0, -1].item()
        assert tokenizer.vocabulary_[pad_token] == pad_token_id

    def test_not_fitted(self, tokenizer_not_fitted, data):
        with pytest.raises(NotFittedError):
            tokenizer_not_fitted.transform(data)

    def test_fit_with_numpy_array(self, tokenizer, data):
        # does not raise
        tokenizer.fit(np.array(data))

    def test_fit_with_generator(self, tokenizer, data):
        # does not raise
        tokenizer.fit(row for row in data)

    def test_fit_str_raises(self, tokenizer, data):
        msg = r"Iterable over raw text documents expected, string object received"
        with pytest.raises(ValueError, match=msg):
            tokenizer.fit(data[0])


class TestHuggingfacePretrainedTokenizer(_HuggingfaceTokenizersBaseTest):
    @pytest.fixture(scope='module', params=['as string', 'as instance'])
    def tokenizer(self, request, data):
        from transformers import AutoTokenizer
        from skorch.hf import HuggingfacePretrainedTokenizer

        if request.param == 'as string':
            return HuggingfacePretrainedTokenizer('bert-base-cased').fit(data)

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        return HuggingfacePretrainedTokenizer(tokenizer).fit(data)

    def test_no_training_but_vocab_size_set_raises(self, data):
        # Raise an error when user sets vocab_size but has train=False, since it
        # doesn't do anything.
        from transformers import AutoTokenizer
        from skorch.hf import HuggingfacePretrainedTokenizer

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        hf_tokenizer = HuggingfacePretrainedTokenizer(
            tokenizer, train=False, vocab_size=123
        )

        msg = "Setting vocab_size has no effect if train=False"
        with pytest.raises(ValueError, match=msg):
            hf_tokenizer.fit(data)

    def test_fixed_vocabulary(self, tokenizer):
        assert tokenizer.fixed_vocabulary_ is True


class TestHuggingfacePretrainedTokenizerWithFit(_HuggingfaceTokenizersBaseTest):
    vocab_size = 123

    @pytest.fixture(scope='module', params=['as string', 'as instance'])
    def tokenizer(self, request, data):
        # pylint: disable=missing-function-docstring
        from transformers import AutoTokenizer
        from skorch.hf import HuggingfacePretrainedTokenizer

        kwargs = {'train': True, 'vocab_size': self.vocab_size}

        if request.param == 'as string':
            return HuggingfacePretrainedTokenizer('bert-base-cased', **kwargs).fit(data)

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        return HuggingfacePretrainedTokenizer(tokenizer, **kwargs).fit(data)

    def test_fit_with_generator(self, tokenizer, data):
        # does not raise
        tokenizer.fit(row for row in data)

    def test_vocab_size_argument_honored(self, tokenizer):
        vocab_size = len(tokenizer.vocabulary_)
        assert vocab_size == self.vocab_size

    def test_vocab_size_argument_none(self, data):
        # If not set explicitly, the vocab size should be the same one as of the
        # original tokenizer. However, for this test, we don't have enough data
        # to reach that vocab size (28996). Therefore, we test instead that the
        # vocab size is considerably greater than the one seen when we set
        # vocab_size explictly.
        from transformers import AutoTokenizer
        from skorch.hf import HuggingfacePretrainedTokenizer

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        hf_tokenizer = HuggingfacePretrainedTokenizer(tokenizer, train=True).fit(data)

        # The vocab_size is much bigger than in the previous test
        vocab_size = len(hf_tokenizer.vocabulary_)
        assert vocab_size >= 100 + self.vocab_size

    def test_fixed_vocabulary(self, tokenizer):
        assert tokenizer.fixed_vocabulary_ is False


class TestAccelerate:
    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture(scope='module')
    def module_cls(self, classifier_module):
        return classifier_module

    @pytest.fixture
    def accelerator_cls(self):
        # pylint: disable=missing-function-docstring
        pytest.importorskip('accelerate')

        from accelerate import Accelerator
        from accelerate.state import AcceleratorState

        # We have to use this private method because otherwise, the
        # AcceleratorState is not completely reset, which results in an error
        # initializing the Accelerator more than once in the same process.
        # pylint: disable=protected-access
        AcceleratorState._reset_state()
        return Accelerator

    @pytest.fixture
    def net_cls(self, module_cls):
        from skorch import NeuralNetClassifier
        from skorch.hf import AccelerateMixin

        class AcceleratedNet(AccelerateMixin, NeuralNetClassifier):
            pass

        return partial(
            AcceleratedNet,
            module=module_cls,
            max_epochs=2,
            lr=0.1,
        )

    @pytest.mark.parametrize('mixed_precision', ['no', 'fp16', 'bf16'])
    def test_mixed_precision(self, net_cls, accelerator_cls, data, mixed_precision):
        # Only test if training works at all, no specific test of whether the
        # indicated precision is actually used, since that depends on the
        # underlying hardware.
        from accelerate.utils import is_bf16_available

        if (mixed_precision != 'no') and not torch.cuda.is_available():
            pytest.skip('skipping AMP test because device does not support it')
        if (mixed_precision == 'bf16') and not is_bf16_available():
            pytest.skip('skipping bf16 test because device does not support it')

        accelerator = accelerator_cls(mixed_precision=mixed_precision)
        net = net_cls(accelerator=accelerator)
        X, y = data
        net.fit(X, y)  # does not raise
        assert np.isfinite(net.history[:, "train_loss"]).all()

    def test_force_cpu(self, net_cls, accelerator_cls, data):
        accelerator = accelerator_cls(device_placement=False, cpu=True)
        net = net_cls(accelerator=accelerator)
        net.set_params(device='cpu')
        net.fit(*data)  # does not raise
        assert np.isfinite(net.history[:, "train_loss"]).all()

    def test_device_placement(self, net_cls, accelerator_cls, data):
        accelerator = accelerator_cls(device_placement=True)
        net = net_cls(accelerator=accelerator)
        net.set_params(device='cpu')
        msg = "When device placement is performed by the accelerator, set device=None"
        with pytest.raises(ValueError, match=msg):
            net.fit(*data)

    def test_print_log_sink_auto_uses_accelerator_print(self, net_cls, accelerator_cls):
        # the net defaults to using the accelerator's print function
        accelerator = accelerator_cls()
        net = net_cls(accelerator=accelerator)
        net.initialize()
        print_log = dict(net.callbacks_)['print_log']
        assert print_log.sink == accelerator.print

    def test_print_log_sink_can_be_overwritten(self, net_cls, accelerator_cls):
        # users can still set their own sinks for print log
        accelerator = accelerator_cls()
        net = net_cls(accelerator=accelerator, callbacks__print_log__sink=123)
        net.initialize()
        print_log = dict(net.callbacks_)['print_log']
        assert print_log.sink == 123

    def test_print_log_sink_uses_print_if_accelerator_has_no_print(
            self, net_cls, accelerator_cls
    ):
        # we should not depend on the accelerator having a print function

        # we need to use Mock here because Accelerator does not allow attr
        # deletion
        accelerator = Mock(spec=accelerator_cls())
        accelerator.prepare = lambda x: x
        delattr(accelerator, 'print')
        net = net_cls(accelerator=accelerator)
        net.initialize()
        print_log = dict(net.callbacks_)['print_log']
        assert print_log.sink is print

    def test_all_components_prepared(self, module_cls, data):
        # We cannot test whether accelerate is really performing its job.
        # Instead, we test that all modules, optimizers, and lr schedulers, even
        # custom user-defined ones, are properly prepared. We also test that
        # loss.backward() is called. This means that we do test implementation
        # details of accelerate that may change in the future.
        from skorch import NeuralNetClassifier
        from skorch.callbacks import LRScheduler
        from skorch.hf import AccelerateMixin

        # pylint: disable=missing-docstring
        class MockAccelerator:
            def __init__(self):
                self.device_placement = True
                self.print = print

            def prepare(self, *args):
                for arg in args:
                    arg.is_prepared = True
                return args if len(args) > 1 else args[0]

            def backward(self, loss, **kwargs):
                loss.backward(**kwargs)
                loss.backward_was_called = True

            def unwrap_model(self, model):
                return model

            # pylint: disable=unused-argument
            @contextmanager
            def accumulate(self, model):
                yield

        # pylint: disable=missing-docstring,arguments-differ
        class AcceleratedNet(AccelerateMixin, NeuralNetClassifier):
            def get_iterator(self, *args, **kwargs):
                iterator = super().get_iterator(*args, **kwargs)
                assert iterator.is_prepared
                return iterator

            def initialize_criterion(self):
                super().initialize_criterion()
                kwargs = self.get_params_for('criterion')
                # pylint: disable=attribute-defined-outside-init
                self.criterion2_ = self.criterion(**kwargs)
                return self

            def initialize_module(self):
                super().initialize_module()
                kwargs = self.get_params_for('module')
                # pylint: disable=attribute-defined-outside-init
                self.module2_ = self.module(**kwargs)
                return self

            def initialize_optimizer(self, *args, **kwargs):
                super().initialize_optimizer(*args, **kwargs)
                named_parameters = self.module2_.named_parameters()
                args, kwargs = self.get_params_for_optimizer(
                    'optimizer', named_parameters)
                # pylint: disable=attribute-defined-outside-init
                self.optimizer2_ = self.optimizer(*args, **kwargs)
                return self

            def infer(self, *args, **kwargs):
                # check that all modules and criteria are prepared
                assert self.module_.is_prepared
                assert self.module2_.is_prepared
                assert self.criterion_.is_prepared
                assert self.criterion2_.is_prepared
                return super().infer(*args, **kwargs)

            def train_step_single(self, *args, **kwargs):
                # check that all optimizers and the lr scheduler are prepared,
                # and that loss.backward() was called,
                assert self.optimizer_.is_prepared
                assert self.optimizer2_.is_prepared

                lr_scheduler = dict(self.callbacks_)['lr_scheduler'].policy_
                assert lr_scheduler.is_prepared

                output = super().train_step_single(*args, **kwargs)
                assert output['loss'].backward_was_called
                return output

        accelerator = MockAccelerator()
        net = AcceleratedNet(
            module_cls,
            device=None,
            accelerator=accelerator,
            max_epochs=2,
            callbacks=[('lr_scheduler', LRScheduler)],
        )
        X, y = data
        # does not raise
        net.fit(X, y)
        net.predict(X)

        # make sure that even after resetting parameters, components are still prepared
        net.set_params(
            module__hidden_units=7,
            lr=0.05,
            batch_size=33,
            criterion__reduction='sum',
            callbacks__lr_scheduler__policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
        )
        # does not raise
        net.fit(X, y)
        net.predict(X)

    def test_gradient_accumulation_with_accelerate(
            self, module_cls, accelerator_cls, data
    ):
        # Check that using gradient accumulation provided by accelerate actually
        # works. Testing this is not quite trivial. E.g. we cannot check haven
        # often optimizer.step() is called because accelerate still calls it on
        # each step but does not necessarily update the weights. Therefore, we
        # check if there was an update step by comparing the weights before and
        # after the train_step call. If the weights changed, then there was a
        # step, otherwise not.
        from skorch import NeuralNetClassifier
        from skorch.hf import AccelerateMixin

        def weight_sum(module):
            return sum(weights.sum() for weights in module.parameters())

        # Record for each training step if there was an update of the weights
        updated = []

        # pylint: disable=missing-docstring
        class GradAccNet(AccelerateMixin, NeuralNetClassifier):
            # pylint: disable=arguments-differ
            def train_step(self, *args, **kwargs):
                # Note: We use a very simplified way of checking if weights were
                # updated by just comparing their sum. This way, we don't need
                # to keep a copy around.
                weight_sum_before = weight_sum(self.module_)
                step = super().train_step(*args, **kwargs)
                weight_sum_after = weight_sum(self.module_)
                update_occurred = (weight_sum_before != weight_sum_after).item()
                updated.append(update_occurred)
                return step

        max_epochs = 2
        acc_steps = 3
        accelerator = accelerator_cls(gradient_accumulation_steps=acc_steps)
        net = GradAccNet(module_cls, accelerator=accelerator, max_epochs=max_epochs)
        X, y = data
        net.fit(X, y)

        # Why we expect this outcome: Since acc_steps is 3, we expect that
        # updated should be [False, False, True]. However, since we have 1000
        # samples and a batch size of 128, every 7th batch is the last batch of
        # the epoch, after which there should also be an update. Therefore,
        # every 7th entry is also True.
        updated_expected = [False, False, True, False, False, True, True] * max_epochs
        assert updated == updated_expected
