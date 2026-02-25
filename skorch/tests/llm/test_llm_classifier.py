"""Tests for skorch.llm.classifier"""

import re
import timeit

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError


class TestZeroShotClassifier:
    @pytest.fixture(scope='class')
    def model(self):
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')

    @pytest.fixture(scope='class')
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('google/flan-t5-small')

    @pytest.fixture(scope='class')
    def classifier_cls(self):
        from skorch.llm import ZeroShotClassifier
        return ZeroShotClassifier

    @pytest.fixture
    def X(self):
        return [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]

    def test_classes(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, ['positive', 'negative', 'very positive', 'foobar'])
        # classes_ are sorted
        expected = np.array(['foobar', 'negative', 'positive', 'very positive'])
        np.testing.assert_equal(clf.classes_, expected)

    def test_init_encoder_decoder_with_caching_raises(
            self, classifier_cls, model, tokenizer
    ):
        msg = (
            "Caching is not supported for encoder-decoder models, "
            "initialize the model with use_caching=False."
        )
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=True)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_encoder_decoder_with_caching_using_model_name_raises(
            self, classifier_cls, model, tokenizer
    ):
        msg = (
            "Caching is not supported for encoder-decoder models, "
            "initialize the model with use_caching=False."
        )
        clf = classifier_cls('google/flan-t5-small', use_caching=True)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_without_model_raises(self, classifier_cls, tokenizer):
        msg = (
            "ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls(tokenizer=tokenizer)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_without_tokenizer_raises(self, classifier_cls, model):
        msg = (
            "ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls(model=model)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_with_model_and_model_name_raises(self, classifier_cls, model):
        msg = (
            "ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls('google/flan-t5-small', model=model)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_with_tokenizer_and_model_name_raises(self, classifier_cls, tokenizer):
        msg = (
            "ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls('google/flan-t5-small', tokenizer=tokenizer)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_wrong_error_low_prob_raises(self, classifier_cls, model, tokenizer):
        clf = classifier_cls(model=model, tokenizer=tokenizer, error_low_prob='foo')
        msg = (
            "error_low_prob must be one of ignore, raise, warn, return_none; "
            "got foo instead"
        )
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_wrong_threshold_low_prob_raises(
            self, classifier_cls, model, tokenizer
    ):
        clf = classifier_cls(model=model, tokenizer=tokenizer, threshold_low_prob=-0.1)
        msg = "threshold_low_prob must be between 0 and 1, got -0.1 instead"
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

        clf = classifier_cls(model=model, tokenizer=tokenizer, threshold_low_prob=99)
        msg = "threshold_low_prob must be between 0 and 1, got 99 instead"
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_no_fitting_architecture_raises(self, classifier_cls):
        # resnet-18 exists but cannot be used for language generation
        clf = classifier_cls('microsoft/resnet-18')
        msg = (
            "Could not identify architecture for model 'microsoft/resnet-18', "
            "try loading model and tokenizer directly using the corresponding 'Auto' "
            "classes from transformers and pass them to the classifier"
        )
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_no_fit_predict_raises(self, classifier_cls, model, tokenizer, X):
        # When calling predict/predict_proba before fitting, an error is raised
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)

        # don't check the exact message, as it might change in the future
        with pytest.raises(NotFittedError):
            clf.predict(X)
        with pytest.raises(NotFittedError):
            clf.predict_proba(X)

    def test_fit_y_none_raises(self, classifier_cls, model, tokenizer):
        # X can be None but y should not be None
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        msg = "y cannot be None, as it is used to infer the existing classes"
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, None)

    def test_fit_warning_if_y_not_strings(
            self, classifier_cls, model, tokenizer, recwarn
    ):
        # y should be strings but also accepts other types, but will give a
        # warning
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, [1, 2, 3])
        assert len(recwarn.list) == 1

        expected = (
            "y should contain the name of the labels as strings, e.g. "
            "'positive' and 'negative', don't pass label-encoded targets"
        )
        assert str(recwarn.list[0].message) == expected

    def test_predict(self, model, tokenizer, classifier_cls, X):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, ['negative', 'positive'])
        y_pred = clf.predict(X)
        np.testing.assert_array_equal(
            y_pred,
            np.array(['positive', 'negative', 'positive']),
        )

    def test_predict_proba(self, model, tokenizer, classifier_cls, X):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, ['positive', 'negative'])
        y_proba = clf.predict_proba(X)

        assert y_proba.shape == (3, 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        assert (y_proba >= 0.0).all()
        assert (y_proba <= 1.0).all()

    def test_init_from_model_name(self, classifier_cls, X):
        clf = classifier_cls('google/flan-t5-small', use_caching=False)
        # check that none of the below raise
        clf.fit(None, ['positive', 'negative'])
        clf.predict_proba(X)
        clf.predict(X)

    def test_proba_for_unlikely_label_low(self, model, tokenizer, classifier_cls, X):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, ['positive', 'negative', 'zip'])
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (3, 3)
        assert (y_proba[:, -1] < 2e-3).all()

    def test_predict_proba_labels_differing_num_tokens(
            self, model, tokenizer, classifier_cls, X
    ):
        # positive and negative have 1 token, foobar has 3 tokens
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, ['foobar', 'positive', 'negative'])
        y_proba = clf.predict_proba(X)

        # foobar is column 0
        assert (y_proba[:, 0] < 1e-3).all()

    def test_predict_proba_not_normalized(self, model, tokenizer, classifier_cls, X):
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, probas_sum_to_1=False
        )
        clf.fit(None, ['negative', 'positive'])
        y_proba = clf.predict_proba(X)
        assert (y_proba.sum(axis=1) < 1.0).all()

    def test_same_X_same_probas(self, model, tokenizer, classifier_cls, X):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(None, ['foo', 'bar'])

        y_proba = clf.predict_proba(X)
        y_proba2 = clf.predict_proba(X)
        y_proba3 = clf.predict_proba(X)

        np.testing.assert_allclose(y_proba, y_proba2)
        np.testing.assert_allclose(y_proba, y_proba3)

    def test_same_X_caching(self, model, tokenizer, classifier_cls):
        # Check that caching is performed correctly. On first call, there is
        # only an uncached call. On subsequent call with the same argument,
        # there are 4 cached calls, 1 from the prompt and 3 for the tokens from
        # the label.
        #clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf = classifier_cls('gpt2')
        clf.fit(None, ['foobar'])
        X = ["A masterpiece, instant classic, 5 stars out of 5"]

        y_proba = clf.predict_proba(X)
        assert clf.cached_model_._uncached_calls == 1
        assert clf.cached_model_._total_calls == 1

        y_proba2 = clf.predict_proba(X)
        assert clf.cached_model_._uncached_calls == 1
        assert clf.cached_model_._total_calls == 2
        np.testing.assert_allclose(y_proba, y_proba2)

        y_proba3 = clf.predict_proba(X)
        assert clf.cached_model_._uncached_calls == 1
        assert clf.cached_model_._total_calls == 3
        np.testing.assert_allclose(y_proba, y_proba3)

    def test_caching_is_faster(self, classifier_cls):
        # use a decoder-only model
        clf = classifier_cls('gpt2')
        # classes should have a long common prefix for caching to make a
        # difference
        X = ["A masterpiece, instant classic, 5 stars out of 5"]
        y = ['absolutely undoubtedly positive', 'absolutely undoubtedly negative']
        clf.fit(X=None, y=y)

        # measure time for uncached call using timeit
        uncached_time = timeit.timeit(lambda: clf.predict_proba(X), number=1)
        cached_time = timeit.timeit(lambda: clf.predict_proba(X), number=1)
        # at least 1/3 faster
        assert cached_time < 0.1 * uncached_time

    def test_caching_works_shared_label_prefix_without_eos(self, classifier_cls):
        clf = classifier_cls('gpt2')

        # carefully chosen class labels so that one label has the other label as
        # its prefix. '11111' = '11' + '111'. For models that tokenize single
        # digits indepdentenly this is far more relevant.
        X = np.array(["Hey there", "No thank you"])
        y = ['11', '11111']

        clf.fit(X, y)

        y_pred_1 = clf.predict(X)
        y_pred_2 = clf.predict(X)

        # does not raise and gives the same results
        np.testing.assert_array_equal(y_pred_1, y_pred_2)

    def test_custom_prompt(self, model, tokenizer, classifier_cls, X):
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        # just checking that this works, we don't necessarily expect the
        # predictions to be correct
        clf.fit(None, ['positive', 'negative'])
        clf.predict_proba(X)
        clf.predict(X)

    def test_defective_prompt_missing_key_raises(
            self, model, tokenizer, classifier_cls, recwarn
    ):
        # the prompt has no 'labels' placeholders
        prompt = "Please classify my text:\n{text}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt may not be correct, it expects 2 "
            "placeholders: 'labels', 'text', missing keys: 'labels'"
        )
        clf.fit(None, ['positive', 'negative'])
        assert str(recwarn.list[0].message) == msg

    def test_defective_prompt_extra_key_raises(
            self, model, tokenizer, classifier_cls, recwarn
    ):
        # the prompt has excess 'examples' placeholder
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        prompt += "Examples: {examples}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt may not be correct, it expects 2 "
            "placeholders: 'labels', 'text', extra keys: 'examples'"
        )
        clf.fit(None, ['positive', 'negative'])
        assert str(recwarn.list[0].message) == msg

    def test_get_prompt(self, classifier_cls, model, tokenizer):
        prompt = "Foo {labels} bar {text}"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )
        clf.fit(None, ['label-a', 'label-b'])
        x = "My input"
        expected = "Foo ['label-a', 'label-b'] bar My input"
        assert clf.get_prompt(x) == expected

    def test_causal_lm(self, classifier_cls, X):
        # flan-t5 has an encoder-decoder architecture, here we check that a pure
        # decoder architecture works as well. We're just interested in it
        # working, not if the predictions are good.
        name = 'gpt2'
        clf = classifier_cls(name, probas_sum_to_1=False, use_caching=True)
        clf.fit(None, ['negative', 'positive'])
        clf.predict_proba(X[:3])
        clf.predict(X[:3])

    def test_no_low_probability_no_warning(
            self, classifier_cls, model, tokenizer, X, recwarn
    ):
        # test to explicitly ensure that there is no false warning, as this
        # would go undetected otherwise
        clf = classifier_cls(
            model=model,
            tokenizer=tokenizer,
            use_caching=False,
            threshold_low_prob=0.000001,
            error_low_prob='warn',
        )
        clf.fit(None, ['negative', 'positive'])
        clf.predict_proba(X)
        assert not recwarn.list

    def test_low_probability_warning(
            self, classifier_cls, model, tokenizer, X, recwarn
    ):
        # With a threshold of 0.993, empirically, 2 samples will fall below it
        # and 1 is above it.
        clf = classifier_cls(
            model=model,
            tokenizer=tokenizer,
            use_caching=False,
            threshold_low_prob=0.993,
            error_low_prob='warn',
        )
        clf.fit(None, ['negative', 'positive'])
        clf.predict_proba(X)

        msg = "to have a total probability below the threshold of 0.99"
        assert len(recwarn.list) == 1
        # use `in` because the exact number of decimals is not clear
        assert msg in str(recwarn.list[0].message)

    def test_low_probability_error(self, classifier_cls, model, tokenizer, X):
        from skorch.llm.classifier import LowProbabilityError

        # With a threshold of 0.993, empirically, 2 samples will fall below it
        # and 1 is above it.
        clf = classifier_cls(
            model=model,
            tokenizer=tokenizer,
            use_caching=False,
            threshold_low_prob=0.993,
            error_low_prob='raise',
        )
        clf.fit(None, ['negative', 'positive'])

        msg = (
            r"The sum of all probabilities is \d\.\d+, "
            "which is below the minimum threshold of 0.99"
        )
        with pytest.raises(LowProbabilityError, match=msg):
            clf.predict_proba(X)

    def test_low_probability_return_none(self, classifier_cls, model, tokenizer, X):
        clf = classifier_cls(
            model=model,
            tokenizer=tokenizer,
            use_caching=False,
            threshold_low_prob=0.993,
            error_low_prob='return_none',
        )
        clf.fit(None, ['negative', 'positive'])
        y_pred = clf.predict(X)

        # With a threshold of 0.993, empirically, the first sample will fall
        # below it and the last two above it.
        expected = [None, 'negative', 'positive']
        np.testing.assert_array_equal(y_pred, expected)

    def test_repr(self, classifier_cls, model, tokenizer):
        expected = (
            "ZeroShotClassifier(model='T5ForConditionalGeneration', "
            "tokenizer='T5Tokenizer', use_caching=False)"
        )
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        assert str(clf) == expected
        assert repr(clf) == expected

        clf.fit(None, ['positive', 'negative'])
        assert str(clf) == expected
        assert repr(clf) == expected

    def test_clear_model_cache(self, classifier_cls, X):
        clf = classifier_cls('gpt2')
        clf.fit(None, ['very negative', 'very positive'])

        cache = clf.cached_model_.cache
        assert not cache  # empty at this point

        clf.predict(X)
        # 2 entries for each sample, one for the prompt itself, one for the
        # prompt + "very"
        assert len(cache) == 2 * len(X)

        clf.clear_model_cache()
        assert not cache  # empty again


class TestFewShotClassifier:
    """Most of the functionality of FewShotClassifier is shared with
    ZeroShotClassifier and is thus not tested again here

    """
    @pytest.fixture(scope='class')
    def model(self):
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')

    @pytest.fixture(scope='class')
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('google/flan-t5-small')

    @pytest.fixture(scope='class')
    def classifier_cls(self):
        from skorch.llm import FewShotClassifier
        return FewShotClassifier

    @pytest.fixture
    def X(self):
        return [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]

    @pytest.fixture
    def y(self):
        return np.asarray(['positive', 'negative', 'positive'])

    @pytest.fixture
    def X_test(self):
        return [
            "This was the worst movie I have ever seen",
            "I can totally see why this won the Oscar, just amazing",
            "It's not the worst movie I saw this year but it comes close",
        ]

    @pytest.fixture
    def y_test(self):
        return np.asarray(['negative', 'positive', 'negative'])

    def test_classes(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        X = 4 * ["something text, doesn't matter what"]
        y = ['positive', 'negative', 'very positive', 'foobar']
        clf.fit(X, y)
        # classes_ are sorted
        expected = np.array(['foobar', 'negative', 'positive', 'very positive'])
        np.testing.assert_equal(clf.classes_, expected)

    def test_fit_fewer_samples_than_classes(self, model, tokenizer, classifier_cls):
        # Check that there is no error if we allow fewer samples than there are
        # classes.
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, max_samples=3
        )
        X = 4 * ["something text, doesn't matter what"]
        y = ['positive', 'negative', 'very positive', 'foobar']
        clf.fit(X, y)
        assert len(clf.examples_) == 3

    def test_fit_fewer_samples_than_max_samples(
            self, model, tokenizer, classifier_cls, X, y
    ):
        # Check that there is no error if X and y are smaller than max_samples
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, max_samples=5
        )
        assert len(X) < 5

        clf.fit(X, y)
        assert len(clf.examples_) == len(X)

    def test_fit_X_none_raises(self, model, tokenizer, classifier_cls, y):
        # for zero-shot, having no X is acceptable, but not for few-shot
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        msg = "For few-shot learning, pass at least one example"
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, y)

    def test_fit_X_empty_raises(self, model, tokenizer, classifier_cls, y):
        # for zero-shot, having no X is acceptable, but not for few-shot
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        msg = "For few-shot learning, pass at least one example"
        with pytest.raises(ValueError, match=msg):
            clf.fit([], y)

    def test_fit_y_none_raises(self, model, tokenizer, classifier_cls, X):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        msg = "y cannot be None, as it is used to infer the existing classes"
        with pytest.raises(ValueError, match=msg):
            clf.fit(X, None)

    def test_fit_warning_if_y_not_strings(
            self, classifier_cls, model, tokenizer, X, recwarn
    ):
        # y should be strings but also accepts other types, but will give a
        # warning
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(X, [1, 2, 3])
        assert len(recwarn.list) == 1

        expected = (
            "y should contain the name of the labels as strings, e.g. "
            "'positive' and 'negative', don't pass label-encoded targets"
        )
        assert str(recwarn.list[0].message) == expected

    def test_fit_X_and_y_not_matching_raises(
            self, model, tokenizer, classifier_cls, X, y
    ):
        # in contrast to zero-shot, few-shot requires X and y to have the same
        # number of samples
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        msg = (
            "X and y don't have the same number of samples, found 2 and 3 samples, "
            "respectively"
        )
        with pytest.raises(ValueError, match=msg):
            clf.fit(X[:2], y[:3])

    def test_predict(self, model, tokenizer, classifier_cls, X, X_test, y_test):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(X, ['positive', 'negative', 'positive'])
        y_pred = clf.predict(X_test)
        np.testing.assert_array_equal(y_pred, y_test)

    def test_predict_proba(self, model, tokenizer, classifier_cls, X, y):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (3, 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        assert (y_proba >= 0.0).all()
        assert (y_proba <= 1.0).all()

    def test_init_from_model_name(self, classifier_cls, X, y):
        clf = classifier_cls('google/flan-t5-small', use_caching=False)
        # check that none of the below raise
        clf.fit(X, y)
        clf.predict_proba(X)
        clf.predict(X)

    def test_proba_for_unlikely_label_low(
            self, model, tokenizer, classifier_cls, X, y, X_test
    ):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        X = X[:] + ["This movie is a zip"]
        y = y.tolist() + ['zip']
        clf.fit(X, y)
        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (3, 3)
        # predictions for class 'zip' should be low
        assert (y_proba[:, -1] < 1e-3).all()

    def test_predict_proba_not_normalized(self, model, tokenizer, classifier_cls, X, y):
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, probas_sum_to_1=False
        )
        # the classes have different number of tokens
        clf.fit(X, y)
        y_proba = clf.predict_proba(X)
        assert (y_proba.sum(axis=1) < 1.0).all()

    def test_same_X_same_probas(self, model, tokenizer, classifier_cls, X, y):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(X, y)

        y_proba = clf.predict_proba(X)
        y_proba2 = clf.predict_proba(X)
        y_proba3 = clf.predict_proba(X)

        np.testing.assert_allclose(y_proba, y_proba2)
        np.testing.assert_allclose(y_proba, y_proba3)

    def test_custom_prompt(self, model, tokenizer, classifier_cls, X, y):
        prompt = (
            "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
            "Examples: {examples}\n\n"
        )
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        # just checking that this works, we don't necessarily expect the
        # predictions to be correct
        clf.fit(X, y)
        clf.predict_proba(X)
        clf.predict(X)

    def test_defective_prompt_missing_keys_raises(
            self, model, tokenizer, classifier_cls, X, y, recwarn
    ):
        # the prompt has no 'examples' placeholders
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt may not be correct, it expects 3 "
            "placeholders: 'examples', 'labels', 'text', missing keys: "
            "'examples'"
        )
        clf.fit(X, y)
        assert str(recwarn.list[0].message) == msg

    def test_defective_prompt_extra_keys_raises(
            self, model, tokenizer, classifier_cls, X, y, recwarn
    ):
        # the prompt has extra 'foo' and 'bar' placeholders
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        prompt += "foo: {foo}\n\nExamples: {examples}\n\nbar: {bar}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt may not be correct, it expects 3 "
            "placeholders: 'examples', 'labels', 'text', extra keys: "
            "'bar', 'foo'"
        )
        clf.fit(X, y)
        assert str(recwarn.list[0].message) == msg

    def test_get_prompt(self, classifier_cls, model, tokenizer):
        prompt = "Foo {labels} bar {text} baz {examples}"
        clf = classifier_cls(
            model=model,
            tokenizer=tokenizer,
            use_caching=False,
            prompt=prompt,
            random_state=0,
        )
        clf.fit(['example-1', 'example-2'], ['label-a', 'label-b'])
        x = "My input"
        expected = (
            "Foo ['label-a', 'label-b'] bar My input baz "
            "```\nexample-2\n```\n\nYour response:\nlabel-b\n\n"
            "```\nexample-1\n```\n\nYour response:\nlabel-a\n"
        )
        assert clf.get_prompt(x) == expected

    def test_repr(self, classifier_cls, model, tokenizer, X, y):
        expected = (
            "FewShotClassifier(model='T5ForConditionalGeneration', "
            "tokenizer='T5Tokenizer', use_caching=False)"
        )

        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        assert str(clf) == expected
        assert repr(clf) == expected

        clf.fit(X, y)
        assert str(clf) == expected
        assert repr(clf) == expected

    def test_get_examples_more_samples_than_X(
            self, classifier_cls, model, tokenizer, X, y
    ):
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(X, y)
        examples = clf.get_examples(X, y, n_samples=10)

        # all X's and y's are included in the examples
        assert len(examples) == len(X)
        assert sorted(examples) == sorted(zip(X, y))

    def test_get_examples_fewer_samples_than_X(
            self, classifier_cls, model, tokenizer, X, y
    ):
        # there are fewer examples than X or even unique labels
        clf = classifier_cls(model=model, tokenizer=tokenizer, use_caching=False)
        clf.fit(X, y)
        examples = clf.get_examples(X, y, n_samples=1)

        assert len(examples) == 1
        assert set(map(tuple, examples)).issubset(set(zip(X, y)))

    def test_get_examples_deterministic(self, classifier_cls, model, tokenizer, X, y):
        # as long as we set random_state, the examples should be chosen
        # deterministically
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, random_state=0
        )
        clf.fit(X, y)
        examples = [clf.get_examples(X, y, 3) for _ in range(10)]

        first = examples[0]
        for other in examples[1:]:
            assert first == other

    def test_get_works_with_non_str_data(self, classifier_cls, model, tokenizer, y):
        # even if not recommended, X should also be allowed to be non-string
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, random_state=0
        )
        X = list(range(len(y)))
        clf.fit(X, y)
        examples = [clf.get_examples(X, y, 3) for _ in range(10)]

        first = examples[0]
        for other in examples[1:]:
            assert first == other
