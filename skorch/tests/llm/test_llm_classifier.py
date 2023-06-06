"""Tests for skorch.llm.classifier"""

import re
import timeit

import numpy as np
import pytest


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
            f"ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls(tokenizer=tokenizer)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_without_tokenizer_raises(self, classifier_cls, model):
        msg = (
            f"ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls(model=model)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_with_model_and_model_name_raises(self, classifier_cls, model):
        msg = (
            f"ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls('google/flan-t5-small', model=model)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

    def test_init_with_tokenizer_and_model_name_raises(self, classifier_cls, tokenizer):
        msg = (
            f"ZeroShotClassifier needs to be initialized with either a model name, "
            "or a model & tokenizer, but not both."
        )
        clf = classifier_cls('google/flan-t5-small', tokenizer=tokenizer)
        with pytest.raises(ValueError, match=msg):
            clf.fit(None, ['positive', 'negative'])

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
        assert (y_proba[:, -1] < 1e-3).all()

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
            self, model, tokenizer, classifier_cls
    ):
        # the prompt has no 'labels' placeholders
        prompt = "Please classify my text:\n{text}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt is not correct, it should have exactly 2 "
            "placeholders: 'labels', 'text', missing keys: 'labels'"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            clf.fit(None, ['positive', 'negative'])

    def test_defective_prompt_extra_key_raises(
            self, model, tokenizer, classifier_cls
    ):
        # the prompt has excess 'examples' placeholder
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        prompt += "Examples: {examples}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt is not correct, it should have exactly 2 "
            "placeholders: 'labels', 'text', extra keys: 'examples'"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            clf.fit(None, ['positive', 'negative'])

    def test_gpt2(self, classifier_cls, X):
        name = 'gpt2'
        name = 'bigscience/bloom-560m'
        #name = 'bigcode/santacoder'
        clf = classifier_cls(name, probas_sum_to_1=False)
        clf.fit(None, ['negative', 'positive'])
        clf.predict_proba(X[:3])
        clf.predict(X[:3])


class TestFewShotClassifier:
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
            self, model, tokenizer, classifier_cls, X, y
    ):
        # the prompt has no 'examples' placeholders
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt is not correct, it should have exactly 3 "
            "placeholders: 'examples', 'labels', 'text', missing keys: "
            "'examples'"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            clf.fit(X, y)

    def test_defective_prompt_extra_keys_raises(
            self, model, tokenizer, classifier_cls, X, y
    ):
        # the prompt has extra 'foo' and 'bar' placeholders
        prompt = "Please classify my text:\n{text}\n\nLabels: {labels}\n\n"
        prompt += "foo: {foo}\n\nExamples: {examples}\n\nbar: {bar}\n\n"
        clf = classifier_cls(
            model=model, tokenizer=tokenizer, use_caching=False, prompt=prompt
        )

        msg = (
            "The prompt is not correct, it should have exactly 3 "
            "placeholders: 'examples', 'labels', 'text', extra keys: "
            "'bar', 'foo'"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            clf.fit(X, y)
