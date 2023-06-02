"""Tests for skorch.llm.classifier"""

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

    def test_classes(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['positive', 'negative', 'very positive', 'foobar'])
        # classes_ are sorted
        expected = np.array(['foobar', 'negative', 'positive', 'very positive'])
        np.testing.assert_equal(clf.classes_, expected)

    def test_predict(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['negative', 'positive'])
        X = [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]
        y_pred = clf.predict(X)
        np.testing.assert_array_equal(
            y_pred,
            np.array(['positive', 'negative', 'positive']),
        )

    def test_predict_proba(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['positive', 'negative'])

        X = [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (3, 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        assert (y_proba >= 0.0).all()
        assert (y_proba <= 1.0).all()

    def test_proba_for_unlikely_label_low(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['positive', 'negative', 'zip'])

        X = [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]
        y_proba = clf.predict_proba(X)
        assert y_proba.shape == (3, 3)
        assert (y_proba[:, -1] < 1e-3).all()

    def test_predict_proba_labels_differing_num_tokens(
            self, model, tokenizer, classifier_cls
    ):
        # positive and negative have 1 token, foobar has 3 tokens
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['foobar', 'positive', 'negative'])
        X = [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]

        y_proba = clf.predict_proba(X)
        # foobar is column 0
        assert (y_proba[:, 0] < 1e-3).all()

    def test_predict_proba_not_normalized(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer, probas_sum_to_1=False)
        # the classes have different number of tokens
        clf.fit(None, ['negative', 'positive'])
        X = [
            "A masterpiece, instant classic, 5 stars out of 5",
            "I was bored. Would not recommend.",
            "My friends and I really enjoyed this one. Best time of my life",
        ]
        y_proba = clf.predict_proba(X)
        assert (y_proba.sum(axis=1) < 1.0).all()

    def test_same_X_same_probas(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['foobar'])
        X = ["A masterpiece, instant classic, 5 stars out of 5"]

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
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        clf.fit(None, ['foobar'])
        X = ["A masterpiece, instant classic, 5 stars out of 5"]

        y_proba = clf.predict_proba(X)
        assert clf.cached_model_._uncached_calls == 1
        assert clf.cached_model_._cached_calls == 0

        y_proba2 = clf.predict_proba(X)
        assert clf.cached_model_._uncached_calls == 1
        assert clf.cached_model_._cached_calls == 4  # 4 tokens
        np.testing.assert_allclose(y_proba, y_proba2)

        y_proba3 = clf.predict_proba(X)
        assert clf.cached_model_._uncached_calls == 1
        assert clf.cached_model_._cached_calls == 8  # 4 more tokens
        np.testing.assert_allclose(y_proba, y_proba3)

    def test_caching_is_faster(self, model, tokenizer, classifier_cls):
        clf = classifier_cls(model=model, tokenizer=tokenizer)
        # classes should have a long common prefix for caching to make a
        # difference
        X = ["A masterpiece, instant classic, 5 stars out of 5"]
        y = ['absolutely undoubtedly positive', 'absolutely undoubtedly negative']
        clf.fit(X=None, y=y)

        # measure time for uncached call using timeit
        uncached_time = timeit.timeit(lambda: clf.predict_proba(X), number=1)
        cached_time = timeit.timeit(lambda: clf.predict_proba(X), number=1)
        # at least 1/3 faster
        assert cached_time < 2/3 * uncached_time
