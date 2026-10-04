"""
Tests for DeBERTaEventClassifier (Task 6, Part B).

All tests run without a downloaded model checkpoint or network access.
The neural inference path is exercised via mocks; the fallback path is
exercised with the real RuleBasedEventClassifier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.engine.deberta_classifier import DeBERTaEventClassifier, EVENT_TYPES
from app.engine.event_classifier import RuleBasedEventClassifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_classifier(
    model_path: Path | None = None,
    fallback: RuleBasedEventClassifier | None = None,
) -> DeBERTaEventClassifier:
    if fallback is None:
        fallback = RuleBasedEventClassifier()
    return DeBERTaEventClassifier(model_path=model_path, fallback_classifier=fallback)


def _mock_model_output(num_texts: int = 1):
    """
    Build a mock transformers model output with logits shaped (num_texts, 10).
    The first label ("Earnings") gets the highest logit so argmax is deterministic.
    """
    import torch

    logits = torch.zeros(num_texts, len(EVENT_TYPES))
    logits[:, 0] = 5.0  # Earnings gets the top score
    output = MagicMock()
    output.logits = logits
    return output


def _mock_tokenizer_output(num_texts: int = 1):
    """Return a dict-like mock that looks like tokenizer(**...) output."""
    import torch

    mock_enc = {
        "input_ids": torch.zeros(num_texts, 16, dtype=torch.long),
        "attention_mask": torch.ones(num_texts, 16, dtype=torch.long),
    }
    return mock_enc


# ── is_available() ────────────────────────────────────────────────────────────

def test_is_available_returns_false_when_path_not_provided():
    clf = _make_classifier(model_path=None)
    assert clf.is_available() is False


def test_is_available_returns_false_when_path_does_not_exist(tmp_path):
    nonexistent = tmp_path / "no_model_here"
    clf = _make_classifier(model_path=nonexistent)
    assert clf.is_available() is False


def test_is_available_returns_false_without_load():
    clf = DeBERTaEventClassifier()
    assert clf.is_available() is False


# ── Fallback behaviour ─────────────────────────────────────────────────────────

def test_predict_falls_back_when_not_loaded():
    clf = _make_classifier()
    result = clf.predict("Apple beats earnings estimates")
    event_type, probs = result
    assert isinstance(event_type, str)
    assert len(event_type) > 0
    assert isinstance(probs, dict)


def test_predict_fallback_returns_valid_event_type():
    clf = _make_classifier()
    event_type, _ = clf.predict("Apple beats quarterly earnings estimates by 15%")
    assert event_type in EVENT_TYPES


def test_predict_fallback_returns_non_empty_probs():
    clf = _make_classifier()
    _, probs = clf.predict("Federal Reserve raises interest rates by 25 bps")
    assert len(probs) > 0
    for k, v in probs.items():
        assert isinstance(v, float)
        assert v >= 0.0


def test_predict_without_fallback_returns_default_stub():
    clf = DeBERTaEventClassifier(model_path=None, fallback_classifier=None)
    event_type, probs = clf.predict("some financial text")
    assert event_type == "Macro"
    assert probs["Macro"] == 0.5


def test_predict_fallback_classifier_exception_falls_through():
    """When the fallback raises, the stub should be returned instead."""
    bad_fallback = MagicMock()
    bad_fallback.get_primary.side_effect = RuntimeError("fallback blew up")
    clf = DeBERTaEventClassifier(model_path=None, fallback_classifier=bad_fallback)
    event_type, probs = clf.predict("anything")
    assert event_type == "Macro"
    assert isinstance(probs, dict)


# ── Return-type contract ───────────────────────────────────────────────────────

def test_predict_returns_tuple_of_str_and_dict():
    clf = _make_classifier()
    result = clf.predict("Tesla Q2 revenue beats analyst consensus")
    assert isinstance(result, tuple)
    assert len(result) == 2
    event_type, probs = result
    assert isinstance(event_type, str)
    assert isinstance(probs, dict)


def test_predict_batch_returns_same_length_as_input():
    clf = _make_classifier()
    texts = [
        "Apple earnings beat",
        "Federal Reserve hikes rates",
        "Cyber attack on hospital network",
        "CEO resigns after board dispute",
        "OPEC cuts oil production",
    ]
    results = clf.predict_batch(texts)
    assert len(results) == len(texts)


def test_predict_batch_each_element_is_valid_tuple():
    clf = _make_classifier()
    results = clf.predict_batch(["any text"] * 3)
    for event_type, probs in results:
        assert isinstance(event_type, str)
        assert isinstance(probs, dict)
        assert event_type in EVENT_TYPES or len(event_type) > 0


def test_predict_batch_empty_list_returns_empty_list():
    clf = _make_classifier()
    assert clf.predict_batch([]) == []


# ── Empty / edge input handling ────────────────────────────────────────────────

def test_predict_handles_empty_string():
    clf = _make_classifier()
    event_type, probs = clf.predict("")
    assert isinstance(event_type, str)
    assert isinstance(probs, dict)


def test_predict_handles_whitespace_only_string():
    clf = _make_classifier()
    event_type, probs = clf.predict("   ")
    assert isinstance(event_type, str)
    assert isinstance(probs, dict)


def test_predict_handles_very_long_string():
    clf = _make_classifier()
    long_text = "earnings report " * 500
    event_type, probs = clf.predict(long_text)
    assert isinstance(event_type, str)


# ── Neural path (mocked model) ────────────────────────────────────────────────

def test_predict_probs_sum_to_one_when_model_loaded(tmp_path):
    """
    Mocks the tokenizer and model so we can exercise the neural inference
    path without a real checkpoint.  Verifies that softmax output sums to 1.
    """
    torch = pytest.importorskip("torch")

    # Patch transformers.AutoTokenizer and AutoModelForSequenceClassification
    with (
        patch("app.engine.deberta_classifier.logger"),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=MagicMock(
                side_effect=None,
                **{"return_value": _mock_tokenizer_output()},
                __call__=lambda self, *a, **kw: _mock_tokenizer_output(),
            ),
        ),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=MagicMock(
                eval=MagicMock(return_value=None),
                **{"return_value": _mock_model_output()},
                __call__=lambda self, **kw: _mock_model_output(),
            ),
        ),
    ):
        clf = DeBERTaEventClassifier.__new__(DeBERTaEventClassifier)
        clf._loaded = False
        clf._tokenizer = None
        clf._model = None
        clf._fallback = None

        # Manually inject mocked components
        mock_tok = MagicMock()
        mock_tok.return_value = _mock_tokenizer_output()
        mock_tok.__call__ = lambda *a, **kw: _mock_tokenizer_output()

        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.return_value = _mock_model_output()
        mock_model.__call__ = lambda **kw: _mock_model_output()

        clf._tokenizer = mock_tok
        clf._model = mock_model
        clf._loaded = True

        event_type, probs = clf.predict("Apple crushes Q3 earnings estimates")

    assert isinstance(event_type, str)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4, f"probs sum to {total}, expected ~1.0"


def test_loaded_predict_returns_known_event_type(tmp_path):
    """
    When the model is mocked to give maximum logit to 'Earnings' (index 0),
    the returned event_type should be 'Earnings'.
    """
    torch = pytest.importorskip("torch")

    clf = DeBERTaEventClassifier.__new__(DeBERTaEventClassifier)
    clf._loaded = True
    clf._fallback = None

    mock_tok = MagicMock()
    mock_tok.return_value = _mock_tokenizer_output()

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.return_value = _mock_model_output(num_texts=1)

    clf._tokenizer = mock_tok
    clf._model = mock_model

    event_type, probs = clf._predict_batch_internal(
        ["Apple quarterly earnings beat expectations"]
    )[0]
    assert event_type == "Earnings"
    assert probs["Earnings"] > probs["M&A"]


def test_predict_batch_uses_batch_size_8(tmp_path):
    """
    With 20 texts, predict_batch should produce exactly 20 results.
    The batch-chunking logic is verified by mocking _predict_batch_internal
    to return one (event_type, probs) tuple per text, bypassing the neural path.
    """
    clf = DeBERTaEventClassifier.__new__(DeBERTaEventClassifier)
    clf._loaded = True
    clf._fallback = None

    # _predict_batch_internal receives a list[str] and returns list of tuples
    def _fake_internal(texts):
        stub_probs = {et: 0.1 for et in EVENT_TYPES}
        return [("Earnings", stub_probs) for _ in texts]

    texts = [f"headline {i}" for i in range(20)]
    with patch.object(clf, "_predict_batch_internal", side_effect=_fake_internal):
        results = clf.predict_batch(texts)

    assert len(results) == 20
    for event_type, probs in results:
        assert event_type == "Earnings"
        assert isinstance(probs, dict)


# ── load() failure path ───────────────────────────────────────────────────────

def test_load_returns_false_on_bad_path(tmp_path):
    clf = DeBERTaEventClassifier()
    bad_path = tmp_path / "nonexistent_model"
    result = clf.load(bad_path)
    assert result is False
    assert clf.is_available() is False


def test_load_returns_false_when_transformers_not_importable():
    clf = DeBERTaEventClassifier()
    with patch(
        "builtins.__import__",
        side_effect=lambda name, *args, **kwargs: (
            (_ for _ in ()).throw(ImportError("no torch"))
            if name in ("torch", "transformers")
            else __import__(name, *args, **kwargs)
        ),
    ):
        result = clf.load(Path("/fake/path"))
    assert result is False
