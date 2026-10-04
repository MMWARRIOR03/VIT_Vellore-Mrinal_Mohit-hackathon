"""
DeBERTa-based financial event classifier (Task 6).

Wraps a fine-tuned microsoft/deberta-v3-base checkpoint for 10-class
financial event classification.  Falls back to a rule-based classifier
(or a fixed default) when the model checkpoint has not been downloaded or
training has not been run yet.

Typical usage
─────────────
    from app.engine.deberta_classifier import DeBERTaEventClassifier
    from app.engine.event_classifier import RuleBasedEventClassifier
    from pathlib import Path

    fallback = RuleBasedEventClassifier()
    classifier = DeBERTaEventClassifier(
        model_path=Path("data/models/event_deberta"),
        fallback_classifier=fallback,
    )

    event_type, probs = classifier.predict("Apple beats Q3 earnings estimates")

Model availability
──────────────────
``is_available()`` returns True only when the checkpoint was loaded
successfully.  Callers should treat any output from an unavailable model
as coming from the fallback path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# The 10 event categories — same ordering used during fine-tuning.
# Index 0 → label 0, so label2id and id2label must match this list.
EVENT_TYPES: list[str] = [
    "Earnings",
    "M&A",
    "Regulatory",
    "Macro",
    "Geopolitical",
    "Credit",
    "Commodity",
    "Cyber",
    "Leadership",
    "Product",
]

_NUM_LABELS: int = len(EVENT_TYPES)
_MAX_TOKENS: int = 256
_BATCH_SIZE: int = 8

# Sentinel returned when the model is not loaded and no fallback is provided
_DEFAULT_FALLBACK_LABEL: str = "Macro"
_DEFAULT_FALLBACK_PROB: float = 0.5


class DeBERTaEventClassifier:
    """
    Fine-tuned DeBERTa-v3-base classifier for 10-class financial event typing.

    When a valid checkpoint directory is supplied and torch/transformers are
    available, inference uses the neural model.  In all other cases the call
    is forwarded to ``fallback_classifier.get_primary()`` if one was supplied,
    or a fixed ("Macro", 0.5) stub is returned.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        fallback_classifier=None,
    ) -> None:
        self._loaded: bool = False
        self._tokenizer = None
        self._model = None
        self._fallback = fallback_classifier

        if model_path is not None and Path(model_path).exists():
            self.load(Path(model_path))

    # ── Model loading ──────────────────────────────────────────────────────────

    def load(self, model_path: Path) -> bool:
        """
        Load the tokenizer and model from ``model_path``.

        Returns True on success, False on any failure.  On failure a warning
        is logged and the instance remains in fallback mode.
        """
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )

            logger.info("Loading DeBERTa event classifier from %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path)
            )
            self._model.eval()
            self._loaded = True
            logger.info("DeBERTa event classifier loaded — %d labels", _NUM_LABELS)
            return True

        except Exception as exc:
            logger.warning(
                "Failed to load DeBERTa classifier from %s: %s — using fallback",
                model_path,
                exc,
            )
            self._loaded = False
            return False

    # ── Availability ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the DeBERTa model was loaded successfully."""
        return self._loaded

    # ── Single prediction ──────────────────────────────────────────────────────

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Classify ``text`` into one of the 10 financial event categories.

        Returns ``(event_type, probs_dict)`` where ``probs_dict`` maps every
        event label to a probability and all values sum to approximately 1.0.

        When the model is not loaded:
          - Delegates to ``fallback_classifier.get_primary()`` if available.
          - Otherwise returns ("Macro", {"Macro": 0.5, <others>: small}).
        """
        if not self._loaded:
            return self._fallback_predict(text)

        results = self._predict_batch_internal([text])
        return results[0]

    # ── Batch prediction ───────────────────────────────────────────────────────

    def predict_batch(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]:
        """
        Classify a list of texts.  Uses batched inference with batch_size=8.

        Returns a list of ``(event_type, probs_dict)`` tuples in input order.
        Falls back gracefully for individual items if inference errors.
        """
        if not self._loaded:
            return [self._fallback_predict(t) for t in texts]

        results: list[tuple[str, dict[str, float]]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            chunk = texts[start : start + _BATCH_SIZE]
            try:
                results.extend(self._predict_batch_internal(chunk))
            except Exception as exc:
                logger.error(
                    "DeBERTa batch inference error (offset %d): %s — falling back",
                    start,
                    exc,
                )
                results.extend(self._fallback_predict(t) for t in chunk)

        return results

    # ── Internal inference ─────────────────────────────────────────────────────

    def _predict_batch_internal(
        self, texts: list[str]
    ) -> list[tuple[str, dict[str, float]]]:
        """
        Run a single batch through the loaded model.

        Assumes ``self._loaded`` is True.  Callers are responsible for
        splitting into chunks of at most ``_BATCH_SIZE``.
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        # Handle empty / whitespace-only strings by substituting a placeholder
        safe_texts = [t if t and t.strip() else "[EMPTY]" for t in texts]

        inputs = self._tokenizer(
            safe_texts,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_TOKENS,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.logits shape: (batch, num_labels)
        probs_tensor = F.softmax(outputs.logits, dim=-1)  # (batch, 10)

        results: list[tuple[str, dict[str, float]]] = []
        for i in range(probs_tensor.shape[0]):
            prob_list = probs_tensor[i].tolist()
            probs_dict = {label: float(p) for label, p in zip(EVENT_TYPES, prob_list)}
            best_label = max(probs_dict, key=lambda k: probs_dict[k])
            results.append((best_label, probs_dict))

        return results

    # ── Fallback path ──────────────────────────────────────────────────────────

    def _fallback_predict(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Delegate to the rule-based fallback classifier, or return a fixed stub.
        """
        if self._fallback is not None:
            try:
                return self._fallback.get_primary(text)
            except Exception as exc:
                logger.warning("Fallback classifier error: %s", exc)

        # Stub: uniform low probability with Macro as the default winner
        stub_prob = round((1.0 - _DEFAULT_FALLBACK_PROB) / (_NUM_LABELS - 1), 6)
        probs = {label: stub_prob for label in EVENT_TYPES}
        probs[_DEFAULT_FALLBACK_LABEL] = _DEFAULT_FALLBACK_PROB
        return _DEFAULT_FALLBACK_LABEL, probs
