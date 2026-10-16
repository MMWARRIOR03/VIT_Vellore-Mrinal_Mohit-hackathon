"""
Sentiment analysis using FinBERT (ProsusAI/finbert).

Falls back to a VADER-inspired rule-based scorer when the transformers library
is not installed or the model files have not been downloaded yet.  The fallback
is finance-aware: it weights domain-specific positive/negative terms rather
than relying on the general-purpose VADER lexicon.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Model cache directory — absolute path pointing to finrisk/data/models/
_MODEL_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
_FINBERT_MODEL_ID = "ProsusAI/finbert"

# ── Transformers availability flag ────────────────────────────────────────────

try:
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers/torch not available — FinBERTSentimentModel will use "
        "the rule-based fallback scorer."
    )

if TYPE_CHECKING:
    import torch as _torch
    from transformers import (
        AutoTokenizer as _AutoTokenizer,
        AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
        PreTrainedTokenizer,
        PreTrainedModel,
    )


# ── Finance-aware VADER-style lexicon ─────────────────────────────────────────

_POSITIVE_TERMS: dict[str, float] = {
    "beat": 0.8, "beats": 0.8, "exceeded": 0.7, "exceed": 0.7,
    "record": 0.6, "surged": 0.7, "surge": 0.6, "growth": 0.5,
    "profit": 0.6, "profits": 0.6, "revenue growth": 0.7,
    "raised guidance": 0.9, "raised its guidance": 0.9,
    "strong": 0.5, "upgrade": 0.7, "upgraded": 0.7,
    "outperform": 0.6, "buy": 0.5, "bullish": 0.7,
    "approval": 0.6, "approved": 0.6, "recovery": 0.5,
    "positive": 0.5, "gain": 0.5, "gains": 0.5, "rally": 0.6,
    "expansion": 0.5, "investment": 0.4, "deal": 0.4,
    "dividend": 0.5, "buyback": 0.5, "share repurchase": 0.6,
    "partnership": 0.4, "collaboration": 0.4, "win": 0.5,
}

_NEGATIVE_TERMS: dict[str, float] = {
    "miss": 0.7, "missed": 0.7, "misses": 0.7,
    "loss": 0.6, "losses": 0.6, "decline": 0.6, "declined": 0.6,
    "fall": 0.5, "fell": 0.5, "drop": 0.6, "dropped": 0.6,
    "cut guidance": 0.9, "lowered guidance": 0.9, "below expectations": 0.8,
    "fine": 0.7, "fined": 0.7, "penalty": 0.7, "sanction": 0.7,
    "lawsuit": 0.6, "litigation": 0.6, "investigation": 0.6,
    "default": 0.8, "bankrupt": 0.9, "bankruptcy": 0.9,
    "downgrade": 0.7, "downgraded": 0.7, "sell": 0.5, "bearish": 0.7,
    "hack": 0.7, "breach": 0.7, "ransomware": 0.8, "cyberattack": 0.8,
    "recall": 0.6, "ban": 0.6, "banned": 0.6, "seized": 0.5,
    "resign": 0.5, "departure": 0.4, "fired": 0.6, "terminated": 0.5,
    "layoff": 0.6, "layoffs": 0.6, "restructuring": 0.5,
    "war": 0.7, "conflict": 0.6, "tariff": 0.6, "sanction": 0.7,
    "negative": 0.5, "concern": 0.4, "risk": 0.3, "uncertain": 0.4,
}

_NEGATION_WORDS: frozenset[str] = frozenset(
    ["not", "no", "never", "neither", "nor", "without", "n't", "hardly", "barely"]
)

_NEGATION_WINDOW = 3  # words


def _rule_based_sentiment(text: str) -> tuple[float, float]:
    """
    Finance-aware rule-based sentiment scorer.

    Returns (sentiment_score, confidence) where sentiment is in [-1, 1] and
    confidence reflects how many weighted term hits were found.
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    # Also build a set of bigrams for multi-word term matching
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]

    pos_score = 0.0
    neg_score = 0.0

    # Check trigrams first (most specific), then bigrams, then unigrams
    matched_positions: set[int] = set()

    for phrase, weight in {**{t: v for t, v in _POSITIVE_TERMS.items() if len(t.split()) == 3},
                            **{t: v for t, v in _NEGATIVE_TERMS.items() if len(t.split()) == 3}}.items():
        if phrase in trigrams:
            idx = trigrams.index(phrase)
            if idx not in matched_positions:
                if phrase in _POSITIVE_TERMS:
                    pos_score += _POSITIVE_TERMS[phrase]
                else:
                    neg_score += _NEGATIVE_TERMS[phrase]
                matched_positions.update([idx, idx + 1, idx + 2])

    for phrase, weight in {**{t: v for t, v in _POSITIVE_TERMS.items() if len(t.split()) == 2},
                            **{t: v for t, v in _NEGATIVE_TERMS.items() if len(t.split()) == 2}}.items():
        if phrase in bigrams:
            idx = bigrams.index(phrase)
            if idx not in matched_positions:
                if phrase in _POSITIVE_TERMS:
                    pos_score += _POSITIVE_TERMS[phrase]
                else:
                    neg_score += _NEGATIVE_TERMS[phrase]
                matched_positions.update([idx, idx + 1])

    for i, word in enumerate(words):
        if i in matched_positions:
            continue
        # Check for negation window before this word
        window_start = max(0, i - _NEGATION_WINDOW)
        window = words[window_start:i]
        negated = any(neg in window for neg in _NEGATION_WORDS)

        if word in _POSITIVE_TERMS:
            if negated:
                neg_score += _POSITIVE_TERMS[word] * 0.5
            else:
                pos_score += _POSITIVE_TERMS[word]
        elif word in _NEGATIVE_TERMS:
            if negated:
                pos_score += _NEGATIVE_TERMS[word] * 0.5
            else:
                neg_score += _NEGATIVE_TERMS[word]

    total = pos_score + neg_score
    if total < 1e-6:
        # No signal found — neutral with low confidence
        return 0.0, 0.3

    # Normalise to [-1, 1]
    raw_score = (pos_score - neg_score) / total
    # Confidence scales with signal density relative to text length
    word_count = max(len(words), 1)
    density = min(total / word_count, 1.0)
    confidence = 0.3 + 0.5 * density + 0.2 * min(total / 3.0, 1.0)
    confidence = max(0.0, min(1.0, confidence))

    return float(raw_score), float(confidence)


def _finbert_embedding_from_hidden(hidden_state) -> list[float]:
    """Extract a mean-pooled embedding from the last hidden state tensor."""
    import torch
    # hidden_state shape: (1, seq_len, hidden_size) → mean over seq_len
    embedding = hidden_state[0].mean(dim=0)  # (hidden_size,)
    return embedding.detach().cpu().tolist()


# ── Main model class ──────────────────────────────────────────────────────────

class FinBERTSentimentModel:
    """
    Sentiment model backed by ProsusAI/finbert.

    The model is loaded lazily on the first call to ``predict()`` or
    ``predict_batch()``.  If transformers is not installed, all calls fall
    through to the rule-based scorer.

    Model files are cached under ``data/models/`` to avoid repeated downloads.
    """

    LABELS: list[str] = ["positive", "negative", "neutral"]
    MAX_TOKENS: int = 512
    BATCH_SIZE: int = 8

    def __init__(
        self,
        model_id: str = _FINBERT_MODEL_ID,
        cache_dir: Path | str | None = None,
    ) -> None:
        self._model_id = model_id
        self._cache_dir = Path(cache_dir) if cache_dir else _MODEL_CACHE_DIR
        self._tokenizer = None
        self._model = None
        self._loaded = False

    # ── Model availability ────────────────────────────────────────────────────

    def has_model(self) -> bool:
        """
        Return True if the FinBERT model files exist in the cache directory.
        Does NOT attempt a download.
        """
        if not _TRANSFORMERS_AVAILABLE:
            return False
        model_dir = self._cache_dir / "models--ProsusAI--finbert"
        return model_dir.exists()

    def _load(self) -> bool:
        """
        Attempt to load the FinBERT tokenizer and model.  Returns True on
        success, False if loading fails (e.g. files not yet downloaded).
        """
        if self._loaded:
            return True
        if not _TRANSFORMERS_AVAILABLE:
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("Loading FinBERT from %s (cache: %s)", self._model_id, self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Try loading from cache first — never go to network if files exist
            _load_kwargs = {
                "cache_dir": str(self._cache_dir),
                "local_files_only": True,
            }
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, **_load_kwargs
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id,
                    output_hidden_states=True,
                    **_load_kwargs,
                )
            except Exception:
                # Cache miss — allow one network download, then it will be cached
                logger.info("FinBERT not in cache — downloading once...")
                _load_kwargs["local_files_only"] = False
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, **_load_kwargs
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_id,
                    output_hidden_states=True,
                    **_load_kwargs,
                )
            self._model.eval()  # inference mode
            self._loaded = True
            logger.info("FinBERT loaded successfully (CPU)")
            return True
        except Exception as exc:
            logger.warning("Failed to load FinBERT: %s — using rule-based fallback", exc)
            return False

    # ── Single prediction ─────────────────────────────────────────────────────

    def predict(self, text: str) -> tuple[float, float]:
        """
        Predict sentiment for a single text string.

        Returns (sentiment_score, confidence):
          - sentiment_score ∈ [-1, 1]: P(positive) - P(negative)
          - confidence: max of the three class probabilities
        """
        if not self._load():
            return _rule_based_sentiment(text)

        results = self.predict_batch([text])
        return results[0]

    # ── Batch prediction ──────────────────────────────────────────────────────

    def predict_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        """
        Predict sentiment for a list of texts.  Processes in sub-batches of
        BATCH_SIZE (≤ 8) to bound memory usage.

        Returns a list of (sentiment_score, confidence) tuples in input order.
        """
        if not self._load():
            return [_rule_based_sentiment(t) for t in texts]

        import torch
        import torch.nn.functional as F  # noqa: N812

        results: list[tuple[float, float]] = []

        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[batch_start: batch_start + self.BATCH_SIZE]
            try:
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.MAX_TOKENS,
                    padding=True,
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)

                logits = outputs.logits  # shape: (batch, 3)
                probs = F.softmax(logits, dim=-1)  # (batch, 3)

                # FinBERT label order: positive=0, negative=1, neutral=2
                for i in range(probs.shape[0]):
                    p = probs[i].tolist()
                    p_pos, p_neg, p_neu = p[0], p[1], p[2]
                    score = float(p_pos - p_neg)
                    confidence = float(max(p_pos, p_neg, p_neu))
                    results.append((score, confidence))

            except Exception as exc:
                logger.error("FinBERT batch inference error: %s — falling back", exc)
                for t in batch:
                    results.append(_rule_based_sentiment(t))

        return results

    # ── Embedding extraction ──────────────────────────────────────────────────

    def get_embedding(self, text: str) -> list[float]:
        """
        Return a mean-pooled CLS-layer embedding for the text.

        Used by NoveltyScorer to compare semantic similarity between articles.
        Falls back to a simple TF-IDF-inspired bag-of-chars vector when the
        model is not loaded.
        """
        if not self._load():
            return _fallback_embedding(text)

        import torch

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_TOKENS,
                padding=False,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)

            hidden_states = outputs.hidden_states  # tuple of (1, seq, hidden)
            last_hidden = hidden_states[-1]  # (1, seq_len, hidden_size)
            embedding = last_hidden[0].mean(dim=0)  # (hidden_size,)
            return embedding.detach().cpu().tolist()

        except Exception as exc:
            logger.error("FinBERT embedding error: %s — using fallback", exc)
            return _fallback_embedding(text)


# ── Fallback embedding ────────────────────────────────────────────────────────

def _fallback_embedding(text: str, dim: int = 128) -> list[float]:
    """
    Lightweight deterministic embedding for use when FinBERT is unavailable.

    Uses character n-gram hashing into a fixed-dimensional float vector.
    Semantically meaningless beyond rough similarity for duplicate detection.
    """
    import math

    vec = [0.0] * dim
    text_lower = text.lower()
    # Trigram hashing
    for i in range(len(text_lower) - 2):
        trigram = text_lower[i: i + 3]
        h = hash(trigram) % dim
        vec[h] += 1.0

    # L2-normalise
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
