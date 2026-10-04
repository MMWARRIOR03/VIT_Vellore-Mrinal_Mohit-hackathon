"""
Novelty scorer for risk signals.

Measures how novel a new signal is relative to recent signals for the same
ticker by comparing embedding cosine similarity.  A score near 1.0 means the
signal is unlike anything seen recently; near 0.0 means it is a near-duplicate.

The in-memory rolling window (last 50 embeddings per ticker) is intentionally
simple: it does not persist across restarts.  On a cold start, the first signal
for any ticker returns 0.8 (moderately novel) because there is nothing to
compare against.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 50
_DEFAULT_NOVELTY = 0.8  # returned when no history exists for a ticker


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns a value in [-1, 1].  Falls back to 0.0 if either vector is
    all-zeros to avoid division by zero.
    """
    if len(a) != len(b):
        # Mismatched dimensions — treat as unrelated
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return dot / (norm_a * norm_b)


class NoveltyScorer:
    """
    In-memory rolling-window novelty scorer.

    One deque of embeddings is maintained per ticker symbol.  The window is
    capped at ``_WINDOW_SIZE`` entries; older embeddings are discarded as new
    ones arrive.
    """

    def __init__(self, window_size: int = _WINDOW_SIZE) -> None:
        self._window_size = window_size
        # ticker → deque of embedding vectors (each a list[float])
        self._history: dict[str, Deque[list[float]]] = {}

    def compute_novelty(
        self,
        text: str,  # kept for API symmetry; not used when embedding is provided
        ticker: str,
        embedding: list[float],
    ) -> float:
        """
        Compute the novelty of a new signal for the given ticker.

        Returns a float in [0.0, 1.0]:
          - 0.0: the embedding is identical to a recent signal (duplicate)
          - 1.0: no similarity to any recent signal (entirely new)
          - 0.8: default when no history exists for this ticker

        The ``text`` parameter is accepted for interface symmetry but is not
        used when a pre-computed ``embedding`` is supplied.
        """
        history = self._history.get(ticker)
        if not history:
            logger.debug("No history for ticker %s — returning default novelty %.2f", ticker, _DEFAULT_NOVELTY)
            return _DEFAULT_NOVELTY

        max_sim = max(
            _cosine_similarity(embedding, past_emb)
            for past_emb in history
        )
        # Clamp similarity to [0, 1] before inverting — cosine can be negative
        # for orthogonal vectors, which would inflate novelty beyond 1.0.
        max_sim_clamped = max(0.0, min(1.0, max_sim))
        novelty = 1.0 - max_sim_clamped
        return float(novelty)

    def add_to_history(self, ticker: str, embedding: list[float]) -> None:
        """
        Add an embedding to the rolling window for the given ticker.

        If the window is full, the oldest entry is dropped.
        """
        if ticker not in self._history:
            self._history[ticker] = deque(maxlen=self._window_size)
        self._history[ticker].append(embedding)

    def window_size_for(self, ticker: str) -> int:
        """Return the number of embeddings currently held for a ticker."""
        return len(self._history.get(ticker, []))
