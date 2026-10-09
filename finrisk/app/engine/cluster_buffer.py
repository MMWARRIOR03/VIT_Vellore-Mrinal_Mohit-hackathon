"""
cluster_buffer.py — Rolling buffer for cluster-level FinBERT sentiment aggregation.

When 3+ articles about the same event arrive in a short window, aggregating
their scores prevents the "same story 20 times" problem and makes diffusion
scores more meaningful.

Cluster identity: cluster_id is the same SHA256-based key used by the NLP
worker (ticker + event_type + calendar day).  Entries older than max_age_hours
are pruned by cleanup_old().

Thread safety: NOT thread-safe — designed for a single-threaded asyncio worker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union


class ClusterSentimentBuffer:
    """
    Rolling buffer that aggregates FinBERT sentiment scores for articles sharing
    the same cluster_id (same ticker + event_type + calendar day).

    Purpose: when 3+ articles about the same event hit in a short window, score
    them as a cluster rather than independently. This prevents the "same story
    20 times" problem and makes diffusion scores more meaningful.
    """

    def __init__(self, max_age_hours: float = 2.0) -> None:
        # buffer: dict[cluster_id, list[tuple[sentiment, confidence, source_weight, ts]]]
        self._buffer: dict[
            str, list[tuple[float, float, float, datetime]]
        ] = {}
        self._max_age_hours = max_age_hours

    # ── Public interface ──────────────────────────────────────────────────────

    def add(
        self,
        cluster_id: str,
        sentiment: float,
        confidence: float,
        source_weight: float,
        ts: datetime,
    ) -> None:
        """Add an article's sentiment to the cluster buffer."""
        if cluster_id not in self._buffer:
            self._buffer[cluster_id] = []
        self._buffer[cluster_id].append((sentiment, confidence, source_weight, ts))

    def get_aggregate(
        self, cluster_id: str
    ) -> tuple[float, float]:
        """
        Return (weighted_avg_sentiment, combined_confidence) for the cluster.

        weighted_avg_sentiment:
            sum(s * c * w for s, c, w in entries) / sum(c * w for c, w in entries)
            Falls back to the raw sentiment of the single entry if only one entry.

        combined_confidence:
            min(1.0, base_conf + 0.05 * (n - 1))
            where base_conf is the confidence of the first entry and n = len(entries).
            Confidence increases with more corroborating sources.

        Returns (0.0, 0.0) if the cluster is not in the buffer.
        """
        entries = self._buffer.get(cluster_id)
        if not entries:
            return (0.0, 0.0)

        n = len(entries)

        if n == 1:
            sentiment, confidence, _w, _ts = entries[0]
            return (sentiment, confidence)

        # Weighted average sentiment: weight = confidence * source_weight
        numerator = sum(s * c * w for s, c, w, _ts in entries)
        denominator = sum(c * w for _s, c, w, _ts in entries)
        weighted_avg = numerator / denominator if denominator != 0.0 else 0.0

        # Combined confidence: grows with more corroborating sources
        base_conf = entries[0][1]
        combined_conf = min(1.0, base_conf + 0.05 * (n - 1))

        return (weighted_avg, combined_conf)

    def cleanup_old(self, max_age_hours: Union[float, None] = None) -> int:
        """
        Remove entries older than max_age_hours.
        Returns the number of clusters removed.
        Uses the instance default if max_age_hours is None.
        """
        effective_age = max_age_hours if max_age_hours is not None else self._max_age_hours
        now = datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - effective_age * 3600.0

        stale = [
            cid
            for cid, entries in self._buffer.items()
            if entries and all(entry[3].timestamp() < cutoff for entry in entries)
        ]
        for cid in stale:
            del self._buffer[cid]

        return len(stale)

    def get_source_count(self, cluster_id: str) -> int:
        """Return the number of articles currently in the cluster buffer."""
        return len(self._buffer.get(cluster_id, []))
