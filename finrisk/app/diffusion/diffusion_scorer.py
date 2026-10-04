"""
Diffusion scorer — tracks how broadly and quickly a cluster of articles is
spreading across information sources.

Two scoring layers:

  L1 (count + velocity): how many articles have arrived and how fast.
  L2 (exponential decay sum): a time-discounted aggregate that rewards both
      recency and volume. Normalised to [0, 1) via score / (1 + score).

Typical usage
─────────────
    scorer = DiffusionScorer()

    # As each article arrives:
    scorer.record_arrival(cluster_id, ts_published)

    # When building the DiffusionInfo for a RiskSignal:
    diffusion = scorer.compute(cluster_id, source_count=n_sources)

Thread safety: not thread-safe. Use one instance per worker coroutine.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from app.schema.risk_signal import DiffusionInfo

logger = logging.getLogger(__name__)

# Default L2 decay: beta=0.5 → article from 2 hours ago contributes e^(-1) ≈ 0.37
_DEFAULT_BETA: float = 0.5
# Velocity window: look back 2 hours for the article-per-hour rate
_VELOCITY_WINDOW_HOURS: float = 2.0
# History ring buffer capacity per cluster
_MAX_HISTORY_LEN: int = 200


class DiffusionScorer:
    """
    Per-cluster arrival tracker that computes L1 (sources + velocity) and
    L2 (time-discounted score) diffusion metrics.

    The scorer maintains a bounded deque of arrival timestamps per cluster_id.
    Old clusters are evicted by ``cleanup_old_clusters()``, which callers
    should run periodically (e.g. every hour).
    """

    def __init__(self) -> None:
        # cluster_id → deque of UTC-aware datetimes, newest at the right
        self._history: dict[str, deque[datetime]] = {}

    # ── Arrival recording ──────────────────────────────────────────────────────

    def record_arrival(self, cluster_id: str, ts: datetime) -> None:
        """
        Record that an article belonging to ``cluster_id`` arrived at ``ts``.

        ``ts`` should be timezone-aware (UTC).  Naive datetimes are coerced to
        UTC with a warning rather than raising, to keep the hot path resilient.
        """
        if ts.tzinfo is None:
            logger.warning(
                "record_arrival received naive datetime for cluster %s — coercing to UTC",
                cluster_id,
            )
            ts = ts.replace(tzinfo=timezone.utc)

        if cluster_id not in self._history:
            self._history[cluster_id] = deque(maxlen=_MAX_HISTORY_LEN)

        self._history[cluster_id].append(ts)

    # ── L1 metrics ─────────────────────────────────────────────────────────────

    def compute_l1(
        self,
        cluster_id: str,
        now: Optional[datetime] = None,
    ) -> dict[str, float]:
        """
        Compute L1 metrics for the cluster.

        Returns a dict with:
          ``sources`` — total number of arrivals recorded (int, ≥ 0)
          ``velocity`` — articles per hour within the last 2-hour window (float)

        ``now`` defaults to the current UTC time.  Pass an explicit value in
        tests to make behaviour deterministic.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        history = self._history.get(cluster_id, deque())
        total = len(history)

        # Count arrivals within the velocity window
        cutoff = now.timestamp() - _VELOCITY_WINDOW_HOURS * 3600.0
        recent_count = sum(1 for t in history if t.timestamp() >= cutoff)

        velocity = recent_count / _VELOCITY_WINDOW_HOURS if recent_count > 0 else 0.0

        return {"sources": float(total), "velocity": velocity}

    # ── L2 score ───────────────────────────────────────────────────────────────

    def compute_l2(
        self,
        cluster_id: str,
        beta: float = _DEFAULT_BETA,
        now: Optional[datetime] = None,
    ) -> float:
        """
        Compute the L2 exponential-decay diffusion score for the cluster.

        For each recorded arrival at time t_i:
            contribution = exp(-beta * delta_hours_i)
        where delta_hours_i = (now - t_i).total_seconds() / 3600.

        The raw sum is then normalised: score = raw / (1 + raw) → [0, 1).

        A cluster with no history returns 0.0.
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)

        history = self._history.get(cluster_id)
        if not history:
            return 0.0

        now_ts = now.timestamp()
        raw_score = 0.0
        for t in history:
            delta_seconds = now_ts - t.timestamp()
            # Guard against future timestamps (negative delta)
            delta_hours = max(delta_seconds / 3600.0, 0.0)
            raw_score += math.exp(-beta * delta_hours)

        # Normalise to [0, 1)
        return raw_score / (1.0 + raw_score)

    # ── Combined compute ───────────────────────────────────────────────────────

    def compute(
        self,
        cluster_id: str,
        source_count: Optional[int] = None,
        beta: float = _DEFAULT_BETA,
    ) -> DiffusionInfo:
        """
        Build a ``DiffusionInfo`` combining L1 and L2 metrics.

        ``source_count`` is the caller-supplied count of distinct upstream
        sources that contributed to this cluster (e.g. 3 if three different
        news outlets ran the same story).  The ``sources`` field is set to
        ``max(source_count or 1, len(history))`` so it never under-reports.

        The ``score`` field carries the L2 value, which reflects both how many
        articles arrived and how recently — more articles = higher score,
        older articles = decayed contribution.
        """
        now = datetime.now(tz=timezone.utc)

        l1 = self.compute_l1(cluster_id, now=now)
        l2_score = self.compute_l2(cluster_id, beta=beta, now=now)

        history_len = len(self._history.get(cluster_id, []))
        sources = max(source_count if source_count is not None else 1, history_len)
        # sources must be at least 1 (DiffusionInfo.sources has ge=1)
        sources = max(sources, 1)

        velocity = l1["velocity"]

        return DiffusionInfo(
            sources=sources,
            velocity=round(velocity, 6),
            score=round(l2_score, 6),
        )

    # ── Cluster eviction ───────────────────────────────────────────────────────

    def cleanup_old_clusters(self, max_age_hours: float = 48.0) -> int:
        """
        Remove clusters whose most-recent arrival is older than ``max_age_hours``.

        Returns the number of clusters removed.  Call this periodically to
        prevent unbounded memory growth in long-running workers.
        """
        cutoff = datetime.now(tz=timezone.utc).timestamp() - max_age_hours * 3600.0
        stale = [
            cid
            for cid, hist in self._history.items()
            if not hist or hist[-1].timestamp() < cutoff
        ]
        for cid in stale:
            del self._history[cid]

        if stale:
            logger.debug("DiffusionScorer evicted %d stale clusters", len(stale))

        return len(stale)
