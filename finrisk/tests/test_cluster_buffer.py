"""
Tests for ClusterSentimentBuffer.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pytest

from app.engine.cluster_buffer import ClusterSentimentBuffer


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── 1. Single article ─────────────────────────────────────────────────────────

def test_single_article():
    buf = ClusterSentimentBuffer()
    cid = "cluster_a"
    buf.add(cid, sentiment=0.7, confidence=0.8, source_weight=1.0, ts=_utcnow())

    agg_sentiment, agg_conf = buf.get_aggregate(cid)
    assert agg_sentiment == pytest.approx(0.7)
    assert agg_conf == pytest.approx(0.8)
    assert buf.get_source_count(cid) == 1


# ── 2. Three articles, same cluster ───────────────────────────────────────────

def test_three_articles_same_cluster():
    buf = ClusterSentimentBuffer()
    cid = "cluster_b"
    articles = [
        # (sentiment, confidence, source_weight)
        (0.8, 0.9, 1.0),
        (0.5, 0.7, 0.8),
        (0.2, 0.6, 0.5),
    ]
    for s, c, w in articles:
        buf.add(cid, sentiment=s, confidence=c, source_weight=w, ts=_utcnow())

    assert buf.get_source_count(cid) == 3

    # Combined confidence formula: base_conf + 0.05 * (n - 1), capped at 1.0
    base_conf = articles[0][1]  # 0.9
    expected_conf = min(1.0, base_conf + 0.05 * 2)  # 0.9 + 0.10 = 1.0
    agg_sentiment, agg_conf = buf.get_aggregate(cid)
    assert agg_conf == pytest.approx(expected_conf)

    # Weighted average sentiment: sum(s*c*w) / sum(c*w)
    numerator = sum(s * c * w for s, c, w in articles)
    denominator = sum(c * w for _s, c, w in articles)
    expected_sentiment = numerator / denominator
    assert agg_sentiment == pytest.approx(expected_sentiment, rel=1e-6)


# ── 3. Source-weight weighting ────────────────────────────────────────────────

def test_source_weight_weighting():
    buf = ClusterSentimentBuffer()
    cid = "cluster_c"

    # High-quality source: strong positive sentiment
    buf.add(cid, sentiment=0.9, confidence=0.9, source_weight=1.0, ts=_utcnow())
    # Low-quality source: weak negative sentiment
    buf.add(cid, sentiment=0.1, confidence=0.5, source_weight=0.4, ts=_utcnow())

    agg_sentiment, _ = buf.get_aggregate(cid)

    # High-quality source should dominate; result must be closer to 0.9 than 0.1
    midpoint = (0.9 + 0.1) / 2  # 0.5
    assert agg_sentiment > midpoint, (
        f"Expected agg_sentiment > {midpoint}, got {agg_sentiment}"
    )


# ── 4. Cleanup removes old clusters ──────────────────────────────────────────

def test_cleanup_removes_old_clusters():
    buf = ClusterSentimentBuffer()
    cid = "cluster_old"

    old_ts = _utcnow() - timedelta(hours=3)
    buf.add(cid, sentiment=0.5, confidence=0.7, source_weight=1.0, ts=old_ts)

    removed = buf.cleanup_old(max_age_hours=2.0)
    assert removed == 1
    assert buf.get_source_count(cid) == 0
    assert buf.get_aggregate(cid) == (0.0, 0.0)


# ── 5. get_source_count tracks correctly across multiple clusters ─────────────

def test_get_source_count_tracks():
    buf = ClusterSentimentBuffer()
    cid_a = "cluster_x"
    cid_b = "cluster_y"

    for _ in range(5):
        buf.add(cid_a, sentiment=0.5, confidence=0.8, source_weight=1.0, ts=_utcnow())

    assert buf.get_source_count(cid_a) == 5

    for _ in range(2):
        buf.add(cid_b, sentiment=0.3, confidence=0.7, source_weight=0.9, ts=_utcnow())

    assert buf.get_source_count(cid_a) == 5
    assert buf.get_source_count(cid_b) == 2
