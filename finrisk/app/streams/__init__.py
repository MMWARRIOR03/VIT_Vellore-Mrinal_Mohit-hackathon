"""
Streams package — re-exports the Redis stream client and stream name constants.
"""

from app.streams.redis_streams import (
    STREAM_RAW_ARTICLES,
    STREAM_PROCESSED_SIGNALS,
    STREAM_SIGNALS_REBALANCE,
    STREAM_SIGNALS_STRESS,
    RedisStreamClient,
    StreamProducer,
    StreamConsumer,
)

__all__ = [
    "STREAM_RAW_ARTICLES",
    "STREAM_PROCESSED_SIGNALS",
    "STREAM_SIGNALS_REBALANCE",
    "STREAM_SIGNALS_STRESS",
    "RedisStreamClient",
    "StreamProducer",
    "StreamConsumer",
]
