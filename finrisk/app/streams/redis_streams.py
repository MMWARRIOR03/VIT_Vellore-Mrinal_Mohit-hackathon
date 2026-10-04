"""
Redis Streams abstraction for FinRisk.

Stream topology
───────────────
  raw.articles          ← ingestion layer pushes raw article dicts
  processed.signals     ← NLP engine (or fake generator) pushes RiskSignal JSON
  signals.rebalance     ← Module A subscribes; signals that triggered a rebalance
  signals.stress        ← Module B subscribes; signals that triggered a stress run

Each stream uses Redis Streams (XADD / XREAD) rather than Pub/Sub so that:
  - messages are durable until the stream is trimmed (MAXLEN),
  - consumer groups and acknowledgements can be layered on later without a
    client-side API change, and
  - the last-seen ID mechanism gives per-consumer resumability.

Message envelope
────────────────
Every message is a Redis hash with a single field ``"data"`` whose value is a
JSON-encoded string.  This keeps the consumer side simple — always decode
``msg["data"]`` — while allowing additional metadata fields to be added later.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from redis.asyncio.client import Redis

logger = logging.getLogger(__name__)

# ── Stream name constants ──────────────────────────────────────────────────────

STREAM_RAW_ARTICLES: str = "raw.articles"
STREAM_PROCESSED_SIGNALS: str = "processed.signals"
STREAM_SIGNALS_REBALANCE: str = "signals.rebalance"
STREAM_SIGNALS_STRESS: str = "signals.stress"

ALL_STREAMS: tuple[str, ...] = (
    STREAM_RAW_ARTICLES,
    STREAM_PROCESSED_SIGNALS,
    STREAM_SIGNALS_REBALANCE,
    STREAM_SIGNALS_STRESS,
)


# ── Connection manager ─────────────────────────────────────────────────────────

class RedisStreamClient:
    """
    Thin lifecycle wrapper around an async Redis connection pool.

    Usage::

        client = RedisStreamClient("redis://localhost:6379/0")
        await client.connect()
        producer = client.producer(STREAM_PROCESSED_SIGNALS)
        consumer = client.consumer(STREAM_PROCESSED_SIGNALS)
        ...
        await client.close()
    """

    def __init__(self, url: str, stream_maxlen: int = 10_000) -> None:
        self._url = url
        self._stream_maxlen = stream_maxlen
        self._redis: Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            retry_on_timeout=True,
        )
        # Verify connectivity
        await self._redis.ping()
        logger.info("Redis connected: %s", self._url)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis connection closed.")

    def _assert_connected(self) -> Redis:
        if self._redis is None:
            raise RuntimeError(
                "RedisStreamClient is not connected.  Call connect() first."
            )
        return self._redis

    def producer(self, stream: str) -> "StreamProducer":
        return StreamProducer(
            redis=self._assert_connected(),
            stream=stream,
            maxlen=self._stream_maxlen,
        )

    def consumer(self, stream: str, *, last_id: str = "$") -> "StreamConsumer":
        return StreamConsumer(
            redis=self._assert_connected(),
            stream=stream,
            last_id=last_id,
        )

    # Convenience: publish directly without constructing a producer
    async def publish(self, stream: str, payload: dict[str, Any]) -> str:
        """XADD a JSON-encoded payload dict.  Returns the Redis message ID."""
        r = self._assert_connected()
        msg_id: str = await r.xadd(
            stream,
            {"data": json.dumps(payload, default=str)},
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        return msg_id

    async def publish_json_str(self, stream: str, json_str: str) -> str:
        """XADD a pre-serialised JSON string.  Returns the Redis message ID."""
        r = self._assert_connected()
        msg_id: str = await r.xadd(
            stream,
            {"data": json_str},
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        return msg_id


# ── Producer ──────────────────────────────────────────────────────────────────

class StreamProducer:
    """
    Writes messages to a single Redis stream.

    All messages are wrapped in the ``{"data": <json_str>}`` envelope so
    consumers always know the field name.
    """

    def __init__(self, redis: Redis, stream: str, maxlen: int = 10_000) -> None:
        self._redis = redis
        self._stream = stream
        self._maxlen = maxlen

    async def send(self, payload: dict[str, Any]) -> str:
        """Serialise ``payload`` as JSON and XADD to the stream."""
        msg_id: str = await self._redis.xadd(
            self._stream,
            {"data": json.dumps(payload, default=str)},
            maxlen=self._maxlen,
            approximate=True,
        )
        logger.debug("Produced to %s: id=%s", self._stream, msg_id)
        return msg_id

    async def send_raw(self, json_str: str) -> str:
        """XADD a pre-serialised JSON string."""
        msg_id: str = await self._redis.xadd(
            self._stream,
            {"data": json_str},
            maxlen=self._maxlen,
            approximate=True,
        )
        logger.debug("Produced (raw) to %s: id=%s", self._stream, msg_id)
        return msg_id


# ── Consumer ──────────────────────────────────────────────────────────────────

class StreamConsumer:
    """
    Reads messages from a single Redis stream via XREAD.

    ``last_id`` controls where reading starts:
      - ``"$"``  — only new messages arriving after this consumer was created.
      - ``"0"``  — replay all messages from the beginning.
      - ``"<id>"`` — resume from a specific message ID.

    The consumer tracks the last seen ID internally so iteration is resumable
    within the same process lifetime.
    """

    def __init__(
        self,
        redis: Redis,
        stream: str,
        last_id: str = "$",
        block_ms: int = 2_000,
        batch_size: int = 10,
    ) -> None:
        self._redis = redis
        self._stream = stream
        self._last_id = last_id
        self._block_ms = block_ms
        self._batch_size = batch_size

    async def read_batch(self) -> list[tuple[str, dict[str, Any]]]:
        """
        Perform one XREAD call and return a list of (message_id, payload_dict)
        tuples.  Blocks up to ``block_ms`` if no messages are available.
        Returns an empty list if the timeout fires with no messages.
        """
        response = await self._redis.xread(
            {self._stream: self._last_id},
            count=self._batch_size,
            block=self._block_ms,
        )
        if not response:
            return []

        results: list[tuple[str, dict[str, Any]]] = []
        for _stream_name, messages in response:
            for msg_id, fields in messages:
                self._last_id = msg_id
                raw = fields.get("data", "{}")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Non-JSON message on %s id=%s: %s", self._stream, msg_id, exc
                    )
                    payload = {"_raw": raw}
                results.append((msg_id, payload))
        return results

    async def iter_messages(
        self,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """
        Async generator that yields (message_id, payload_dict) indefinitely.

        Pass a ``stop_event`` to break the loop cleanly from outside::

            stop = asyncio.Event()
            async for msg_id, payload in consumer.iter_messages(stop):
                process(payload)
                if done:
                    stop.set()
        """
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                batch = await self.read_batch()
            except Exception as exc:
                logger.error("Error reading from %s: %s", self._stream, exc)
                await asyncio.sleep(1.0)
                continue
            for item in batch:
                yield item

    @property
    def last_id(self) -> str:
        """The last consumed message ID (useful for checkpointing)."""
        return self._last_id
