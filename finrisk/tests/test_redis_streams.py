"""
Integration tests for Redis Streams.

Marked with ``pytest.mark.integration`` — excluded from the default unit-test
run.  Requires a live Redis instance at the URL in REDIS_URL (defaults to
``redis://localhost:6379/0``).

Run with:
    pytest tests/test_redis_streams.py -m integration -v

Covers:
  - StreamProducer.send → StreamConsumer.read_batch round-trip
  - Message payload preserved end-to-end
  - Multiple messages in a single read_batch call
  - last_id tracking advances after each read
  - RedisStreamClient convenience methods (publish / publish_json_str)
  - Consumer with last_id="0" replays historical messages
  - iter_messages stops cleanly when stop_event is set
  - All four stream name constants exist and are non-empty strings
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.streams.redis_streams import (
    ALL_STREAMS,
    STREAM_PROCESSED_SIGNALS,
    STREAM_RAW_ARTICLES,
    STREAM_SIGNALS_REBALANCE,
    STREAM_SIGNALS_STRESS,
    RedisStreamClient,
    StreamConsumer,
    StreamProducer,
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _unique_stream() -> str:
    """Generate a unique ephemeral stream name so tests don't collide."""
    return f"test.{uuid.uuid4().hex}"


def _sample_payload(label: str = "test") -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "label": label,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "value": 42.0,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def redis_client():
    """Module-scoped Redis client — one connection for all integration tests."""
    client = RedisStreamClient(url=REDIS_URL)
    await client.connect()
    yield client
    await client.close()


# ── Stream name constants ─────────────────────────────────────────────────────

@pytest.mark.integration
class TestStreamNameConstants:

    def test_all_stream_names_are_non_empty_strings(self):
        for name in ALL_STREAMS:
            assert isinstance(name, str) and len(name) > 0

    def test_expected_streams_are_present(self):
        assert STREAM_RAW_ARTICLES in ALL_STREAMS
        assert STREAM_PROCESSED_SIGNALS in ALL_STREAMS
        assert STREAM_SIGNALS_REBALANCE in ALL_STREAMS
        assert STREAM_SIGNALS_STRESS in ALL_STREAMS

    def test_all_streams_are_distinct(self):
        assert len(set(ALL_STREAMS)) == len(ALL_STREAMS)


# ── Produce → consume round-trip ──────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
class TestProduceConsumeRoundTrip:

    async def test_single_message_round_trip(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        payload = _sample_payload("round-trip")

        producer = redis_client.producer(stream)
        msg_id = await producer.send(payload)
        assert isinstance(msg_id, str) and len(msg_id) > 0

        # Consumer starts from "0" to read the message we just wrote
        consumer = redis_client.consumer(stream, last_id="0")
        batch = await consumer.read_batch()

        assert len(batch) == 1
        received_id, received_payload = batch[0]
        assert received_id == msg_id
        assert received_payload["event_id"] == payload["event_id"]
        assert received_payload["label"] == "round-trip"
        assert received_payload["value"] == pytest.approx(42.0)

    async def test_multiple_messages_in_single_read_batch(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        sent_ids = []
        for i in range(5):
            msg_id = await producer.send(_sample_payload(f"msg-{i}"))
            sent_ids.append(msg_id)

        consumer = redis_client.consumer(stream, last_id="0")
        # Request up to 10 messages; should get all 5
        consumer._batch_size = 10  # noqa: SLF001
        batch = await consumer.read_batch()

        assert len(batch) == 5
        received_ids = [msg_id for msg_id, _ in batch]
        assert received_ids == sent_ids

    async def test_payload_with_nested_dict_preserved(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        nested_payload = {
            "event_id": "nested-001",
            "data": {"a": 1, "b": {"c": [1, 2, 3]}},
        }
        await producer.send(nested_payload)

        consumer = redis_client.consumer(stream, last_id="0")
        batch = await consumer.read_batch()
        _, received = batch[0]
        assert received["data"]["b"]["c"] == [1, 2, 3]

    async def test_raw_json_string_producer(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        raw = json.dumps({"event_id": "raw-001", "x": 99})
        await producer.send_raw(raw)

        consumer = redis_client.consumer(stream, last_id="0")
        batch = await consumer.read_batch()
        _, received = batch[0]
        assert received["x"] == 99

    async def test_last_id_advances_after_read(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        id1 = await producer.send(_sample_payload("first"))
        id2 = await producer.send(_sample_payload("second"))

        consumer = redis_client.consumer(stream, last_id="0")
        # First read: should return message 1
        consumer._batch_size = 1  # noqa: SLF001
        batch1 = await consumer.read_batch()
        assert len(batch1) == 1
        assert batch1[0][0] == id1
        assert consumer.last_id == id1

        # Second read: should return message 2
        batch2 = await consumer.read_batch()
        assert len(batch2) == 1
        assert batch2[0][0] == id2
        assert consumer.last_id == id2

    async def test_no_messages_returns_empty_list(self, redis_client: RedisStreamClient):
        """Consumer with $ last_id on a new stream should time out and return []."""
        stream = _unique_stream()
        consumer = RedisStreamClient(REDIS_URL).consumer(stream, last_id="$")
        # Use a short timeout for the test
        consumer._block_ms = 200  # noqa: SLF001
        # The consumer's redis connection is not yet open — get one from the fixture
        consumer._redis = redis_client._redis  # noqa: SLF001
        batch = await consumer.read_batch()
        assert batch == []


# ── RedisStreamClient convenience methods ─────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
class TestRedisStreamClientPublish:

    async def test_publish_dict_round_trip(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        payload = _sample_payload("client-publish")
        msg_id = await redis_client.publish(stream, payload)
        assert isinstance(msg_id, str)

        consumer = redis_client.consumer(stream, last_id="0")
        batch = await consumer.read_batch()
        assert len(batch) == 1
        assert batch[0][1]["label"] == "client-publish"

    async def test_publish_json_str_round_trip(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        payload = {"event_id": "jstr-001", "data": "hello"}
        json_str = json.dumps(payload)
        await redis_client.publish_json_str(stream, json_str)

        consumer = redis_client.consumer(stream, last_id="0")
        batch = await consumer.read_batch()
        assert batch[0][1]["data"] == "hello"


# ── iter_messages / stop_event ────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
class TestIterMessages:

    async def test_iter_messages_stops_on_stop_event(self, redis_client: RedisStreamClient):
        """
        Publish 3 messages before starting iteration, then set stop_event after
        consuming 2.  The generator must exit before consuming the third.
        """
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        for i in range(3):
            await producer.send({"seq": i})

        consumer = redis_client.consumer(stream, last_id="0")
        consumer._block_ms = 200  # noqa: SLF001

        stop = asyncio.Event()
        collected = []

        async for msg_id, payload in consumer.iter_messages(stop_event=stop):
            collected.append(payload)
            if len(collected) == 2:
                stop.set()

        assert len(collected) == 2

    async def test_iter_messages_yields_all_preloaded_messages(self, redis_client: RedisStreamClient):
        """Consume all messages already in the stream then stop."""
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        n = 4
        for i in range(n):
            await producer.send({"index": i})

        consumer = redis_client.consumer(stream, last_id="0")
        consumer._block_ms = 200  # noqa: SLF001
        consumer._batch_size = 10  # noqa: SLF001

        stop = asyncio.Event()
        collected = []

        async for _msg_id, payload in consumer.iter_messages(stop_event=stop):
            collected.append(payload)
            if len(collected) == n:
                stop.set()

        assert len(collected) == n
        assert [p["index"] for p in collected] == list(range(n))


# ── Consumer with last_id="0" replays history ─────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
class TestHistoricalReplay:

    async def test_consumer_from_zero_replays_all_messages(self, redis_client: RedisStreamClient):
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        ids = []
        for i in range(3):
            ids.append(await producer.send({"i": i}))

        # First consumer reads all
        c1 = redis_client.consumer(stream, last_id="0")
        c1._batch_size = 10  # noqa: SLF001
        b1 = await c1.read_batch()
        assert len(b1) == 3

        # Second fresh consumer also replays all from history
        c2 = redis_client.consumer(stream, last_id="0")
        c2._batch_size = 10  # noqa: SLF001
        b2 = await c2.read_batch()
        assert len(b2) == 3

    async def test_consumer_from_dollar_misses_historical_messages(
        self, redis_client: RedisStreamClient
    ):
        stream = _unique_stream()
        producer = redis_client.producer(stream)
        await producer.send({"old": True})

        # Start consumer AFTER the old message was written
        consumer = redis_client.consumer(stream, last_id="$")
        consumer._block_ms = 200  # noqa: SLF001
        batch = await consumer.read_batch()
        # Should not see the old message
        assert batch == []
