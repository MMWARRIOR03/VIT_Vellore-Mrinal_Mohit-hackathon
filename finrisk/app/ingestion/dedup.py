"""
SHA-256-based deduplication store for the ingestion layer.

Design
──────
Content is normalised (lowercase, punctuation stripped, whitespace collapsed)
before hashing so that trivial reformatting doesn't produce duplicate entries.
The store is in-memory with a 24-hour TTL and a 50 000-entry hard cap.  When
the cap is hit, the oldest entries (by insertion time) are evicted in a single
batch — a simple LRU-style eviction that keeps memory bounded without the
overhead of a full LRU cache.

Thread-safety: this store is designed for use inside a single asyncio event
loop.  It does not use locks because all mutations happen in synchronous code
called from coroutines (no awaits inside mutating methods).  If you move to
multi-threaded execution, wrap mutations in a threading.Lock.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Optional


# ── Text normalisation ────────────────────────────────────────────────────────

# Remove everything that isn't a letter, digit, or space.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
# Collapse runs of whitespace to a single space.
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalise text for content-based deduplication.

    Steps: lowercase → strip punctuation → collapse whitespace → strip edges.
    """
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def compute_content_hash(text: str) -> str:
    """Return the hex SHA-256 digest of the normalised form of ``text``."""
    normalised = normalize_text(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ── Dedup store ───────────────────────────────────────────────────────────────

# 24 hours in seconds
_DEFAULT_TTL_SECONDS: float = 86_400.0
_DEFAULT_MAX_ENTRIES: int = 50_000
# Fraction of entries to evict when the cap is hit (evict oldest 10 %)
_EVICT_FRACTION: float = 0.10


class InMemoryDedupStore:
    """
    In-memory deduplication store with TTL expiry and a hard size cap.

    Tracks both article_id and content_hash — a document is a duplicate if
    either was seen before and hasn't expired.

    Entries are stored as::

        _seen: dict[str, float]   key → insertion_timestamp

    where ``key`` is either ``"id:<article_id>"`` or ``"hash:<content_hash>"``.
    """

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        # key → insertion_timestamp (monotonic seconds)
        self._seen: dict[str, float] = {}
        # Insertion-order tracking for eviction (list of keys in order added)
        self._insertion_order: list[str] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def is_duplicate(self, article_id: str, content_hash: str) -> bool:
        """
        Return True if either ``article_id`` or ``content_hash`` was seen
        within the TTL window.

        Does NOT mark the document as seen — call ``mark_seen()`` separately.
        """
        now = time.monotonic()
        id_key = f"id:{article_id}"
        hash_key = f"hash:{content_hash}"

        if id_key in self._seen and (now - self._seen[id_key]) < self._ttl:
            return True
        if hash_key in self._seen and (now - self._seen[hash_key]) < self._ttl:
            return True
        return False

    def mark_seen(self, article_id: str, content_hash: str) -> None:
        """
        Record ``article_id`` and ``content_hash`` as seen now.

        Prunes expired entries first, then evicts oldest entries if the store
        is at capacity.
        """
        self._prune_expired()
        if len(self._seen) >= self._max:
            self._evict_oldest()

        now = time.monotonic()
        for key in (f"id:{article_id}", f"hash:{content_hash}"):
            if key not in self._seen:
                self._insertion_order.append(key)
            self._seen[key] = now

    def size(self) -> int:
        """Return the current number of live entries (before TTL pruning)."""
        return len(self._seen)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prune_expired(self) -> None:
        """Remove all entries whose TTL has elapsed."""
        now = time.monotonic()
        expired = [k for k, ts in self._seen.items() if (now - ts) >= self._ttl]
        for k in expired:
            del self._seen[k]
        # Rebuild insertion order without expired keys; this is O(n) but
        # pruning is infrequent relative to normal operations.
        if expired:
            expired_set = set(expired)
            self._insertion_order = [
                k for k in self._insertion_order if k not in expired_set
            ]

    def _evict_oldest(self) -> None:
        """
        Evict the oldest EVICT_FRACTION of entries by insertion order.

        This keeps the store below max_entries without repeatedly evicting
        one-at-a-time (which would thrash on every call at capacity).
        """
        n_evict = max(1, int(self._max * _EVICT_FRACTION))
        to_evict = self._insertion_order[:n_evict]
        for k in to_evict:
            self._seen.pop(k, None)
        self._insertion_order = self._insertion_order[n_evict:]
