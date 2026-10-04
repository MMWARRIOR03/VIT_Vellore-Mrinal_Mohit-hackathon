"""
Base types for the ingestion layer.

Every source adapter returns a list of RawDocument objects.  The pipeline
consumes them uniformly regardless of origin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


# Source-weight defaults by document type.  These reflect the expected
# signal quality of each source class and are used downstream by the NLP
# scoring layer to calibrate confidence.
SOURCE_WEIGHT_DEFAULTS: dict[str, float] = {
    "news": 0.85,
    "social": 0.40,
    "filing": 1.0,
}


@dataclass
class RawDocument:
    """
    Canonical representation of a raw ingested document before NLP processing.

    Fields
    ──────
    article_id      Globally unique identifier for this document (e.g. SHA256
                    of URL+title, tweet ID, or SEC accession number).
    source          Source class: "news", "social", or "filing".
    ticker_symbols  Tickers resolved at ingest time (may be augmented by the
                    entity linker after construction).
    title           Headline or short summary.  May be empty for social posts.
    body            Full text body.  May be empty if only a headline is available.
    url             Canonical URL to the original document.
    published_at    Publication timestamp (timezone-aware UTC preferred).
    source_weight   Credibility weight for this source class (0–1).
                    Defaults to SOURCE_WEIGHT_DEFAULTS[source] if not supplied.
    raw_metadata    Adapter-specific extra fields (e.g. event_type_hint for
                    EDGAR, sentiment_hint for StockTwits).
    """

    article_id: str
    source: Literal["news", "social", "filing"]
    ticker_symbols: list[str]
    title: str
    body: str
    url: str
    published_at: datetime
    source_weight: float = field(default=0.0)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Apply default weight if caller left it at the zero sentinel.
        if self.source_weight == 0.0:
            self.source_weight = SOURCE_WEIGHT_DEFAULTS.get(self.source, 0.5)

        # Normalise ticker symbols to uppercase, deduplicate, preserve order.
        seen: set[str] = set()
        normalised: list[str] = []
        for t in self.ticker_symbols:
            t_upper = t.upper().strip()
            if t_upper and t_upper not in seen:
                seen.add(t_upper)
                normalised.append(t_upper)
        self.ticker_symbols = normalised


class SourceAdapter(ABC):
    """
    Abstract base for all ingestion adapters.

    Concrete adapters implement ``fetch()`` which returns a batch of
    RawDocument objects gathered in one poll cycle.  The ``run_forever()``
    method is the long-running coroutine that calls ``fetch()`` on a schedule
    and hands documents to the pipeline.
    """

    @abstractmethod
    async def fetch(self) -> list[RawDocument]:
        """
        Poll the source for new documents.

        Returns a list (possibly empty) of RawDocument objects.  Implementations
        must not raise on transient network errors — catch, log, and return [].
        """
        ...
