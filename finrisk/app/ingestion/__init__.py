"""
Ingestion layer for FinRisk.

Adapters pull raw text from GDELT, SEC EDGAR, StockTwits, and a Kaggle
tweet replay CSV.  Every document flows through deduplication and entity
linking before being written to DuckDB and published to the raw.articles
Redis stream.
"""

from app.ingestion.base import RawDocument, SourceAdapter
from app.ingestion.dedup import InMemoryDedupStore
from app.ingestion.entity_linker import EntityLinker
from app.ingestion.pipeline import IngestionPipeline

__all__ = [
    "RawDocument",
    "SourceAdapter",
    "InMemoryDedupStore",
    "EntityLinker",
    "IngestionPipeline",
]
