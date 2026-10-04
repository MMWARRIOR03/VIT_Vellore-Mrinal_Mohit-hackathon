"""
Article endpoints.

GET /articles/recent — fetch recent raw articles from DuckDB.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/recent", response_model=list[dict[str, Any]])
async def get_articles_recent(
    request: Request,
    ticker: str | None = Query(
        default=None,
        description="Filter articles mentioning this ticker, e.g. NVDA.",
        min_length=1,
        max_length=10,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of articles to return (1–500).",
    ),
) -> list[dict[str, Any]]:
    """
    Return the most recent ingested articles, newest first.

    Articles are stored by the ingestion layer (Task 2).  Before ingestion is
    wired up, this endpoint returns an empty list rather than an error so that
    the dashboard and integration tests can call it unconditionally.
    """
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available.")

    try:
        articles = await db.get_articles_recent(
            ticker=ticker.upper() if ticker else None,
            limit=limit,
        )
    except Exception as exc:
        logger.exception("Failed to fetch articles: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch articles.") from exc

    return articles
