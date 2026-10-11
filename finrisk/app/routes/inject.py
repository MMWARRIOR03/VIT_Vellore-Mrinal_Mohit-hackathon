"""
POST /inject — demo endpoint.

Accepts arbitrary headline text and runs the full FinRisk pipeline synchronously:

  1. Build an article dict from the request
  2. NLPWorker.process_one() → RiskSignal
  3. Persist signal to DuckDB
  4. Run Module B stress test (if impact >= 7.0)
  5. Run Module A rebalancer
  6. Get LLM explanation (Groq → Ollama → template)
  7. Return everything in InjectResponse

The endpoint is designed to work end-to-end on a clean clone with Redis
running and no external APIs configured. It processes synchronously so its
response is not duplicated by the background stream consumers. Stress and
rebalance results are None when their gating conditions aren't met — that's
expected behaviour, not an error.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inject", tags=["Demo"])


# ── Request / response models ─────────────────────────────────────────────────

class InjectRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Headline or body text to analyse.",
        json_schema_extra={"example": "NVIDIA faces major export restrictions on AI chips"},
    )
    ticker: str = Field(
        default="AAPL",
        min_length=1,
        max_length=10,
        description="Primary ticker symbol to associate with the text.",
        json_schema_extra={"example": "NVDA"},
    )
    source: str = Field(
        default="news",
        description="Source class: 'news' | 'social' | 'filing'.",
    )


class InjectResponse(BaseModel):
    signal: dict[str, Any] = Field(description="Full RiskSignal as a dict.")
    stress_result: dict[str, Any] | None = Field(
        default=None,
        description="StressResult dict if impact >= 7.0 triggered a stress test, else null.",
    )
    rebalance_event: dict[str, Any] | None = Field(
        default=None,
        description="RebalanceEvent dict if the rebalancer fired, else null.",
    )
    explanation: str | None = Field(
        default=None,
        description="LLM-generated or template explanation of the signal.",
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("", response_model=InjectResponse, summary="Inject text through the full pipeline")
async def inject_text(body: InjectRequest, request: Request) -> InjectResponse:
    """
    Run arbitrary text through the complete FinRisk pipeline and return all outputs.

    Useful for demos and manual testing without needing a live news feed.
    """
    db = request.app.state.db
    redis = request.app.state.redis

    # Normalise source
    source = body.source if body.source in ("news", "social", "filing") else "news"
    ticker = body.ticker.upper().strip()

    # ── 1. Build article dict ──────────────────────────────────────────────────
    article_dict: dict[str, Any] = {
        "article_id": str(uuid.uuid4()),
        "title": body.text,
        "body": body.text,
        "source": source,
        "ticker_symbols": [ticker],
        "published_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_weight": 0.9,
        "source_count": 1,
    }

    # ── 2. NLP processing ──────────────────────────────────────────────────────
    # Reuse the worker instance stored on app.state if available,
    # otherwise create one ad-hoc (costs ~100ms on first call for model load).
    nlp_worker = getattr(request.app.state, "nlp_worker", None)
    if nlp_worker is None:
        from app.engine.nlp_worker import NLPWorker
        nlp_worker = NLPWorker(db=db, redis=redis)
        request.app.state.nlp_worker = nlp_worker
        logger.info("Created ad-hoc NLPWorker for /inject")

    signal = nlp_worker.process_one(article_dict)
    if signal is None:
        raise HTTPException(
            status_code=422,
            detail="NLP processing failed — the text may be too short or malformed.",
        )

    # ── 3. Persist signal ──────────────────────────────────────────────────────
    try:
        await db.write_signal(signal)
    except Exception as exc:
        logger.error("DuckDB write failed in /inject: %s", exc)
        # Non-fatal — continue so the response is still useful

    # ── 4. Module B stress test ────────────────────────────────────────────────
    # /inject runs both modules synchronously so it can return their results.
    # Do not also publish this signal to processed.signals: the background
    # runners would process it a second time and create duplicate outputs.
    stress_result_dict: dict[str, Any] | None = None
    if signal.impact >= 7.0:
        try:
            stress_engine = _get_or_create_stress_engine(request, db, redis)
            stress_result = await stress_engine.run_stress_test_full(signal)
            if stress_result is not None:
                # Persist and publish
                await stress_engine.persist_result(stress_result)
                stress_result_dict = stress_result.to_dict()
                logger.info(
                    "/inject stress test: run_id=%s ΔNAV=$%.0f",
                    stress_result.run_id, stress_result.delta_nav,
                )
        except Exception as exc:
            logger.error("Module B stress test failed in /inject: %s", exc)

    # ── 5. Module A rebalancer ─────────────────────────────────────────────────
    rebalance_dict: dict[str, Any] | None = None
    try:
        rebalancer = await _get_or_create_rebalancer(request, db, redis)
        rebalance_event = await rebalancer.on_signal(signal)
        if rebalance_event is not None:
            await rebalance_event.persist(db)
            await redis.publish("signals.rebalance", rebalance_event.to_dict())
            rebalance_dict = rebalance_event.to_dict()
            logger.info(
                "/inject rebalance: id=%s reason=%r",
                rebalance_event.rebalance_id, rebalance_event.reason,
            )
    except Exception as exc:
        logger.error("Module A rebalancer failed in /inject: %s", exc)

    # ── 6. LLM explanation ─────────────────────────────────────────────────────
    explanation: str | None = None
    explainer = getattr(request.app.state, "explainer", None)
    if explainer is not None:
        try:
            explanation = await explainer.explain(signal, headline=body.text)
        except Exception as exc:
            logger.error("LLM explainer failed in /inject: %s", exc)

    # ── 7. Return response ─────────────────────────────────────────────────────
    # Use the explanation from the LLM to enrich the signal dict in the response.
    signal_dict = json.loads(signal.model_dump_json())
    if explanation:
        signal_dict["explanation"] = explanation

    return InjectResponse(
        signal=signal_dict,
        stress_result=stress_result_dict,
        rebalance_event=rebalance_dict,
        explanation=explanation,
    )


# ── Lazy singletons ────────────────────────────────────────────────────────────

def _get_or_create_stress_engine(request: Request, db, redis):
    """Return or lazily create a StressEngine on app.state."""
    engine = getattr(request.app.state, "stress_engine", None)
    if engine is None:
        from module_b.stress_engine import StressEngine
        from app.config import get_settings
        settings = get_settings()
        n_paths = settings.mc_n_paths if settings.mc_enabled else 100
        engine = StressEngine(db=db, redis=redis, mc_n_paths=n_paths)
        request.app.state.stress_engine = engine
        logger.info("Created ad-hoc StressEngine for /inject")
    return engine


async def _get_or_create_rebalancer(request: Request, db, redis):
    """Return or lazily create an IndexRebalancer on app.state."""
    rebalancer = getattr(request.app.state, "rebalancer", None)
    if rebalancer is None:
        from module_a.rebalancer import IndexRebalancer
        rebalancer = IndexRebalancer()
        request.app.state.rebalancer = rebalancer
        logger.info("Created ad-hoc IndexRebalancer for /inject")
    return rebalancer
