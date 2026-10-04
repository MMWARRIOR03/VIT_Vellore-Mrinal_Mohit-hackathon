"""
generate_sample_data.py — standalone script to pre-generate replay fixtures.

Writes:
  data/sample/articles.jsonl  — 500 synthetic articles (replay input)
  data/sample/signals.jsonl   — 100 pre-processed RiskSignals (dashboard seed)

Run from the finrisk/ root:
    python scripts/generate_sample_data.py

The script does NOT require Redis or a running API.  It uses NLPWorker.process_one()
directly to produce the signals.jsonl, so FinBERT will be downloaded on first run
(~500 MB) and cached in ~/.cache/huggingface.  Subsequent runs use the cache.

Pass --no-nlp to skip signal generation (articles only):
    python scripts/generate_sample_data.py --no-nlp
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the finrisk/ root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("generate_sample_data")

_SAMPLE_DIR = Path("data/sample")
_ARTICLES_PATH = _SAMPLE_DIR / "articles.jsonl"
_SIGNALS_PATH = _SAMPLE_DIR / "signals.jsonl"
_N_ARTICLES = 500
_N_SIGNALS = 100


def main(skip_nlp: bool = False) -> None:
    _SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Generate articles ─────────────────────────────────────────────────────
    logger.info("Generating %d synthetic articles…", _N_ARTICLES)
    from app.replay.replay_runner import generate_sample_articles
    articles = generate_sample_articles(n=_N_ARTICLES, hours=4.0)

    with _ARTICLES_PATH.open("w") as f:
        for article in articles:
            f.write(json.dumps(article, default=str) + "\n")
    logger.info("Wrote %d articles to %s", len(articles), _ARTICLES_PATH)

    if skip_nlp:
        logger.info("--no-nlp set; skipping signal generation.")
        return

    # ── Generate signals via NLPWorker ────────────────────────────────────────
    logger.info("Processing %d articles through NLPWorker…", _N_SIGNALS)
    logger.info("(FinBERT will be downloaded on first run — this may take a few minutes)")

    try:
        from app.engine.nlp_worker import NLPWorker
        # NLPWorker can be constructed without db/redis for process_one() calls
        # by passing stub objects; simpler to pass None and let it handle it.
        worker = _make_headless_worker()
    except Exception as exc:
        logger.error("Failed to initialise NLPWorker: %s", exc)
        logger.info("Skipping signal generation.  Run again after installing NLP deps.")
        return

    signals = []
    subset = articles[:_N_SIGNALS]
    for i, article in enumerate(subset):
        try:
            signal = worker.process_one(article)
            if signal is not None:
                signals.append(json.loads(signal.model_dump_json()))
        except Exception as exc:
            logger.warning("Failed to process article %d: %s", i, exc)
        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d articles", i + 1, len(subset))

    with _SIGNALS_PATH.open("w") as f:
        for sig in signals:
            f.write(json.dumps(sig, default=str) + "\n")
    logger.info("Wrote %d signals to %s", len(signals), _SIGNALS_PATH)


def _make_headless_worker():
    """
    Construct an NLPWorker without a real DB or Redis client.

    We pass placeholder stubs because process_one() only needs the NLP models;
    the persistence and publish steps are not called on the sync path.
    """
    from app.engine.nlp_worker import NLPWorker

    class _NullDB:
        async def write_signal(self, *a, **kw): pass

    class _NullRedis:
        def producer(self, *a, **kw):
            class _P:
                async def send(self, *a, **kw): pass
            return _P()

    return NLPWorker(db=_NullDB(), redis=_NullRedis())  # type: ignore[arg-type]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-nlp",
        action="store_true",
        help="Skip NLP signal generation (articles.jsonl only).",
    )
    args = parser.parse_args()
    main(skip_nlp=args.no_nlp)
