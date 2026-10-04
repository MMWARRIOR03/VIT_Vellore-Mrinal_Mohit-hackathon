"""
PriceCache — recent closing prices for vol estimation.

yfinance is an optional dependency.  When it is unavailable (no install,
network outage, or API limit), the cache silently falls back to a deterministic
synthetic random walk seeded from the ticker string.  This keeps the rebalancer
functional in CI and offline environments while producing the same vol estimate
for the same ticker across runs.

Cache TTL is 1 hour.  A full refresh downloads 30 days of daily closes.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS: int = 3600
_DEFAULT_LOOKBACK_DAYS: int = 30
_SYNTHETIC_START_PRICE: float = 100.0


class PriceCache:
    """
    Per-ticker price cache backed by yfinance with a synthetic fallback.

    Prices are stored newest-first so callers can slice ``[:n]`` to get
    the n most recent prices without reversing.
    """

    def __init__(self) -> None:
        # {ticker: [price_newest_first, ...]}
        self._cache: dict[str, list[float]] = {}
        self._last_refresh: Optional[datetime] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def refresh(self, tickers: list[str]) -> None:
        """
        Attempt to download the last 30 days of daily closes from yfinance.

        Per-ticker failures are caught and logged; the ticker falls back to
        synthetic prices.  The cache TTL is reset on completion regardless
        of individual ticker outcomes.
        """
        try:
            import yfinance as yf  # type: ignore[import]
        except ImportError:
            logger.warning(
                "yfinance not installed; using synthetic prices for all tickers."
            )
            for ticker in tickers:
                self._cache[ticker] = self._synthetic_prices(ticker)
            self._last_refresh = datetime.now(tz=timezone.utc)
            return

        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    period="30d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if data.empty:
                    raise ValueError("Empty dataframe returned")
                closes = list(reversed(data["Close"].dropna().tolist()))
                self._cache[ticker] = closes
                logger.debug("Refreshed %d prices for %s", len(closes), ticker)
            except Exception as exc:
                logger.warning(
                    "Price fetch failed for %s (%s); using synthetic.", ticker, exc
                )
                self._cache[ticker] = self._synthetic_prices(ticker)

        self._last_refresh = datetime.now(tz=timezone.utc)
        logger.info(
            "PriceCache refresh complete. %d tickers updated.", len(tickers)
        )

    def get_prices(self, ticker: str, n_days: int = _DEFAULT_LOOKBACK_DAYS) -> list[float]:
        """
        Return up to ``n_days`` recent daily closes for the ticker (newest-first).

        If the ticker has never been fetched, a synthetic series is generated
        and cached on the spot.
        """
        if ticker not in self._cache:
            self._cache[ticker] = self._synthetic_prices(ticker)
        return self._cache[ticker][:n_days]

    def is_stale(self) -> bool:
        """Return True if the cache has never been refreshed or TTL has elapsed."""
        if self._last_refresh is None:
            return True
        age = (datetime.now(tz=timezone.utc) - self._last_refresh).total_seconds()
        return age > CACHE_TTL_SECONDS

    # ── Synthetic fallback ────────────────────────────────────────────────────

    @staticmethod
    def _synthetic_prices(ticker: str, n: int = _DEFAULT_LOOKBACK_DAYS) -> list[float]:
        """
        Deterministic synthetic price series for offline/fallback use.

        Seeded from a hash of the ticker string so the same ticker always
        produces the same series.  This keeps vol estimates stable across
        restarts even when yfinance is unavailable.
        """
        seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Build a geometric random walk (oldest → newest).
        prices: list[float] = [_SYNTHETIC_START_PRICE]
        daily_vol = 0.015 + rng.random() * 0.025  # 1.5%–4% daily vol
        drift = 0.0001 * rng.gauss(0, 1)           # mild daily drift

        for _ in range(n - 1):
            ret = drift + daily_vol * rng.gauss(0, 1)
            prices.append(prices[-1] * math.exp(ret))

        # Return newest-first.
        prices.reverse()
        return prices
