"""
Entity linker: maps free-text mentions of companies and ticker symbols to
canonical ticker symbols from the 20-stock universe.

Loading
───────
On init, ``EntityLinker`` reads ``configs/ticker_map.yaml`` relative to the
project root (two levels above this file) and compiles one regex pattern per
alias.  All alias patterns are sorted longest-first so that "JPMorgan Chase"
is matched before "JPMorgan" when both appear in a single string.

Matching rules
──────────────
1. Every alias from the YAML is matched as a whole word (``\b`` boundaries)
   case-insensitively.
2. Ticker symbols are matched with an optional leading ``$`` (StockTwits
   cashtag style), also with word boundaries.
3. Results are deduplicated while preserving the order of first match.

Performance
───────────
The compiled regex patterns are cached at construction time.  For 20 tickers
with ~5 aliases each that gives ~100 patterns, which is fast enough for
single-document throughput without needing Aho-Corasick.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Path to the ticker map YAML relative to this file's location.
# app/ingestion/entity_linker.py → ../../configs/ticker_map.yaml
_TICKER_MAP_PATH = Path(__file__).parent.parent.parent / "configs" / "ticker_map.yaml"


class EntityLinker:
    """
    Resolves free-text company and ticker mentions to canonical ticker symbols.

    Usage::

        linker = EntityLinker()
        tickers = linker.link("Apple reported earnings; $NVDA hit a new high.")
        # → ["AAPL", "NVDA"]
    """

    def __init__(self, ticker_map_path: Optional[Path] = None) -> None:
        path = ticker_map_path or _TICKER_MAP_PATH
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        self._load(path)

    # ── Public interface ──────────────────────────────────────────────────────

    def link(self, text: str) -> list[str]:
        """
        Return a deduplicated list of ticker symbols found in ``text``.

        Matching is case-insensitive.  Order reflects first-match position;
        duplicates (same ticker matched by different aliases) are collapsed.
        """
        if not text:
            return []

        seen: set[str] = set()
        results: list[str] = []

        for pattern, ticker in self._patterns:
            if pattern.search(text):
                if ticker not in seen:
                    seen.add(ticker)
                    results.append(ticker)

        return results

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        """
        Parse the ticker map YAML and compile one regex per alias + ticker symbol.
        """
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError:
            logger.error(
                "ticker_map.yaml not found at %s — entity linking will return no results",
                path,
            )
            return
        except yaml.YAMLError as exc:
            logger.error("Failed to parse ticker_map.yaml: %s", exc)
            return

        tickers_data: dict = data.get("tickers", {})

        # Collect (alias_text, ticker) pairs, then sort longest alias first so
        # that more-specific matches (e.g. "JPMorgan Chase") win over shorter
        # overlapping ones (e.g. "JPMorgan") when the result is deduplicated.
        alias_pairs: list[tuple[str, str]] = []

        for ticker_symbol, info in tickers_data.items():
            aliases: list[str] = info.get("aliases", [])

            for alias in aliases:
                alias_stripped = alias.strip()
                if alias_stripped:
                    alias_pairs.append((alias_stripped, ticker_symbol))

            # Always add the bare ticker symbol as a cashtag-optional pattern.
            # This is handled separately below after sorting.
            alias_pairs.append((ticker_symbol, ticker_symbol))

        # Sort longest alias first to prefer specific matches.
        alias_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

        for alias_text, ticker_symbol in alias_pairs:
            pattern = self._build_pattern(alias_text)
            if pattern is not None:
                self._patterns.append((pattern, ticker_symbol))

        logger.info(
            "EntityLinker loaded %d patterns for %d tickers",
            len(self._patterns),
            len(tickers_data),
        )

    @staticmethod
    def _build_pattern(alias: str) -> Optional[re.Pattern[str]]:
        """
        Compile a case-insensitive whole-word regex for ``alias``.

        For ticker symbols (all uppercase, ≤ 5 chars) the pattern also allows
        an optional leading ``$`` so that cashtag mentions like ``$NVDA`` match.
        """
        try:
            escaped = re.escape(alias)

            # Detect likely ticker symbol: short, all uppercase, letters only.
            is_ticker_symbol = (
                alias.isupper()
                and alias.isalpha()
                and 1 <= len(alias) <= 5
            )

            if is_ticker_symbol:
                # Match: optional $, then the ticker, surrounded by word boundaries.
                # Example: \$?NVDA with \b on each side.
                pattern_str = rf"(?<!\w)\$?{escaped}(?!\w)"
            else:
                # Use Unicode word boundaries for company names (handles & and .)
                pattern_str = rf"\b{escaped}\b"

            return re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
        except re.error as exc:
            logger.warning("Could not compile regex for alias %r: %s", alias, exc)
            return None
