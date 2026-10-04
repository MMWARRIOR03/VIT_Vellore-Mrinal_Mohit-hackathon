"""
Rule-based event classifier (v1).

Classifies financial text into one of 10 event categories by counting keyword
matches, then normalises the counts into a probability distribution.  This is
the Day 1 baseline; Task 6 replaces it with a DeBERTa fine-tuned classifier.

Design notes
────────────
- Keywords use substring matching on lowercased text to catch morphological
  variants without a stemmer (e.g. "restructur" catches "restructuring",
  "restructured", "restructure").
- Softmax normalisation ensures probabilities sum to ≤ 1.0 while preserving
  the relative ordering of categories.
- A minimum floor probability is applied to the winning category so that even
  single-hit matches register a non-trivial probability.
"""

from __future__ import annotations

import math
import re

# ── Keyword definitions ───────────────────────────────────────────────────────

RULES: dict[str, list[str]] = {
    "Geopolitical": [
        "sanction", "war", "tariff", "export ban", "military", "conflict",
        "invasion", "nato", "geopolit", "treaty", "diplomat", "embargo",
        "trade war", "hostile",
    ],
    "Regulatory": [
        "investigation", "fine", "sec", "compliance", "ban", "antitrust",
        "lawsuit", "penalty", "probe", "ftc", "doj", "regulation",
        "enforcement", "subpoena", "consent decree", "settlement",
    ],
    "M&A": [
        "acquisition", "merger", "takeover", "buyout", "acquire", "deal",
        "purchase", "tender offer", "spinoff", "divest", "carve-out",
        "strategic combination", "all-cash offer", "all-stock",
    ],
    "Earnings": [
        "revenue", "earnings", "profit", "loss", "guidance", "eps",
        "quarterly", "beat", "miss", "outlook", "q1", "q2", "q3", "q4",
        "fiscal year", "annual results", "sales growth", "operating income",
    ],
    "Credit": [
        "default", "downgrade", "bankruptcy", "debt", "credit rating",
        "junk", "moody", "s&p", "fitch", "restructur", "bondholder",
        "covenant", "high-yield", "leverage", "credit watch",
    ],
    "Macro": [
        "fed", "federal reserve", "interest rate", "inflation", "gdp",
        "unemployment", "cpi", "fomc", "monetary policy", "recession",
        "rate hike", "rate cut", "central bank", "yield curve", "treasury",
        "quantitative easing", "tapering",
    ],
    "Product": [
        "launch", "recall", "release", "patent", "fda", "approval",
        "product", "device", "drug", "clinical trial", "pipeline",
        "feature", "version", "platform", "service launch",
    ],
    "Cyber": [
        "hack", "breach", "ransomware", "data leak", "cyberattack",
        "cybersecurity", "malware", "intrusion", "phishing", "zero-day",
        "data theft", "credential", "exploit",
    ],
    "Leadership": [
        "ceo", "cfo", "resign", "appoint", "executive", "director",
        "departure", "succession", "promoted", "c-suite", "board member",
        "interim", "named", "steps down", "chief operating",
    ],
    "Commodity": [
        "oil", "gas", "gold", "silver", "copper", "commodit", "supply chain",
        "opec", "energy price", "crude", "natural gas", "wheat", "corn",
        "lithium", "rare earth", "barrel",
    ],
}

# Decay half-lives in hours per event type.
# Short-lived: earnings surprises fade in hours; structural: regulatory changes persist.
_DECAY_HALFLIFE: dict[str, float] = {
    "Earnings": 12.0,
    "M&A": 48.0,
    "Regulatory": 72.0,
    "Macro": 24.0,
    "Geopolitical": 36.0,
    "Credit": 48.0,
    "Commodity": 18.0,
    "Cyber": 24.0,
    "Leadership": 36.0,
    "Product": 18.0,
}

_MIN_PRIMARY_PROB = 0.02  # floor for the winning category


class RuleBasedEventClassifier:
    """
    Keyword-frequency event classifier.

    ``classify()`` returns a probability dict over the 10 categories.
    ``get_primary()`` returns the top category label alongside the full dict.
    """

    def __init__(self) -> None:
        # Pre-compile patterns once at construction time for speed.
        # Each entry maps category → list of (pattern, weight) tuples.
        self._patterns: dict[str, list[re.Pattern[str]]] = {
            cat: [re.compile(re.escape(kw.lower())) for kw in kws]
            for cat, kws in RULES.items()
        }

    def _raw_scores(self, text: str) -> dict[str, float]:
        """Count keyword hits per category (weighted by match count)."""
        text_lower = text.lower()
        scores: dict[str, float] = {}
        for cat, patterns in self._patterns.items():
            count = sum(
                len(pattern.findall(text_lower)) for pattern in patterns
            )
            scores[cat] = float(count)
        return scores

    def classify(self, text: str) -> dict[str, float]:
        """
        Return a probability distribution over all 10 event categories.

        Uses softmax over raw keyword hit counts so that categories with
        more matches receive higher probabilities while all values sum to ≤ 1.0.
        Categories with zero hits receive a near-zero but positive probability.
        """
        raw = self._raw_scores(text)

        # Softmax: exp(score) / sum(exp(scores))
        # Add a small epsilon to avoid zero-mass categories collapsing entirely.
        epsilon = 0.1
        adjusted = {cat: raw[cat] + epsilon for cat in raw}

        exp_vals = {cat: math.exp(min(v, 20.0)) for cat, v in adjusted.items()}
        total = sum(exp_vals.values())

        probs = {cat: ev / total for cat, ev in exp_vals.items()}

        # Apply minimum floor to winning category
        best_cat = max(probs, key=lambda k: probs[k])
        if probs[best_cat] < _MIN_PRIMARY_PROB:
            probs[best_cat] = _MIN_PRIMARY_PROB
            # Re-normalise (the floor is tiny so numeric impact is negligible)
            total_after = sum(probs.values())
            probs = {cat: v / total_after for cat, v in probs.items()}

        # Sanity check: ensure sum ≤ 1.0 (floating-point safety)
        total_final = sum(probs.values())
        if total_final > 1.0 + 1e-6:
            probs = {cat: v / total_final for cat, v in probs.items()}

        return probs

    def get_primary(self, text: str) -> tuple[str, dict[str, float]]:
        """
        Return (primary_event_type, event_probs).

        The primary event type is the argmax of the probability distribution.
        """
        probs = self.classify(text)
        primary = max(probs, key=lambda k: probs[k])
        return primary, probs

    @staticmethod
    def decay_halflife_for_event(event_type: str) -> float:
        """
        Return the expected signal half-life in hours for the given event type.

        Falls back to 24 hours for unknown types.
        """
        return _DECAY_HALFLIFE.get(event_type, 24.0)
