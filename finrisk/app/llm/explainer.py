"""
SignalExplainer — generates human-readable explanations for RiskSignal objects.

Provider chain
──────────────
  1. Groq (API key + model configured)   → fast, cloud-hosted LLM
  2. Ollama (base URL configured)         → local LLM, requires Ollama running
  3. Rule-based template                  → always available, no network

Caching
───────
Explanations are NOT cached in DuckDB between restarts — the DuckDB schema
stores explanation as part of the RiskSignal payload, and the inject endpoint
writes the signal (with explanation) in one shot.  In-process caching is done
by cluster_id so repeated calls within the same process reuse the result.

Prompting strategy
──────────────────
Both LLM providers receive the same system prompt and a compact user message
containing the ticker, event type, sentiment, impact, and the original headline.
The expected response is a JSON object ``{"explanation": "..."}`` to make
extraction reliable regardless of model verbosity.

Error handling
──────────────
Each provider returns None on any failure (network error, timeout, malformed
response, missing API key).  Failures are logged at WARNING and the chain falls
through to the next provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.schema.risk_signal import RiskSignal

logger = logging.getLogger(__name__)

# Timeout for LLM calls — long enough for Ollama on modest hardware
_GROQ_TIMEOUT_S = 15.0
_OLLAMA_TIMEOUT_S = 30.0

_SYSTEM_PROMPT = (
    "You are a concise financial risk analyst. "
    "Given a news event signal, write a 2-sentence explanation of what happened "
    "and why it matters for the ticker. "
    "Be factual and use plain business language. "
    'Respond ONLY in valid JSON with a single key: {"explanation": "..."}'
)


class SignalExplainer:
    """
    Generates 2-sentence explanations for RiskSignal objects.

    Instantiate once and attach to app.state.explainer.
    """

    def __init__(
        self,
        provider: str = "none",
        groq_api_key: str = "",
        groq_model: str = "llama-3.1-8b-instant",
        ollama_base_url: str = "http://host.docker.internal:11434",
        ollama_model: str = "qwen3:4b",
    ) -> None:
        self._provider = provider.lower().strip()
        self._groq_api_key = groq_api_key
        self._groq_model = groq_model
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._ollama_model = ollama_model
        # In-process cache: cluster_id → explanation string
        self._cache: dict[str, str] = {}
        # Rate-limit guard: track last Groq call time
        self._last_groq_call: float = 0.0
        self._min_interval_s: float = 3.0  # max ~20 RPM, well under 30 RPM free tier

        logger.info(
            "SignalExplainer initialised: provider=%r, groq_model=%r, ollama_model=%r",
            self._provider, self._groq_model, self._ollama_model,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def explain(self, signal: "RiskSignal", headline: str = "") -> str:
        """
        Return a 2-sentence explanation for ``signal``.

        Falls through Groq → Ollama → template, returning the first success.
        Results are cached by cluster_id within the process lifetime.
        """
        # Check in-process cache first
        cached = self._cache.get(signal.cluster_id)
        if cached:
            logger.debug("SignalExplainer cache hit for cluster_id=%s", signal.cluster_id)
            return cached

        explanation: str | None = None

        if self._provider == "groq" and self._groq_api_key:
            explanation = await self._explain_groq(signal, headline)
            if explanation:
                logger.debug("Groq explanation for cluster_id=%s", signal.cluster_id)

        if explanation is None and self._provider == "ollama":
            explanation = await self._explain_ollama(signal, headline)
            if explanation:
                logger.debug("Ollama explanation for cluster_id=%s", signal.cluster_id)

        if explanation is None:
            explanation = self._template_explanation(signal)
            logger.debug("Template explanation for cluster_id=%s", signal.cluster_id)

        self._cache[signal.cluster_id] = explanation
        # Prevent unbounded growth in long-running processes
        if len(self._cache) > 5_000:
            # Evict oldest 1000 entries (dict insertion order in Python 3.7+)
            for key in list(self._cache.keys())[:1000]:
                del self._cache[key]

        return explanation

    # ── Groq ───────────────────────────────────────────────────────────────────

    async def _explain_groq(self, signal: "RiskSignal", headline: str) -> str | None:
        """POST to Groq OpenAI-compatible chat completions endpoint."""
        import time
        # Enforce minimum inter-call interval to stay under free-tier 30 RPM
        now = time.monotonic()
        elapsed = now - self._last_groq_call
        if elapsed < self._min_interval_s:
            await asyncio.sleep(self._min_interval_s - elapsed)
        user_msg = _build_user_message(signal, headline)
        payload = {
            "model": self._groq_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 120,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self._groq_api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=_GROQ_TIMEOUT_S) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
            if resp.status_code != 200:
                logger.warning(
                    "Groq API returned %d: %s", resp.status_code, resp.text[:200]
                )
                return None
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            self._last_groq_call = time.monotonic()  # update after success
            return _extract_explanation(content)
        except httpx.TimeoutException:
            logger.warning("Groq request timed out after %.1fs", _GROQ_TIMEOUT_S)
            return None
        except Exception as exc:
            logger.warning("Groq explain failed: %s", exc)
            return None

    # ── Ollama ─────────────────────────────────────────────────────────────────

    async def _explain_ollama(self, signal: "RiskSignal", headline: str) -> str | None:
        """POST to a locally-running Ollama /api/generate endpoint."""
        prompt = f"{_SYSTEM_PROMPT}\n\n{_build_user_message(signal, headline)}"
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 120},
        }
        try:
            async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_S) as client:
                resp = await client.post(
                    f"{self._ollama_base_url}/api/generate",
                    json=payload,
                )
            if resp.status_code != 200:
                logger.warning(
                    "Ollama returned %d: %s", resp.status_code, resp.text[:200]
                )
                return None
            data = resp.json()
            content = data.get("response", "")
            return _extract_explanation(content)
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out after %.1fs", _OLLAMA_TIMEOUT_S)
            return None
        except httpx.ConnectError:
            logger.warning(
                "Ollama not reachable at %s — falling back to template",
                self._ollama_base_url,
            )
            return None
        except Exception as exc:
            logger.warning("Ollama explain failed: %s", exc)
            return None

    # ── Template fallback ──────────────────────────────────────────────────────

    def _template_explanation(self, signal: "RiskSignal") -> str:
        """Rule-based explanation — always succeeds, no I/O required."""
        primary_ticker = signal.tickers[0].ticker if signal.tickers else "UNKNOWN"
        sentiment_word = _sentiment_word(signal.sentiment)
        source_label = {
            "news": "a news report",
            "social": "social media activity",
            "filing": "a regulatory filing",
        }.get(signal.source, "a market event")

        line1 = (
            f"A {signal.event_type} event for {primary_ticker} was detected via "
            f"{source_label} with {sentiment_word} sentiment "
            f"(score: {signal.sentiment:+.2f})."
        )
        line2 = (
            f"The signal carries an impact score of {signal.impact:.1f}/10 "
            f"(confidence: {signal.confidence:.0%}) and is expected to remain "
            f"relevant for approximately {signal.decay_halflife_hours:.0f} hours."
        )
        return f"{line1} {line2}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_user_message(signal: "RiskSignal", headline: str) -> str:
    """Compact signal summary sent as the user turn to the LLM."""
    primary_ticker = signal.tickers[0].ticker if signal.tickers else "UNKNOWN"
    headline_part = f'Headline: "{headline}"\n' if headline else ""
    return (
        f"{headline_part}"
        f"Ticker: {primary_ticker}\n"
        f"Event type: {signal.event_type}\n"
        f"Sentiment: {signal.sentiment:+.2f} ({_sentiment_word(signal.sentiment)})\n"
        f"Impact score: {signal.impact:.1f}/10\n"
        f"Source: {signal.source}\n"
        f"Confidence: {signal.confidence:.0%}\n"
    )


def _extract_explanation(content: str) -> str | None:
    """
    Extract the explanation string from LLM output.

    Tries JSON parsing first, then falls back to stripping code fences and
    retrying, then looks for the pattern ``"explanation": "..."`` directly.
    Returns None if nothing useful can be extracted.
    """
    if not content or not content.strip():
        return None

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()

    # Try full JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "explanation" in data:
            val = data["explanation"]
            if isinstance(val, str) and val.strip():
                return val.strip()
    except json.JSONDecodeError:
        pass

    # Try regex extraction of the value
    match = re.search(r'"explanation"\s*:\s*"([^"]+)"', text)
    if match:
        val = match.group(1).strip()
        if val:
            return val

    # If the model ignored instructions and returned plain text, use it directly
    stripped = text.strip()
    if len(stripped) > 20 and not stripped.startswith("{"):
        return stripped[:500]

    return None


def _sentiment_word(score: float) -> str:
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "negative"
    return "neutral"
