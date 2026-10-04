"""
RiskSignal — the frozen contract between all FinRisk components.

Design decisions
────────────────
- ``model_config = ConfigDict(frozen=True)`` makes every instance immutable
  after construction.  Any component that needs to "update" a signal must
  create a new one via ``signal.model_copy(update={…})``.
- All floats with bounded ranges carry field-level validators rather than
  ``Annotated`` wrappers so the constraints are self-documenting.
- ``event_probs`` values must be non-negative and sum to ≤ 1.0 (they may
  not cover all event types simultaneously).
- ``impact_range`` is validated so that low ≤ impact ≤ high.
- ``ts_published`` and ``ts_processed`` are stored as timezone-aware UTC
  datetimes; naive datetimes are rejected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Supporting models ─────────────────────────────────────────────────────────

class TickerRef(BaseModel):
    """A single equity ticker with an optional exchange qualifier."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Canonical uppercase ticker symbol, e.g. 'AAPL'.",
    )
    exchange: str | None = Field(
        default=None,
        max_length=10,
        description="Exchange mic or suffix, e.g. 'NASDAQ'.  Optional.",
    )
    relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How strongly this ticker is associated with the event (0–1).",
    )

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class DiffusionInfo(BaseModel):
    """
    Captures how broadly and quickly a signal is spreading across
    information sources at the time it was processed.
    """

    model_config = ConfigDict(frozen=True)

    sources: int = Field(
        ...,
        ge=1,
        description="Number of distinct sources that have picked up this event.",
    )
    velocity: float = Field(
        ...,
        ge=0.0,
        description=(
            "Rate of new source adoption per hour.  0 = static, higher = viral."
        ),
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Composite diffusion score normalised to 0–1. "
            "Combines source count and velocity relative to recent baselines."
        ),
    )


# ── Primary contract ──────────────────────────────────────────────────────────

class RiskSignal(BaseModel):
    """
    The canonical risk signal produced by the NLP engine and consumed by
    Module A (rebalancer), Module B (stress tester), and the dashboard.

    This schema is intentionally frozen — downstream consumers may cache
    instances safely.  The NLP engine owns creation; all other components
    are read-only consumers.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True, protected_namespaces=())

    # ── Identity ──────────────────────────────────────────────────────────────
    event_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Globally unique identifier for this event.  Typically a UUIDv4 "
            "string, but the schema does not enforce UUID format to allow "
            "deterministic IDs during testing."
        ),
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    ts_published: datetime = Field(
        ...,
        description="When the underlying article / filing / post was published (UTC).",
    )
    ts_processed: datetime = Field(
        ...,
        description="When the NLP engine finished processing and emitted this signal (UTC).",
    )

    # ── Source ────────────────────────────────────────────────────────────────
    source: Literal["news", "social", "filing"] = Field(
        ...,
        description="The class of source that originated the event.",
    )

    # ── Tickers ───────────────────────────────────────────────────────────────
    tickers: list[TickerRef] = Field(
        ...,
        min_length=1,
        description="One or more equity tickers materially related to this event.",
    )

    # ── Sentiment ─────────────────────────────────────────────────────────────
    sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Signed sentiment score. "
            "-1.0 = maximally negative, 0 = neutral, +1.0 = maximally positive."
        ),
    )
    sentiment_conf: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the sentiment score (0–1).",
    )

    # ── Event classification ──────────────────────────────────────────────────
    event_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Primary event category label, e.g. 'Earnings', 'M&A', 'Regulatory'. "
            "Must be the argmax key of ``event_probs``."
        ),
    )
    event_probs: dict[str, float] = Field(
        ...,
        description=(
            "Distribution over event type labels.  Values must be non-negative "
            "and sum to ≤ 1.0.  The argmax label should match ``event_type``."
        ),
    )

    # ── Impact ────────────────────────────────────────────────────────────────
    impact: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Point estimate of event impact on a 1–10 scale.",
    )
    impact_range: tuple[float, float] = Field(
        ...,
        description=(
            "95 % credible interval for impact: (low, high).  "
            "Both must be within [1, 10] and low ≤ impact ≤ high."
        ),
    )

    # ── Signal quality ────────────────────────────────────────────────────────
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall signal confidence combining sentiment and classification quality.",
    )
    novelty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How novel this event is relative to recent signals in the same cluster "
            "(0 = complete duplicate, 1 = entirely new information)."
        ),
    )

    # ── Diffusion ─────────────────────────────────────────────────────────────
    diffusion: DiffusionInfo = Field(
        ...,
        description="How broadly and quickly this signal is spreading.",
    )

    # ── Decay ─────────────────────────────────────────────────────────────────
    decay_halflife_hours: float = Field(
        ...,
        gt=0.0,
        description=(
            "Expected half-life for signal relevance in hours.  "
            "Short-lived events (earnings surprise) → small values; "
            "structural events (regulatory change) → large values."
        ),
    )

    # ── Narrative ─────────────────────────────────────────────────────────────
    explanation: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional human-readable explanation generated by the NLP engine.",
    )

    # ── Clustering ────────────────────────────────────────────────────────────
    cluster_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Identifier for the topic/event cluster this signal belongs to. "
            "Signals about the same underlying event share a cluster_id."
        ),
    )

    # ── Provenance ────────────────────────────────────────────────────────────
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of model component → version string, e.g. "
            "{'sentiment': 'finbert-v2.1', 'classifier': 'roberta-base-v3'}."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("ts_published", "ts_processed", mode="before")
    @classmethod
    def _require_utc(cls, v: datetime | str) -> datetime:
        """Accept ISO strings and coerce them; reject naive datetimes."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                raise ValueError(
                    "Timestamps must be timezone-aware (UTC). "
                    f"Received naive datetime: {v!r}"
                )
        return v

    @field_validator("event_probs")
    @classmethod
    def _validate_event_probs(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("event_probs must not be empty.")
        for label, prob in v.items():
            if prob < 0.0:
                raise ValueError(
                    f"Probability for '{label}' is negative ({prob}). "
                    "All event_probs values must be ≥ 0."
                )
        total = sum(v.values())
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"event_probs values sum to {total:.4f}, which exceeds 1.0."
            )
        return v

    @field_validator("impact_range")
    @classmethod
    def _validate_impact_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        low, high = v
        if low < 1.0 or high > 10.0:
            raise ValueError(
                f"impact_range ({low}, {high}) must lie within [1.0, 10.0]."
            )
        if low > high:
            raise ValueError(
                f"impact_range low ({low}) must be ≤ high ({high})."
            )
        return v

    @model_validator(mode="after")
    def _event_type_consistent(self) -> "RiskSignal":
        """event_type must be the argmax of event_probs."""
        if self.event_probs:
            best_label = max(self.event_probs, key=lambda k: self.event_probs[k])
            if best_label != self.event_type:
                raise ValueError(
                    f"event_type '{self.event_type}' does not match the argmax of "
                    f"event_probs ('{best_label}'). "
                    "Set event_type to the highest-probability label."
                )
        return self

    @model_validator(mode="after")
    def _impact_within_range(self) -> "RiskSignal":
        """Point estimate must lie within its own credible interval."""
        low, high = self.impact_range
        if not (low <= self.impact <= high):
            raise ValueError(
                f"impact ({self.impact}) must be within impact_range "
                f"({low}, {high})."
            )
        return self

    @model_validator(mode="after")
    def _ts_order(self) -> "RiskSignal":
        """Processing timestamp must not precede publication timestamp."""
        if self.ts_processed < self.ts_published:
            raise ValueError(
                "ts_processed cannot be earlier than ts_published. "
                f"published={self.ts_published.isoformat()}, "
                f"processed={self.ts_processed.isoformat()}"
            )
        return self
