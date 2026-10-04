"""
Central configuration — all settings read from environment / .env file.
Uses pydantic-settings v2 so every field is type-validated at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration surfaces here.  Import ``get_settings()`` everywhere
    rather than constructing ``Settings()`` directly — the lru_cache ensures
    a single instance per process.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown env vars
        case_sensitive=False,
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Full Redis connection URL.",
    )
    redis_stream_maxlen: int = Field(
        default=10_000,
        ge=100,
        description="MAXLEN cap for every stream (approximate trim).",
    )

    # ── DuckDB ────────────────────────────────────────────────────────────────
    duckdb_path: Path = Field(
        default=Path("./data/finrisk.duckdb"),
        description="Filesystem path to the DuckDB database file.",
    )

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_reload: bool = Field(default=False)

    # ── Fake signal generator ─────────────────────────────────────────────────
    fake_signal_interval_seconds: float = Field(
        default=5.0,
        ge=0.1,
        description="Seconds between synthetic signal emissions.",
    )
    fake_signal_enabled: bool = Field(default=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── NLP (placeholder for Task 2) ─────────────────────────────────────────
    nlp_model_name: str = Field(default="placeholder/finbert")
    nlp_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # ── Event classifier (Task 6) ─────────────────────────────────────────────
    event_classifier: str = Field(
        default="rule-based",
        description=(
            "Which event classifier to use: 'rule-based' or 'deberta'. "
            "Set to 'deberta' to enable the fine-tuned DeBERTa model."
        ),
    )
    deberta_model_path: str = Field(
        default="data/models/event_deberta",
        description="Path to the fine-tuned DeBERTa event classifier checkpoint.",
    )

    # ── Ingestion layer (Task 2) ──────────────────────────────────────────────
    ingestion_enabled: bool = Field(
        default=False,
        description="Enable the real-time ingestion pipeline on startup.",
    )
    kaggle_tweets_csv_path: str = Field(
        default="./data/sample/tweets.csv",
        description="Path to the Kaggle tweet replay CSV file.",
    )
    kaggle_replay_speed: float = Field(
        default=10.0,
        ge=0.01,
        description="Tweet replay speed multiplier (1.0 = real time).",
    )
    gdelt_poll_interval_seconds: float = Field(
        default=900.0,
        ge=60.0,
        description="How often (seconds) GDELT is polled.",
    )
    edgar_poll_interval_seconds: float = Field(
        default=1800.0,
        ge=60.0,
        description="How often (seconds) SEC EDGAR is polled.",
    )
    stocktwits_poll_interval_seconds: float = Field(
        default=300.0,
        ge=30.0,
        description="How often (seconds) StockTwits is polled.",
    )
    ingestion_adapters: str = Field(
        default="gdelt,edgar,stocktwits,kaggle",
        description="Comma-separated list of adapters to enable.",
    )

    # ─────────────────────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @field_validator("duckdb_path")
    @classmethod
    def _ensure_duckdb_parent(cls, v: Path) -> Path:
        """Create the parent directory for DuckDB at config-load time."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
