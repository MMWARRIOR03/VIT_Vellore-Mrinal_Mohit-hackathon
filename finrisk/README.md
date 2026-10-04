# FinRisk Intelligence Engine

FinRisk is a real-time financial risk intelligence system that converts raw news, social, and regulatory filings into structured risk signals, then routes those signals through two downstream modules: a signal-driven index rebalancer (Module A) and a portfolio stress tester with Monte Carlo simulation (Module B). The system runs fully offline in replay mode — no API keys required for the demo — and can be switched to live ingestion from GDELT, SEC EDGAR, and StockTwits via a single Docker Compose profile flag.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Sources                                │
│   GDELT News ──┐   SEC EDGAR ──┐   StockTwits ──┐   Kaggle Tweets  │
└────────────────┼───────────────┼────────────────┼──────────────────┘
                 │               │                │
                 └───────────────┴────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Ingestion Manager     │
                    │  (dedup • entity link)  │
                    └────────────┬────────────┘
                                 │ raw articles
                    ┌────────────▼────────────┐
                    │      NLP Worker         │
                    │  FinBERT sentiment      │
                    │  RuleBased/DeBERTa cls  │
                    │  XGBoost impact model   │
                    │  Novelty + Diffusion    │
                    └────────────┬────────────┘
                                 │ RiskSignal (frozen Pydantic v2)
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────┐
   │  Module A        │  │   Module B     │  │  DuckDB    │
   │  IndexRebalancer │  │  StressEngine  │  │  storage   │
   │  (FinRisk-20)    │  │  + MC Engine   │  └─────┬──────┘
   └──────────┬──────┘  └───────┬────────┘        │
              │                 │                  │
              └─────────────────┴──────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Redis Streams       │
                    │  (processed.signals   │
                    │   signals.stress)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Streamlit Dashboard  │
                    │  Live feed • Module A │
                    │  Module B • Explorer  │
                    └───────────────────────┘
```

## Quick Start

### One-command demo (no internet required)

```bash
git clone https://github.com/MMWARRIOR03/VIT_Vellore-Mrinal_Mohit-hackathon.git
cd VIT_Vellore-Mrinal_Mohit-hackathon/finrisk
cp .env.example .env
docker compose up --build
```

Access:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

The default `.env` runs in synthetic signal mode (`FAKE_SIGNAL_ENABLED=true`), so no external data sources are needed.

### With live data ingestion

```bash
docker compose --profile live up
```

This adds the `ingestion` container, which polls GDELT (every 15 min), SEC EDGAR (every 30 min), and StockTwits (every 5 min). Set `GROQ_API_KEY` in `.env` to enable LLM-generated explanations.

### Local development (without Docker)

```bash
make install           # create .venv and install dependencies
make redis-up          # start Redis in Docker
make dev               # FastAPI on :8000 with hot reload
make dashboard         # Streamlit on :8501
```

To run the fake signal generator as a separate process:

```bash
make dev-gen
```

## Environment Variables

| Variable | Description | Default | Required |
|---|---|---|---|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` | Yes |
| `DUCKDB_PATH` | DuckDB file path | `./data/finrisk.duckdb` | Yes |
| `LLM_PROVIDER` | LLM for explanations: `groq`, `ollama`, `none` | `none` | No |
| `GROQ_API_KEY` | Groq API key (free tier available) | — | Only if `LLM_PROVIDER=groq` |
| `OLLAMA_BASE_URL` | Ollama endpoint | `http://host.docker.internal:11434` | Only if `LLM_PROVIDER=ollama` |
| `REPLAY_MODE` | Use offline synthetic replay instead of live ingestion | `false` | No |
| `REPLAY_SPEED_MULTIPLIER` | Replay speed multiplier over wall-clock time | `10.0` | No |
| `NLP_WORKER_ENABLED` | Run NLP worker on API startup | `true` | No |
| `MC_N_PATHS` | Monte Carlo simulation paths for Module B | `10000` | No |
| `EVENT_CLASSIFIER` | Event classifier backend: `rule-based` or `deberta` | `rule-based` | No |
| `FAKE_SIGNAL_ENABLED` | Generate synthetic signals automatically | `true` | No |

## Testing

```bash
make test                    # unit tests (no Redis/DB needed)
make test-integration        # integration tests (Redis required)
make test-cov                # coverage report
```

Tests live in `tests/` and are split by `pytest.mark.integration` for the Redis-dependent ones.

## Training the Impact Model (optional)

The default configuration uses a rule-based impact estimator. To train the XGBoost model on real GDELT + price data:

```bash
python scripts/fetch_training_data.py    # ~30 min, downloads GDELT + yfinance prices
python scripts/build_training_labels.py  # compute abnormal returns as training targets
python scripts/train_impact_model.py     # train XGBoost, saves to data/models/
```

Activate the trained model by setting `MODEL_IMPACT=xgboost` in `.env`.

## Evaluation

```bash
python scripts/evaluate.py             # sentiment + event + impact metrics
python scripts/backtest_module_a.py    # Module A IC / Sharpe / drawdown
python scripts/validate_module_b.py   # Module B historical scenario validation
```

All three scripts are standalone (no Redis or running API required) and write results to `data/results/`.

## Demo Injection

Inject a custom headline directly into the processing pipeline:

```bash
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"text": "NVIDIA faces major export restrictions on AI chips", "ticker": "NVDA"}'
```

The response includes the full `RiskSignal` JSON, the stress test result (if impact ≥ 7), and the LLM-generated explanation (if configured).

## Project Structure

```
finrisk/
├── app/                    # FastAPI application
│   ├── config.py           # Pydantic-settings environment config
│   ├── engine/             # NLP pipeline: sentiment, classifier, impact, novelty
│   ├── generators/         # Synthetic signal generator for demo/testing
│   ├── ingestion/          # Data source adapters (GDELT, EDGAR, StockTwits, Kaggle)
│   ├── llm/                # Groq/Ollama explanation layer
│   ├── replay/             # Offline replay runner
│   ├── routes/             # FastAPI endpoints
│   ├── schema/             # RiskSignal frozen Pydantic v2 contract
│   ├── storage/            # DuckDB persistence layer
│   └── streams/            # Redis Streams producer/consumer
├── module_a/               # Index rebalancer (FinRisk-20 tactical index)
│   ├── rebalancer.py       # IndexRebalancer orchestrator
│   ├── signal_aggregator.py# Decay-weighted signal accumulation
│   ├── weight_calculator.py# Z-score → exponential tilt → constrained weights
│   └── price_cache.py      # yfinance price cache for vol estimation
├── module_b/               # Portfolio stress tester
│   ├── stress_engine.py    # StressEngine orchestrator
│   ├── scenarios.py        # Shock vectors loaded from configs/scenarios.yaml
│   ├── portfolio.py        # Deterministic 300-position synthetic portfolio
│   ├── valuation.py        # First-order P&L calculator
│   ├── monte_carlo.py      # Vectorised 5-factor Student-t MC simulation
│   ├── trigger.py          # Gate with per-(ticker, event_type) cooldown
│   └── reverse_stress.py   # Find minimum impact that breaches loss limit
├── dashboard/              # Streamlit dashboard (4 tabs)
├── scripts/                # Training, evaluation, and data generation
│   ├── evaluate.py         # NLP model evaluation metrics
│   ├── backtest_module_a.py# Module A IC / Sharpe / drawdown backtest
│   ├── validate_module_b.py# Module B historical scenario validation
│   ├── fetch_training_data.py
│   ├── build_training_labels.py
│   ├── train_impact_model.py
│   ├── train_event_classifier.py
│   └── generate_sample_data.py
├── configs/                # YAML configuration files
│   ├── index.yaml          # FinRisk-20 constituent tickers + base weights
│   ├── scenarios.yaml      # Stress scenario shock vectors (10 event types)
│   ├── portfolio.yaml      # Synthetic portfolio construction parameters
│   └── ticker_map.yaml     # Ticker → sector + exchange mapping
├── tests/                  # pytest unit + integration tests
├── docker/                 # Per-service Dockerfiles
├── data/
│   ├── models/             # Trained model artefacts (.pkl, .json)
│   ├── sample/             # Offline replay fixtures (.jsonl)
│   └── results/            # Evaluation and backtest output (.json)
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── .env.example
```

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI 0.111 + Uvicorn |
| Schema validation | Pydantic v2 (frozen models) |
| Message bus | Redis Streams |
| Storage | DuckDB (embedded columnar) |
| Sentiment model | ProsusAI/FinBERT (HuggingFace Transformers) |
| Event classifier | Rule-based keyword matcher; optional DeBERTa fine-tune |
| Impact model | XGBoost regressor; rule-based fallback |
| Monte Carlo | NumPy vectorised, Student-t(df=5), Cholesky-correlated |
| Dashboard | Streamlit 1.35 + Plotly 5.22 |
| LLM explanations | Groq (llama-3.1-8b-instant) or Ollama (qwen3:4b) |
| Containerisation | Docker Compose (5 services) |
| Testing | pytest + pytest-asyncio + pytest-cov |
| Data ingestion | GDELT, SEC EDGAR, StockTwits, Kaggle tweet replay |

## Key Design Decisions

- **Schema-first**: `RiskSignal` is a frozen Pydantic v2 model. It is the single contract between the NLP engine, Module A, Module B, the dashboard, and Redis. Nothing downstream can mutate it.
- **Staged impact model**: Day 1 uses a rule-based heuristic (BASE_IMPACT by event type × sentiment magnitude). The XGBoost model is hot-swapped by setting `MODEL_IMPACT=xgboost` — no code changes required.
- **Reproducible demo**: `REPLAY_MODE=true` uses pre-generated `data/sample/signals.jsonl` with no network calls. `FakeSignalGenerator` produces statistically plausible but entirely synthetic signals when no real data is available.
- **Module B is the showpiece**: The stress test combines a deterministic first-order P&L pass (covering bonds, loans, equities, FX, and swaps) with a vectorised 5-factor Monte Carlo for tail-risk metrics. The MC run is offloaded to a thread-pool executor so it never blocks the FastAPI event loop.
- **LLM as explainer only**: Groq or Ollama generates the human-readable `explanation` field. Core signal generation, scoring, and risk computation are entirely deterministic and LLM-free. The system degrades gracefully to `LLM_PROVIDER=none`.
- **Cooldown gate**: The `StressTrigger` imposes a 30-minute cooldown per (ticker, event_type) pair after each stress run. This prevents a burst of syndicated news articles about the same event from triggering 10 identical stress runs during a demo.

## Evaluation Results

Replace these placeholders with actual numbers after running `python scripts/evaluate.py`:

| Component | Metric | Value |
|---|---|---|
| Sentiment (FinBERT, Financial PhraseBank) | Accuracy | — |
| Sentiment (FinBERT, Financial PhraseBank) | Macro F1 | — |
| Event classifier (rule-based) | Macro F1 | — |
| Event classifier (rule-based) | Accuracy | — |
| Impact model (rule-based baseline) | MAE | — |
| Impact model (XGBoost, if trained) | MAE | — |
| Impact model (XGBoost, if trained) | Spearman ρ | — |
| Module A (signal-tilted, 200 signals) | Sharpe (ann.) | — |
| Module A (signal-tilted, 200 signals) | Max drawdown | — |
| Module A | IC (Spearman) | — |

Run evaluation scripts to populate:

```bash
python scripts/evaluate.py && python scripts/backtest_module_a.py && python scripts/validate_module_b.py
```

Results are saved as JSON in `data/results/`.

## Limitations and Honest Notes

- **Tweet data**: The Kaggle tweet adapter replays a static CSV. Live Twitter/X access is not available via free tiers. The replay approximates social sentiment velocity but is not real-time.
- **yfinance reliability**: Price data for vol estimation in Module A is fetched from yfinance, which has occasional rate limits and data gaps. The `PriceCache` returns synthetic prices when yfinance is unavailable, which degrades vol-adjusted scoring quality.
- **Look-ahead risk**: The synthetic signal generator and backtest use normally-distributed synthetic returns that are not calibrated to real equity covariances. Backtest Sharpe ratios are indicative only and should not be taken as real performance claims.
- **Impact model calibration**: The XGBoost model is trained on GDELT tone scores correlated against yfinance abnormal returns. Both data sources carry noise and structural biases (GDELT over-represents English-language media; yfinance returns are unadjusted for dividends in some tickers). Spearman correlations around 0.4–0.6 are realistic expectations.
- **Portfolio is synthetic**: Module B operates on a deterministically generated $50M wholesale banking book. Sector concentrations, credit spreads, and beta values are plausible but fictional. Stress results are not intended as real risk estimates.
- **No real-time pricing**: Module B does not update position market values using live prices. The portfolio is marked at construction time (seed=42) and held constant across all stress runs.
- **Single-threaded rebalancer**: `IndexRebalancer` is designed for a single asyncio coroutine. It is not thread-safe and should not be shared across concurrent tasks.
