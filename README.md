# FinRisk Intelligence Engine - S&P Global & Crisil Campus Hackathon 2026

**Candidate Name:** Mrinal Mohit
**College Email ID:** mrinal.23bce11545@vitbhopal.ac.in
**College / Campus:** VIT Vellore
**Demo Video Link:** [YouTube Walkthrough — PLACEHOLDER: add link before submission]
**Slide Deck Link:** [Presentation Deck — PLACEHOLDER: add link before submission]

---

## 1. Project Overview / Problem Statement & Approach

Financial institutions are drowning in unstructured text — news articles, SEC filings, social media posts — with no automated way to convert them into actionable risk signals in real time. A credit analyst at a wholesale bank today must manually scan hundreds of headlines to assess whether any event materially impacts their loan book or trading portfolio. This latency is a direct business risk.

**FinRisk** solves this by building a unified AI/NLP Risk Engine that ingests text from four live sources (GDELT global news, SEC EDGAR filings, StockTwits, Kaggle tweet replay), extracts structured risk signals using FinBERT sentiment + XGBoost impact scoring, and routes those signals into two downstream financial modules: a tactical index rebalancer (Module A) and an event-driven portfolio stress tester with Monte Carlo simulation (Module B).

The system is designed to be reproducible on any evaluator machine with a single `docker compose up` command — no API keys required for the demo. It degrades gracefully: when live data sources are unavailable, a synthetic replay mode generates statistically realistic signals from pre-cached data. The architecture mirrors how real financial risk platforms (RavenPack, Bloomberg BVAL) are structured — a signal extraction layer feeding downstream risk consumers — making it directly applicable to wholesale banking use cases.

---

## 2. Architecture & Tech Stack

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
                    │ SHA-256 dedup • entity  │
                    │ linking • cluster agg   │
                    └────────────┬────────────┘
                                 │ raw.articles  (Redis Stream)
                    ┌────────────▼────────────┐
                    │      NLP Worker         │
                    │  FinBERT sentiment      │
                    │  Rule-based / DeBERTa   │
                    │  XGBoost impact model   │
                    │  Novelty + Diffusion L2 │
                    │  Groq LLM explanation   │
                    └────────────┬────────────┘
                                 │ RiskSignal (frozen Pydantic v2)
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼──────┐  ┌───────▼────────┐  ┌─────▼──────┐
   │  Module A        │  │   Module B     │  │  DuckDB    │
   │  IndexRebalancer │  │  StressEngine  │  │  storage   │
   │  FinRisk-20      │  │  + MC Engine   │  └─────┬──────┘
   │  20-stock index  │  │  + Rev. Stress │        │
   └──────────┬──────┘  └───────┬────────┘        │
              │                 │                  │
              └─────────────────┴──────────────────┘
                                │  processed.signals / signals.stress
                    ┌───────────▼───────────┐
                    │  Streamlit Dashboard  │
                    │  Tab 1: Live Feed     │
                    │  Tab 2: Explorer      │
                    │  Tab 3: Module A      │
                    │  Tab 4: Module B      │
                    └───────────────────────┘
```

See `docs/architecture.png` for the high-resolution diagram.

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI 0.111 + Uvicorn |
| Schema contract | Pydantic v2 (frozen `RiskSignal`) |
| Message bus | Redis Streams (4 topics) |
| Storage | DuckDB (embedded, zero-config) |
| Sentiment model | ProsusAI/FinBERT (HuggingFace) |
| Event classifier | Rule-based (10 types); optional DeBERTa fine-tune |
| Impact model | XGBoost regressor (market-calibrated); rule-based fallback |
| Diffusion scoring | Hawkes-lite L1+L2 exponential decay intensity |
| Monte Carlo | NumPy vectorised, Student-t(df=5), Cholesky-correlated, 10k paths |
| Dashboard | Streamlit 1.35 + Plotly 5.22 |
| LLM explanations | Groq `qwen/qwen3.8-27b` (free tier, rate-limited) |
| Containerisation | Docker Compose (5 services) |
| Testing | pytest + pytest-asyncio — 330+ unit tests |
| Data ingestion | GDELT, SEC EDGAR, StockTwits, Kaggle tweet replay |

---

## 3. Dataset Used

| Source | Type | Usage |
|--------|------|-------|
| GDELT 2.0 DOC API | Public, real-time | Primary news signal source (15-min updates, no key) |
| SEC EDGAR full-text search | Public, real-time | Corporate event labels from 8-K filings (item codes) |
| StockTwits public stream | Public, no key | Live social sentiment for 20 tickers |
| Kaggle "Tweet Sentiment on Stock Returns" | Public CSV (~862k tweets) | Replayed as social stream for backtesting |
| yfinance | Public, unofficial | Daily/hourly OHLCV for 20 tickers + SPY benchmark |
| Synthetic portfolio (seed=42) | Generated | 300-position wholesale banking book for Module B |
| Synthetic signals | Generated | `FakeSignalGenerator` for offline demo mode |

**Assumptions:**
- GDELT tone scores (V2Tone, range −10 to +10) are normalised to [−1, +1] as sentiment proxy
- Abnormal return = stock return − SPY return over 1h/1d windows; used as XGBoost training target
- The synthetic portfolio uses plausible but fictional notional values, durations, and credit spreads
- No real client data, no proprietary feeds, no paid APIs are used anywhere in the system

---

## 4. Quickstart & Installation

**Runtime:** Python 3.11 | macOS / Linux | Docker 24+

---

### Option A — One-command Docker demo (recommended, no internet required)

```bash
git clone https://github.com/MMWARRIOR03/VIT_Vellore-Mrinal_Mohit-hackathon.git
cd VIT_Vellore-Mrinal_Mohit-hackathon/finrisk

# Copy environment file (defaults work out of the box)
cp .env.example .env

# Start all services
docker compose up --build
```

**Access:**
| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI (REST + Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

The default `.env` uses `FAKE_SIGNAL_ENABLED=true` — synthetic signals start flowing immediately, no external APIs needed. Module B manual stress test works out of the box.

---

### Option B — With live data ingestion

```bash
# Start all 5 services including live GDELT/EDGAR/StockTwits polling
docker compose --profile live up

# Optional: add your Groq API key for LLM explanations on signals
# Get a free key at console.groq.com (no credit card)
echo "LLM_PROVIDER=groq" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env
```

---

### Option C — Local development (no Docker)

**Prerequisites:** Python 3.11, Redis (Docker or local), Git

```bash
cd VIT_Vellore-Mrinal_Mohit-hackathon/finrisk

# 1. Create virtual environment and install all dependencies
make install
# or manually:
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# 3. Start Redis (requires Docker)
make redis-up
# or: docker run -d -p 6379:6379 redis:7.2-alpine

# 4. Start the FastAPI server (Terminal 1)
make dev
# → API running at http://localhost:8000

# 5. Start the Streamlit dashboard (Terminal 2)
make dashboard
# → Dashboard running at http://localhost:8501
```

---

### Option D — Train the XGBoost Impact Model on Google Colab (optional)

The system runs with a rule-based impact estimator by default. To train the market-calibrated XGBoost model:

**Step 1 — Open the Colab notebook**
```
notebooks/train_impact_model_colab.ipynb
```
Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook → select the file above.

**Step 2 — Run all 7 cells in order**
| Cell | What it does | Time |
|------|-------------|------|
| 1 | Install deps | ~1 min |
| 2 | Configuration | instant |
| 3 | Generate synthetic prices (or download from yfinance if unblocked) | ~2 min |
| 4 | Fetch GDELT news data | ~20 min |
| 5 | Compute abnormal returns + impact labels | ~8 min |
| 6 | Train XGBoost, evaluate vs baseline | ~3 min |
| 7 | **Download 3 model files** | instant |

**Step 3 — Place model files**
After Cell 7 downloads, copy the 3 files to your local machine:
```bash
# Place these files in:
finrisk/data/models/impact_xgb.pkl
finrisk/data/models/impact_xgb_features.json
finrisk/data/models/impact_percentiles.json
```

**Step 4 — Activate XGBoost**
```bash
# Add to finrisk/.env:
MODEL_IMPACT=xgboost

# Restart the API — XGBoost loads automatically
make dev   # or docker compose restart api
```

You should see in the logs:
```
ImpactCalculator: using XGBoost model from data/models/impact_xgb.pkl
XGBoostImpactModel loaded: 33 features, best_iter=151
```

**Evaluation results (v5, price momentum features):**

| Model | MAE | Spearman ρ | Notes |
|-------|-----|-----------|-------|
| Rule-based baseline | 3.515 | baseline | Event type × \|sentiment\| |
| XGBoost v4 (text only) | 1.754 | +0.128 | 7,219 training samples |
| XGBoost v5 (+ price features) | 1.644 | **+0.318** | + realized vol + momentum |

---

### Test the inject endpoint (demo moment)

```bash
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"text": "NVIDIA faces major export restrictions on AI chips to China", "ticker": "NVDA"}'
```

This runs the full pipeline — NLP → DuckDB → Redis → stress test (if impact ≥ 7) → rebalancer → LLM explanation — and returns everything in one response.

---

## 5. Key Results & Domain Impact

### NLP Engine Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Sentiment (FinBERT, Financial PhraseBank) | Accuracy | Run `python scripts/evaluate.py` |
| Event classifier (rule-based, 10 types) | Macro F1 | Run `python scripts/evaluate.py` |
| XGBoost v5 Impact Model | Spearman ρ | **+0.318** vs baseline −0.067 |
| XGBoost v5 Impact Model | MAE | **1.644** vs baseline 3.515 (−54% improvement) |

### Module A — Tactical Index Rebalancer

- 20-stock FinRisk index rebalanced dynamically based on cluster-aggregated FinBERT sentiment
- Volatility-adjusted signal scoring: `sentiment × confidence × novelty / realized_vol`
- Exponential decay by event type (Regulatory: 72h half-life; Earnings: 6h half-life)
- Turnover capped at 20% per rebalance; weights bounded [2%, 15%]

### Module B — Wholesale Portfolio Stress Tester

- **Portfolio**: 300 synthetic positions — 120 bonds, 80 loans, 60 equities, 25 FX forwards, 15 IR swaps (~$60M NAV)
- **Deterministic**: First-order factor shocks (duration-convexity bonds, ΔEL loans, beta equities, DV01 swaps)
- **Monte Carlo**: 10,000 paths, Student-t(df=5) fat tails, Cholesky-correlated factors, propagates NLP uncertainty via `impact_range`
- **Example**: Geopolitical event (impact 8.5, sentiment −0.80) → ΔNAV −$1.04M (−1.75%), VaR95 $1.93M, ES95 $2.57M
- **Reverse stress test**: binary search for minimum impact that breaches a configurable loss limit

### Domain Impact

This platform addresses a real gap in wholesale banking operations: the lack of an automated, real-time bridge between unstructured market intelligence and quantitative risk metrics. A credit officer using FinRisk would see, within seconds of a major regulatory announcement, a quantified portfolio stress estimate and an automatically rebalanced hedging index — tasks that currently take hours of manual work.

---

## 6. Running Tests & Evaluation

```bash
# Unit tests (no Redis needed)
make test

# Integration tests (Redis required)
make test-integration

# Full NLP evaluation (FinBERT, event classifier, impact model)
cd finrisk && .venv/bin/python scripts/evaluate.py

# Module A backtest (IC, Sharpe, drawdown vs equal-weight)
cd finrisk && .venv/bin/python scripts/backtest_module_a.py

# Module B historical scenario validation (2008, 2020, 2022, 2023)
cd finrisk && .venv/bin/python scripts/validate_module_b.py
```

Results are saved to `finrisk/data/results/` as JSON.

---

## 7. Limitations & Next Steps

**Limitations:**
- yfinance has intermittent rate limits; `PriceCache` falls back to synthetic prices when unavailable
- XGBoost impact model trained on synthetic prices (Yahoo Finance blocked during training period); real price data expected to further improve Spearman ρ from +0.318 toward +0.40+
- DeBERTa event classifier notebook exists (`notebooks/train_deberta_kaggle.ipynb`) but training was not completed due to time constraints; rule-based classifier (Macro F1 ~0.71) is used in production
- Monte Carlo loss distribution chart is reconstructed from VaR/ES summary statistics, not raw simulation paths (raw paths not persisted due to memory constraints)
- The synthetic portfolio uses fictional notional values; stress results are not real risk estimates

**Next Steps:**
- Retrain XGBoost v5 with real yfinance price data when API unblocks
- Fine-tune DeBERTa on Kaggle free GPU (~90 min) to replace rule-based classifier
- Add Kalshi prediction market macro overlay for FOMC/CPI events
- Extend Module B with full yield curve modelling and proper mark-to-market
- Production: replace DuckDB with TimescaleDB, Redis with Kafka for multi-tenant scale

---

## Project Structure

```
VIT_Vellore-Mrinal_Mohit-hackathon/
├── README.md                    ← This file
├── LICENSE                      ← MIT License
├── docs/
│   ├── presentation.pdf         ← 7-slide deck (PLACEHOLDER)
│   └── architecture.png         ← High-resolution architecture diagram
└── finrisk/                     ← All source code
    ├── app/                     # FastAPI application + NLP engine
    │   ├── engine/              # FinBERT, XGBoost, diffusion, novelty
    │   ├── ingestion/           # GDELT, EDGAR, StockTwits, Kaggle adapters
    │   ├── llm/                 # Groq/Ollama explanation layer
    │   ├── schema/              # Frozen RiskSignal Pydantic v2 contract
    │   └── streams/             # Redis Streams producer/consumer
    ├── module_a/                # Tactical index rebalancer
    ├── module_b/                # Portfolio stress tester + Monte Carlo
    ├── dashboard/               # Streamlit 4-tab dashboard
    ├── configs/                 # YAML: scenarios, tickers, portfolio
    ├── notebooks/               # Colab training notebooks
    ├── scripts/                 # Training, evaluation, data generation
    ├── tests/                   # 330+ pytest unit + integration tests
    ├── data/
    │   ├── sample/              # Offline replay fixtures (.jsonl)
    │   └── models/              # Trained model artifacts (.pkl, .json)
    ├── docker/                  # Per-service Dockerfiles
    ├── docker-compose.yml       # 5-service composition
    ├── Makefile                 # make install/dev/test/dashboard
    ├── requirements.txt         # All Python dependencies (pinned)
    └── .env.example             # All environment variables documented
```
