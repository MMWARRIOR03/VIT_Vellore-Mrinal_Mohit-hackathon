.PHONY: help install dev test test-cov lint format up down redis-up redis-down clean

PYTHON   := python3.11
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
UVICORN  := $(VENV)/bin/uvicorn

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment ─────────────────────────────────────────────────────────────

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate  ## Create venv and install dependencies
	$(PIP) install -r requirements.txt
	@cp -n .env.example .env 2>/dev/null || true
	@mkdir -p data
	@echo "✓ Environment ready. Copy .env.example → .env and adjust values."

# ─── Development ─────────────────────────────────────────────────────────────

dev:  ## Run API server with hot reload
	@mkdir -p data
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

dev-gen:  ## Run the fake signal generator standalone
	$(VENV)/bin/python -m app.generators.fake_signal

# ─── Testing ─────────────────────────────────────────────────────────────────

test:  ## Run unit tests (excludes integration)
	$(PYTEST) tests/ -v -m "not integration" --tb=short

test-integration:  ## Run integration tests (requires Redis)
	$(PYTEST) tests/ -v -m "integration" --tb=short

test-all:  ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=app --cov-report=term-missing --cov-report=html:htmlcov

# ─── Code quality ────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	$(VENV)/bin/ruff check app/ tests/

format:  ## Auto-format with ruff
	$(VENV)/bin/ruff format app/ tests/

# ─── Docker / infrastructure ─────────────────────────────────────────────────

up:  ## Start all services (Redis + API + worker + dashboard) via docker-compose
	docker compose up -d --build

down:  ## Stop all services
	docker compose down

demo:  ## Run in offline demo/replay mode (no internet required)
	REPLAY_MODE=true docker compose up --build

live:  ## Run with live data ingestion (all 5 services)
	docker compose --profile live up --build

redis-up:  ## Start only Redis (for local dev without Docker API)
	docker compose up -d redis

redis-down:  ## Stop Redis
	docker compose stop redis

inject:  ## Test the /inject endpoint (requires running API)
	curl -s -X POST http://localhost:8000/inject \
	  -H "Content-Type: application/json" \
	  -d '{"text": "NVIDIA faces major export restrictions on AI chips", "ticker": "NVDA"}' \
	  | python3 -m json.tool

sample-data:  ## Pre-generate offline replay data (articles + signals)
	$(VENV)/bin/python scripts/generate_sample_data.py

sample-data-fast:  ## Pre-generate articles only (skip NLP model download)
	$(VENV)/bin/python scripts/generate_sample_data.py --no-nlp

# ─── Housekeeping ────────────────────────────────────────────────────────────

dashboard:  ## Run Streamlit dashboard on port 8501
	$(VENV)/bin/streamlit run dashboard/app.py --server.port 8501

clean:  ## Remove venv, cache, and test artifacts
	rm -rf $(VENV) __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
