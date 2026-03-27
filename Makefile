.PHONY: install dev test lint format docker clean

install:
	uv sync

install-dev:
	uv sync --extra dev

dev:
	uv run python main.py

cli:
	uv run python run_optimization.py --mode optimize

backtest:
	uv run python run_optimization.py --mode backtest

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=portfolio_optimization --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

docker:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache dist build *.egg-info
