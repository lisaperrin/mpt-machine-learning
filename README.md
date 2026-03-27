# PortfolioML

Portfolio optimization platform. Compares equal-weight, Markowitz (max Sharpe / min vol), cross-sectional factor scoring, and an ML strategy that blends Hierarchical Risk Parity with Black-Litterman views. FastAPI backend, SvelteKit frontend, walk-forward backtesting.

## Setup

```bash
uv sync                      # install deps
uv run python main.py        # start API on :8000
cd frontend && npm i && npm run dev  # start frontend on :3000
```

API docs at http://localhost:8000/docs.

## CLI

```bash
uv run python run_optimization.py --mode optimize   # run all strategies, print comparison
uv run python run_optimization.py --mode backtest    # walk-forward backtest
```

## How the ML strategy works

Standard ML portfolio optimization tries to predict individual asset returns and tends to overfit on financial data. This project takes a different approach:

- **HRP base allocation** -- Hierarchical Risk Parity clusters assets by correlation and allocates without predicting returns. Based on [Lopez de Prado (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).
- **Black-Litterman with ML views** -- Ridge regression trained on momentum, quality, and relative strength factors generates per-asset return views with confidence scores. These feed into Black-Litterman, which blends them with a momentum-tilted equilibrium prior. Low-confidence views barely move the posterior; high-confidence views shift it toward the ML prediction.
- **Auto-tuned blend** -- Walk-forward cross-validation tests HRP/BL blend ratios from 30/70 to 70/30 and picks the one with the best out-of-sample Sharpe.
- **Regime detection** -- Short/long-term volatility ratio shifts positioning defensively in high-vol environments.

## API

POST `/api/optimize` with `{ "assets": [...], "constraints": { "min_weight": 0.05, "max_weight": 0.30 } }` returns weights and metrics for all 5 strategies.

Other endpoints: `/api/efficient-frontier`, `/api/correlations`, `/api/backtest-detailed`, `/api/risk-analysis`, `/api/monte-carlo`, `/api/factor-exposures`, `/api/strategy-evaluation`, `/api/assets`.

## Configuration

Asset universe, constraints, model params, and backtest settings live in `config.yaml`.

## Development

```bash
make test        # pytest
make lint        # ruff
make format      # ruff format
```


Not financial advice.
