# PortfolioML

Portfolio optimization platform. Compares equal-weight, Markowitz (max Sharpe / min vol), cross-sectional factor scoring, and an ML strategy that anchors on Hierarchical Risk Parity with bounded Black-Litterman tilts. FastAPI backend, SvelteKit frontend, walk-forward backtesting.

## Setup

```bash
uv sync --extra dev          # install Python deps
uv run python main.py        # start API on :8000
cd frontend
npm ci                       # install frontend deps from lockfile
npm run dev                  # start frontend on :3000
```

On Windows PowerShell, use `npm.cmd` if script execution policy blocks `npm.ps1`.

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
- **Bounded ML tilt** -- The final portfolio starts from HRP and only moves partway toward the BL portfolio. The tilt budget is capped in `config.yaml` and scaled by current signal dispersion and volatility regime, not by choosing the historically best Sharpe blend.
- **Regime detection** -- Short/long-term volatility ratio shifts positioning defensively in high-vol environments.

## API

POST `/api/optimize` with `{ "assets": [...], "constraints": { "min_weight": 0.05, "max_weight": 0.30 } }` returns weights and metrics for all 5 strategies.

Other endpoints: `/api/efficient-frontier`, `/api/correlations`, `/api/backtest-detailed`, `/api/risk-analysis`, `/api/monte-carlo`, `/api/factor-exposures`, `/api/strategy-evaluation`, `/api/assets`.

Optimization responses include diagnostics for each strategy, data-quality metadata, missing tickers, and warnings when a fallback path was used.

## Configuration

Asset universe, constraints, ML tilt budget, model params, and backtest settings live in `config.yaml`.

## Development

```bash
make test        # pytest
make lint        # ruff
make format      # ruff format
cd frontend && npm run check
```

If `uv` is not installed, install it first from https://docs.astral.sh/uv/ or use an equivalent Python 3.11+ virtual environment with the dependencies from `pyproject.toml`.

## Methodology and safeguards

- Portfolio constraints are validated before optimization. Infeasible bounds such as `3 assets * 40% min_weight` are rejected instead of silently relaxed.
- Weight projection preserves both `sum(weights) == 1` and per-asset min/max bounds.
- MPT uses Ledoit-Wolf covariance shrinkage when enough observations are available.
- Efficient-frontier target returns are derived from the achievable return range, not from volatility.
- ML views use time-series cross-validation to avoid ordinary K-fold leakage on chronological data.
- The live ML allocation does not auto-select an HRP/BL blend from historical performance; it applies a capped tilt away from HRP.
- API tests use deterministic synthetic market data rather than live `yfinance` downloads.

## Limitations

- Market data comes from `yfinance`, so production use would need stronger data validation, vendor SLAs, survivorship-bias handling, and corporate-action checks.
- Backtests are educational and do not model taxes, bid/ask spread, market impact, borrow costs, or execution latency.
- ML signals are deliberately regularized and constrained to bounded HRP tilts because short-horizon return prediction is noisy.
- The app is a portfolio and research demo, not an investment recommendation system.


Not financial advice.
