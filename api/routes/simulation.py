import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from scipy.optimize import minimize as scipy_minimize

from api.cache import config, get_portfolio_data, get_cached_result, set_cached_result
from portfolio_optimization.models.mpt import ModernPortfolioTheory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/efficient-frontier")
async def get_efficient_frontier(
    assets: str = Query(default="AAPL,MSFT,GOOGL,JPM,JNJ,TLT,GLD,VNQ"),
    num_portfolios: int = Query(default=30, ge=10, le=100)
):
    """Generate efficient frontier data for visualization."""
    try:
        asset_list = sorted([a.strip() for a in assets.split(",")])
        cache_key = f"frontier_{','.join(asset_list)}_{num_portfolios}"

        cached = get_cached_result(cache_key)
        if cached:
            return cached

        data = await get_portfolio_data(asset_list)
        returns = data['returns']

        available_assets = [a for a in asset_list if a in returns.columns]
        if len(available_assets) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid assets")

        filtered_returns = returns[available_assets]
        mpt = ModernPortfolioTheory(filtered_returns, config.risk_free_rate)

        frontier_volatilities = []
        frontier_returns = []
        frontier_weights = []

        min_vol_portfolio = mpt.min_volatility_portfolio()
        max_sharpe_portfolio = mpt.max_sharpe_portfolio()

        min_ret = min_vol_portfolio['return']
        max_ret = max(mpt.mean_returns)

        target_returns = np.linspace(min_ret, max_ret * 0.95, num_portfolios)

        for target in target_returns:
            try:
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x, t=target: mpt.portfolio_performance(x)[0] - t}
                ]
                bounds = tuple((0, 1) for _ in range(mpt.num_assets))
                initial = np.array([1/mpt.num_assets] * mpt.num_assets)

                result = scipy_minimize(mpt.portfolio_volatility, initial,
                                        method='SLSQP', bounds=bounds, constraints=constraints)

                if result.success:
                    ret, vol, sharpe = mpt.portfolio_performance(result.x)
                    frontier_volatilities.append(vol)
                    frontier_returns.append(ret)
                    frontier_weights.append(dict(zip(available_assets, result.x.tolist())))
            except Exception:
                continue

        individual_assets = []
        for asset in available_assets:
            asset_return = mpt.mean_returns[asset]
            asset_vol = np.sqrt(mpt.cov_matrix.loc[asset, asset])
            individual_assets.append({
                'name': asset,
                'return': float(asset_return),
                'volatility': float(asset_vol)
            })

        result = {
            "success": True,
            "frontier": {
                "volatilities": frontier_volatilities,
                "returns": frontier_returns,
                "weights": frontier_weights
            },
            "special_portfolios": {
                "min_volatility": {
                    "return": min_vol_portfolio['return'],
                    "volatility": min_vol_portfolio['volatility'],
                    "sharpe": min_vol_portfolio['sharpe_ratio'],
                    "weights": min_vol_portfolio['weights']
                },
                "max_sharpe": {
                    "return": max_sharpe_portfolio['return'],
                    "volatility": max_sharpe_portfolio['volatility'],
                    "sharpe": max_sharpe_portfolio['sharpe_ratio'],
                    "weights": max_sharpe_portfolio['weights']
                }
            },
            "individual_assets": individual_assets,
            "risk_free_rate": config.risk_free_rate
        }
        set_cached_result(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Efficient frontier failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monte-carlo")
async def monte_carlo_simulation(
    assets: str = Query(default="AAPL,MSFT,GOOGL,JPM,JNJ,TLT,GLD,VNQ"),
    num_simulations: int = Query(default=1000, ge=100, le=10000),
    horizon_days: int = Query(default=252, ge=21, le=756)
):
    """Monte Carlo simulation for portfolio projections."""
    try:
        asset_list = [a.strip() for a in assets.split(",")]
        data = await get_portfolio_data(asset_list)
        returns = data['returns']

        available_assets = [a for a in asset_list if a in returns.columns]
        filtered_returns = returns[available_assets]

        weights = np.array([1.0/len(available_assets)] * len(available_assets))
        portfolio_returns = (filtered_returns * weights).sum(axis=1)

        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()

        rng = np.random.default_rng(42)
        simulations = np.zeros((num_simulations, horizon_days))

        for i in range(num_simulations):
            daily_returns = rng.normal(mu, sigma, horizon_days)
            simulations[i] = np.cumprod(1 + daily_returns)

        percentiles = {
            'p5': np.percentile(simulations[:, -1], 5),
            'p25': np.percentile(simulations[:, -1], 25),
            'p50': np.percentile(simulations[:, -1], 50),
            'p75': np.percentile(simulations[:, -1], 75),
            'p95': np.percentile(simulations[:, -1], 95)
        }

        paths = {
            'p5': np.percentile(simulations, 5, axis=0).tolist(),
            'p25': np.percentile(simulations, 25, axis=0).tolist(),
            'p50': np.percentile(simulations, 50, axis=0).tolist(),
            'p75': np.percentile(simulations, 75, axis=0).tolist(),
            'p95': np.percentile(simulations, 95, axis=0).tolist()
        }

        prob_loss = float(np.mean(simulations[:, -1] < 1.0))
        prob_10_gain = float(np.mean(simulations[:, -1] > 1.10))
        prob_20_gain = float(np.mean(simulations[:, -1] > 1.20))

        return {
            "success": True,
            "parameters": {
                "daily_mean": float(mu),
                "daily_std": float(sigma),
                "annualized_return": float(mu * 252),
                "annualized_volatility": float(sigma * np.sqrt(252)),
                "num_simulations": num_simulations,
                "horizon_days": horizon_days
            },
            "final_values": percentiles,
            "paths": paths,
            "probabilities": {
                "loss": prob_loss,
                "gain_10pct": prob_10_gain,
                "gain_20pct": prob_20_gain
            },
            "assets": available_assets
        }
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
