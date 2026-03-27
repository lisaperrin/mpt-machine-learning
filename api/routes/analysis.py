import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.cache import config, get_portfolio_data, get_cached_result, set_cached_result
from portfolio_optimization.models.mpt import ModernPortfolioTheory
from portfolio_optimization.risk.metrics import RiskAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/risk-analysis")
async def risk_analysis(
    assets: str = Query(default="AAPL,MSFT,GOOGL,JPM,JNJ,TLT,GLD,VNQ"),
    strategy: str = Query(default="equal")
):
    """Comprehensive risk analysis for a portfolio."""
    try:
        asset_list = sorted([a.strip() for a in assets.split(",")])
        cache_key = f"risk_{','.join(asset_list)}_{strategy}"

        cached = get_cached_result(cache_key)
        if cached:
            return cached

        data = await get_portfolio_data(asset_list)
        returns = data['returns']

        available_assets = [a for a in asset_list if a in returns.columns]
        filtered_returns = returns[available_assets]

        if strategy == "max_sharpe":
            mpt = ModernPortfolioTheory(filtered_returns, config.risk_free_rate)
            portfolio = mpt.max_sharpe_portfolio()
            weights = pd.Series(portfolio['weights'])
        elif strategy == "min_vol":
            mpt = ModernPortfolioTheory(filtered_returns, config.risk_free_rate)
            portfolio = mpt.min_volatility_portfolio()
            weights = pd.Series(portfolio['weights'])
        else:
            weights = pd.Series({a: 1.0/len(available_assets) for a in available_assets})

        portfolio_returns = (filtered_returns * weights).sum(axis=1)
        risk_analyzer = RiskAnalyzer(portfolio_returns, config.risk_free_rate)
        risk_report = risk_analyzer.comprehensive_risk_report()

        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_vol = portfolio_returns.rolling(21).std() * np.sqrt(252)
        rolling_sharpe = (
            (portfolio_returns.rolling(63).mean() * 252 - config.risk_free_rate)
            / (portfolio_returns.rolling(63).std() * np.sqrt(252))
        )
        drawdown = (cumulative_returns - cumulative_returns.expanding().max()) / cumulative_returns.expanding().max()

        result = {
            "success": True,
            "risk_metrics": risk_report,
            "weights": weights.to_dict(),
            "time_series": {
                "cumulative_returns": {str(k): float(v) for k, v in cumulative_returns.tail(252).items()},
                "rolling_volatility": {str(k): float(v) for k, v in rolling_vol.tail(252).dropna().items()},
                "rolling_sharpe": {str(k): float(v) for k, v in rolling_sharpe.tail(252).dropna().items()},
                "drawdown": {str(k): float(v) for k, v in drawdown.tail(252).items()}
            },
            "strategy": strategy
        }
        set_cached_result(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlations")
async def get_correlation_matrix(
    assets: str = Query(default="AAPL,MSFT,GOOGL,JPM,JNJ,TLT,GLD,VNQ")
):
    """Get correlation matrix for selected assets."""
    try:
        asset_list = sorted([a.strip() for a in assets.split(",")])
        cache_key = f"corr_{','.join(asset_list)}"

        cached = get_cached_result(cache_key)
        if cached:
            return cached

        data = await get_portfolio_data(asset_list)
        returns = data['returns']

        available_assets = [a for a in asset_list if a in returns.columns]
        filtered_returns = returns[available_assets]

        corr_matrix = filtered_returns.corr()

        result = {
            "success": True,
            "assets": available_assets,
            "correlation_matrix": corr_matrix.to_dict(),
            "average_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
            "min_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
            "max_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max())
        }
        set_cached_result(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Correlation matrix failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/factor-exposures")
async def get_factor_exposures(
    assets: str = Query(default="AAPL,MSFT,GOOGL,JPM,JNJ,TLT,GLD,VNQ")
):
    """Calculate factor exposures for portfolio."""
    try:
        asset_list = [a.strip() for a in assets.split(",")]
        data = await get_portfolio_data(asset_list)
        returns = data['returns']

        available_assets = [a for a in asset_list if a in returns.columns]
        filtered_returns = returns[available_assets]

        market_return = filtered_returns.mean(axis=1)

        factor_exposures = {}
        for asset in available_assets:
            asset_returns = filtered_returns[asset]

            cov_with_market = asset_returns.cov(market_return)
            market_var = market_return.var()
            beta = cov_with_market / market_var if market_var > 0 else 1.0

            momentum_12m = (
                asset_returns.rolling(252).mean().iloc[-1]
                if len(asset_returns) >= 252
                else asset_returns.mean()
            )
            volatility = asset_returns.std() * np.sqrt(252)
            sharpe = (asset_returns.mean() * 252 - config.risk_free_rate) / volatility if volatility > 0 else 0

            factor_exposures[asset] = {
                'beta': float(beta),
                'momentum_12m': float(momentum_12m * 252),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe)
            }

        avg_beta = np.mean([v['beta'] for v in factor_exposures.values()])
        avg_momentum = np.mean([v['momentum_12m'] for v in factor_exposures.values()])
        avg_volatility = np.mean([v['volatility'] for v in factor_exposures.values()])

        return {
            "success": True,
            "asset_exposures": factor_exposures,
            "portfolio_exposures": {
                "beta": float(avg_beta),
                "momentum": float(avg_momentum),
                "volatility": float(avg_volatility)
            },
            "assets": available_assets
        }
    except Exception as e:
        logger.error(f"Factor exposures failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
