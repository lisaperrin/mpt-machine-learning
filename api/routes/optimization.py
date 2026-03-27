import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.cache import config, get_portfolio_data
from api.schemas import OptimizationRequest, OptimizationResponse, PortfolioWeights
from portfolio_optimization.models.factor_strategy import create_factor_strategy
from portfolio_optimization.models.mpt import ModernPortfolioTheory
from portfolio_optimization.models.optimal_ml_optimizer import OptimalMLPortfolioOptimizer
from portfolio_optimization.risk.metrics import RiskAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """Portfolio optimization with 5 strategies."""
    try:
        logger.info(f"Starting optimization for {len(request.assets)} assets")
        data = await get_portfolio_data(request.assets)
        prices = data['prices']
        returns = data['returns']

        available_assets = [asset for asset in request.assets if asset in returns.columns]
        missing_assets = [asset for asset in request.assets if asset not in returns.columns]

        if missing_assets:
            logger.warning(f"Missing assets: {missing_assets}")

        if len(available_assets) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 valid assets")

        filtered_returns = returns[available_assets]
        filtered_prices = prices[available_assets]

        constraints = request.constraints or {}
        min_weight = constraints.get('min_weight', 0.01)
        max_weight = constraints.get('max_weight', 0.25)

        equal_weight = 1.0 / len(available_assets)
        equal_weights = {asset: equal_weight for asset in available_assets}

        mpt = ModernPortfolioTheory(filtered_returns, config.risk_free_rate)
        max_sharpe = mpt.max_sharpe_portfolio(min_weight=min_weight, max_weight=max_weight)
        min_vol = mpt.min_volatility_portfolio(min_weight=min_weight, max_weight=max_weight)

        factor_weights = create_factor_strategy(filtered_returns, filtered_prices)

        try:
            optimal_ml = OptimalMLPortfolioOptimizer(
                filtered_returns,
                filtered_prices,
                {'portfolio_constraints': constraints}
            )
            optimal_ml_weights = optimal_ml.get_optimal_portfolio_weights(
                constraints={'min_weight': min_weight, 'max_weight': max_weight}
            )
        except Exception as e:
            logger.warning(f"Optimal ML failed, falling back to factor strategy: {e}")
            optimal_ml_weights = factor_weights.copy()

        strategies = {
            'Equal Weight': equal_weights,
            'Max Sharpe (MPT)': max_sharpe['weights'],
            'Min Volatility (MPT)': min_vol['weights'],
            'Factor-Based': factor_weights,
            'Optimal ML (HRP+Black-Litterman)': optimal_ml_weights
        }

        results = {}
        for name, weights in strategies.items():
            weights_series = pd.Series(weights).reindex(filtered_returns.columns, fill_value=0)
            weights_series = weights_series / weights_series.sum()

            portfolio_returns = (filtered_returns * weights_series).sum(axis=1)
            risk_analyzer = RiskAnalyzer(portfolio_returns, config.risk_free_rate)
            risk_metrics = risk_analyzer.comprehensive_risk_report()

            top_holdings = sorted(weights_series.items(), key=lambda x: x[1], reverse=True)

            results[name] = PortfolioWeights(
                weights=weights_series.to_dict(),
                metrics=risk_metrics,
                top_holdings=top_holdings
            )

        logger.info("Optimization completed successfully")
        return OptimizationResponse(
            success=True,
            results=results,
            selected_assets=available_assets
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
