import logging

from fastapi import APIRouter, HTTPException, Query

from api.cache import config, get_portfolio_data
from portfolio_optimization.evaluation.evaluator import StrategyEvaluator
from portfolio_optimization.models.factor_strategy import create_factor_strategy
from portfolio_optimization.models.mpt import ModernPortfolioTheory
from portfolio_optimization.models.optimal_ml_optimizer import OptimalMLPortfolioOptimizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/strategy-evaluation")
async def strategy_evaluation(
    assets: str = Query(default="AAPL,MSFT,GOOGL,AMZN,JPM,V,JNJ,UNH,TLT,GLD"),
    train_window: int = Query(default=252, ge=63, le=504),
    test_window: int = Query(default=63, ge=21, le=126),
    n_bootstrap: int = Query(default=500, ge=100, le=2000)
):
    """Out-of-sample strategy evaluation with bootstrap confidence intervals."""
    try:
        asset_list = [a.strip() for a in assets.split(",")]
        data = await get_portfolio_data(asset_list)

        available_assets = [a for a in asset_list if a in data['returns'].columns]
        if len(available_assets) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 valid assets")

        filtered_returns = data['returns'][available_assets]
        filtered_prices = data['prices'][available_assets]

        min_required = train_window + test_window
        if len(filtered_returns) < min_required:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {min_required} days of data, have {len(filtered_returns)}"
            )

        def equal_weight_strategy(hist_returns, hist_prices):
            return {col: 1.0 / len(hist_returns.columns) for col in hist_returns.columns}

        def max_sharpe_strategy(hist_returns, hist_prices):
            mpt = ModernPortfolioTheory(hist_returns, config.risk_free_rate)
            result = mpt.max_sharpe_portfolio()
            return result['weights']

        def factor_strategy(hist_returns, hist_prices):
            return create_factor_strategy(hist_returns, hist_prices)

        def optimal_ml_strategy(hist_returns, hist_prices):
            try:
                optimizer = OptimalMLPortfolioOptimizer(
                    hist_returns, hist_prices,
                    {'portfolio_constraints': {'min_weight': 0.01, 'max_weight': 0.25}}
                )
                return optimizer.get_optimal_portfolio_weights()
            except Exception:
                return factor_strategy(hist_returns, hist_prices)

        strategies = {
            'Equal Weight': equal_weight_strategy,
            'Max Sharpe (MPT)': max_sharpe_strategy,
            'Factor-Based': factor_strategy,
            'Optimal ML (HRP+BL)': optimal_ml_strategy,
        }

        evaluator = StrategyEvaluator(filtered_returns, filtered_prices, strategies)
        results = evaluator.comprehensive_comparison(
            train_window=train_window,
            test_window=test_window,
            n_bootstrap=n_bootstrap
        )

        return {
            "success": True,
            "evaluation": results,
            "assets": available_assets
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
