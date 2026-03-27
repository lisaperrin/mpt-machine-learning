import logging

from fastapi import APIRouter, HTTPException, Query

from api.cache import config, get_portfolio_data
from portfolio_optimization.backtesting.engine import BacktestEngine
from portfolio_optimization.models.factor_strategy import create_factor_strategy
from portfolio_optimization.models.mpt import ModernPortfolioTheory
from portfolio_optimization.models.optimal_ml_optimizer import OptimalMLPortfolioOptimizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/backtest")
async def backtest_strategies():
    """Backtest all strategies."""
    try:
        logger.info("Starting backtest of all strategies")
        data = await get_portfolio_data()

        test_assets = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN',
            'JPM', 'V',
            'JNJ', 'UNH',
            'TLT', 'GLD'
        ]
        available_test_assets = [asset for asset in test_assets if asset in data['returns'].columns]

        def equal_weight_strategy(historical_returns, historical_prices):
            equal_weight = 1.0 / len(historical_returns.columns)
            return {col: equal_weight for col in historical_returns.columns}

        def max_sharpe_strategy(historical_returns, historical_prices):
            mpt = ModernPortfolioTheory(historical_returns)
            result = mpt.max_sharpe_portfolio()
            return result['weights']

        def factor_strategy(historical_returns, historical_prices):
            return create_factor_strategy(historical_returns, historical_prices)

        strategies = {
            'Equal Weight': equal_weight_strategy,
            'Max Sharpe (MPT)': max_sharpe_strategy,
            'Factor-Based': factor_strategy
        }

        backtest_results = {}
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"Backtesting {strategy_name}")
            engine = BacktestEngine(
                data['prices'][available_test_assets],
                data['returns'][available_test_assets],
                transaction_cost=0.001,
                rebalance_frequency=21
            )
            results = engine.run_backtest(strategy_func, min_history=252)
            backtest_results[strategy_name] = results

        logger.info("Backtest completed")
        return backtest_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest-detailed")
async def backtest_detailed(
    assets: str = Query(default="AAPL,MSFT,GOOGL,AMZN,JPM,V,JNJ,UNH,TLT,GLD"),
    rebalance_days: int = Query(default=21, ge=5, le=63)
):
    """Detailed backtest with equity curves and performance metrics."""
    try:
        asset_list = [a.strip() for a in assets.split(",")]
        data = await get_portfolio_data(asset_list)

        available_assets = [a for a in asset_list if a in data['returns'].columns]
        if len(available_assets) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 valid assets")

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
            'Optimal ML': optimal_ml_strategy
        }

        backtest_results = {}
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"Backtesting {strategy_name}")

            engine = BacktestEngine(
                data['prices'][available_assets],
                data['returns'][available_assets],
                transaction_cost=0.001,
                rebalance_frequency=rebalance_days
            )

            results = engine.run_backtest(strategy_func, min_history=252)

            equity_curve = {}
            if hasattr(engine, 'portfolio_value') and len(engine.portfolio_value) > 0:
                equity_curve = {
                    str(date): float(value)
                    for date, value in engine.portfolio_value.items()
                }

            weights_history = {}
            if hasattr(engine, 'weights_history') and len(engine.weights_history) > 0:
                final_weights = engine.weights_history.iloc[-1].to_dict()
                weights_history = {k: float(v) for k, v in final_weights.items()}

            backtest_results[strategy_name] = {
                **results,
                'equity_curve': equity_curve,
                'final_weights': weights_history,
                'num_periods': len(equity_curve)
            }

        return {
            "success": True,
            "results": backtest_results,
            "assets": available_assets,
            "rebalance_frequency": rebalance_days,
            "transaction_cost": 0.001
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detailed backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
