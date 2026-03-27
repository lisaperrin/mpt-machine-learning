#!/usr/bin/env python3

import argparse
import logging
import sys

import pandas as pd

from portfolio_optimization.backtesting.engine import BacktestEngine
from portfolio_optimization.config import Config
from portfolio_optimization.data.collector import DataCollector
from portfolio_optimization.models.factor_strategy import create_factor_strategy
from portfolio_optimization.models.mpt import ModernPortfolioTheory
from portfolio_optimization.models.optimal_ml_optimizer import OptimalMLPortfolioOptimizer
from portfolio_optimization.risk.metrics import RiskAnalyzer


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def run_analysis(config_path: str = "config.yaml"):
    config = Config(config_path)
    logger = logging.getLogger(__name__)
    logger.info("Starting portfolio optimization analysis")

    collector = DataCollector()
    tickers = config.all_tickers

    logger.info(f"Fetching data for {len(tickers)} assets")
    price_data = collector.fetch_stock_data(tickers, period=config.data_period)

    if price_data is None:
        logger.error("Failed to fetch price data")
        return

    returns_data = collector.calculate_returns()
    logger.info(f"Data loaded: {len(price_data)} days, {len(returns_data.columns)} assets")

    # MPT strategies
    logger.info("Running Modern Portfolio Theory optimization")
    mpt = ModernPortfolioTheory(returns_data, config.risk_free_rate)
    constraints = config.portfolio_constraints
    max_sharpe = mpt.max_sharpe_portfolio(
        min_weight=constraints.get('min_weight', 0.0),
        max_weight=constraints.get('max_weight', 1.0)
    )
    min_vol = mpt.min_volatility_portfolio(
        min_weight=constraints.get('min_weight', 0.0),
        max_weight=constraints.get('max_weight', 1.0)
    )

    # Factor-based strategy
    logger.info("Running factor-based strategy")
    factor_weights = create_factor_strategy(returns_data, price_data)

    # Optimal ML strategy
    logger.info("Running Optimal ML optimization (HRP + Black-Litterman)")
    try:
        optimal_ml = OptimalMLPortfolioOptimizer(
            returns_data, price_data, config.__dict__['_config']
        )
        ml_weights = optimal_ml.get_optimal_portfolio_weights(constraints=constraints)
    except Exception as e:
        logger.warning(f"Optimal ML failed, using factor strategy: {e}")
        ml_weights = factor_weights

    strategies = {
        'Equal Weight': {asset: 1/len(returns_data.columns) for asset in returns_data.columns},
        'Max Sharpe': max_sharpe['weights'],
        'Min Volatility': min_vol['weights'],
        'Factor-Based': factor_weights,
        'Optimal ML': ml_weights
    }

    logger.info("Portfolio Performance Analysis")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<16} {'Return':<10} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<10}")
    logger.info("-" * 80)

    for name, weights in strategies.items():
        weights_clean = {k: v for k, v in weights.items() if k in returns_data.columns}
        weights_series = pd.Series(weights_clean).reindex(returns_data.columns, fill_value=0)
        weights_series = weights_series / weights_series.sum()

        portfolio_returns = (returns_data * weights_series).sum(axis=1)
        risk_analyzer = RiskAnalyzer(portfolio_returns)
        risk_metrics = risk_analyzer.comprehensive_risk_report()

        logger.info(f"{name:<16} {risk_metrics['Annualized_Return']:>8.1%} "
                    f"{risk_metrics['Annualized_Volatility']:>10.1%} "
                    f"{risk_metrics['Sharpe_Ratio']:>10.3f} "
                    f"{risk_metrics['Max_Drawdown']:>8.1%}")

    logger.info("\nTop Optimal ML Positions:")
    ml_sorted = sorted(ml_weights.items(), key=lambda x: x[1], reverse=True)
    for asset, weight in ml_sorted[:10]:
        if weight > 0.01:
            logger.info(f"  {asset}: {weight:.1%}")

    data_quality = collector.get_data_quality_report()
    logger.info(f"\nData Quality: {data_quality['total_assets']} assets, "
                f"{data_quality['failed_assets']} failed")


def run_backtest(config_path: str = "config.yaml"):
    config = Config(config_path)
    logger = logging.getLogger(__name__)

    collector = DataCollector()
    price_data = collector.fetch_stock_data(config.all_tickers, period="5y")
    returns_data = collector.calculate_returns()

    def optimal_ml_strategy(historical_returns, historical_prices):
        try:
            optimizer = OptimalMLPortfolioOptimizer(
                historical_returns, historical_prices, config.__dict__['_config']
            )
            return optimizer.get_optimal_portfolio_weights()
        except Exception:
            return create_factor_strategy(historical_returns, historical_prices)

    def max_sharpe_strategy(historical_returns, historical_prices):
        mpt = ModernPortfolioTheory(historical_returns)
        result = mpt.max_sharpe_portfolio()
        return result['weights']

    def equal_weight_strategy(historical_returns, historical_prices):
        n = len(historical_returns.columns)
        return {col: 1.0 / n for col in historical_returns.columns}

    backtest_params = config.backtesting_params
    strategies = {
        'Equal Weight': equal_weight_strategy,
        'Max Sharpe': max_sharpe_strategy,
        'Optimal ML': optimal_ml_strategy,
    }

    logger.info("Running backtests...")

    for strategy_name, strategy_func in strategies.items():
        logger.info(f"\nBacktesting {strategy_name}")

        engine = BacktestEngine(
            price_data,
            returns_data,
            transaction_cost=backtest_params.get('transaction_cost', 0.001),
            rebalance_frequency=backtest_params.get('rebalance_frequency', 21)
        )

        results = engine.run_backtest(
            strategy_func,
            min_history=backtest_params.get('min_history', 252)
        )

        logger.info(f"  Total Return: {results['total_return']:.1%}")
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.1%}")
        logger.info(f"  Avg Turnover: {results['avg_turnover']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Optimization Platform")
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['optimize', 'backtest'], default='optimize')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.mode == 'optimize':
        run_analysis(args.config)
    elif args.mode == 'backtest':
        run_backtest(args.config)


if __name__ == "__main__":
    main()
