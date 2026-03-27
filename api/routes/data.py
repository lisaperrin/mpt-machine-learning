import logging

import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException

from api.cache import config, get_portfolio_data
from portfolio_optimization.models.factor_strategy import create_factor_strategy
from portfolio_optimization.models.mpt import ModernPortfolioTheory
from portfolio_optimization.risk.metrics import RiskAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/assets")
async def get_available_assets():
    """Get list of available assets by sector."""
    return {
        "asset_universe": config.asset_universe,
        "total_assets": len(config.all_tickers)
    }


@router.get("/data-quality")
async def data_quality():
    """Get data quality metrics."""
    try:
        data = await get_portfolio_data()
        quality_report = data['collector'].get_data_quality_report()
        return quality_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize")
async def create_visualization():
    """Create portfolio visualization charts."""
    try:
        logger.info("Creating portfolio visualization...")

        default_assets = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "TLT", "GLD", "VNQ"]
        data = await get_portfolio_data(default_assets)
        returns = data['returns']

        strategies = {}

        equal_weight = 1.0 / len(default_assets)
        strategies['Equal Weight'] = {asset: equal_weight for asset in default_assets}

        mpt = ModernPortfolioTheory(returns, config.risk_free_rate)
        max_sharpe = mpt.max_sharpe_portfolio(min_weight=0.05, max_weight=0.30)
        strategies['Max Sharpe (MPT)'] = max_sharpe['weights']

        ml_weights = create_factor_strategy(returns, data['prices'])
        strategies['Factor-Based'] = ml_weights

        chart_data = []
        for name, weights in strategies.items():
            weights_series = pd.Series(weights).reindex(returns.columns, fill_value=0)
            weights_series = weights_series / weights_series.sum()

            portfolio_returns = (returns * weights_series).sum(axis=1)
            risk_analyzer = RiskAnalyzer(portfolio_returns, config.risk_free_rate)
            metrics = risk_analyzer.comprehensive_risk_report()

            chart_data.append({
                'name': name,
                'return': metrics['Annualized_Return'],
                'volatility': metrics['Annualized_Volatility'],
                'sharpe': metrics['Sharpe_Ratio'],
            })

        fig = go.Figure()
        for strategy in chart_data:
            fig.add_trace(go.Scatter(
                x=[strategy['volatility']],
                y=[strategy['return']],
                mode='markers',
                name=strategy['name'],
                marker=dict(size=15, opacity=0.7),
                text=f"Sharpe: {strategy['sharpe']:.2f}",
                hovertemplate=(
                    '<b>%{fullData.name}</b><br>'
                    'Return: %{y:.1%}<br>'
                    'Volatility: %{x:.1%}<br>'
                    f"Sharpe: {strategy['sharpe']:.2f}<extra></extra>"
                )
            ))

        fig.update_layout(
            title='Portfolio Risk-Return Analysis',
            xaxis_title='Volatility (Annual)',
            yaxis_title='Expected Return (Annual)',
            hovermode='closest',
            template='plotly_white',
            width=800,
            height=500
        )

        graph_json = fig.to_json()

        logger.info("Portfolio visualization created successfully")
        return {
            "success": True,
            "graph_json": graph_json,
            "strategies": len(strategies)
        }

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
