"""Tests for portfolio optimization models."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.models.factor_strategy import create_factor_strategy
from portfolio_optimization.models.mpt import ModernPortfolioTheory


@pytest.fixture
def sample_returns():
    """Generate realistic synthetic returns for testing."""
    np.random.seed(42)
    n_days = 504  # ~2 years
    assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TLT', 'GLD']

    # Correlated returns with different characteristics
    mean_returns = np.array([0.0005, 0.0004, 0.0006, 0.0003, 0.0001, 0.0002])
    volatilities = np.array([0.02, 0.018, 0.022, 0.016, 0.008, 0.012])

    returns = pd.DataFrame(
        np.random.randn(n_days, len(assets)) * volatilities + mean_returns,
        columns=assets,
        index=pd.bdate_range(end='2025-01-01', periods=n_days)
    )
    return returns


@pytest.fixture
def sample_prices(sample_returns):
    """Generate prices from returns."""
    return (1 + sample_returns).cumprod() * 100


class TestModernPortfolioTheory:
    def test_max_sharpe_weights_sum_to_one(self, sample_returns):
        mpt = ModernPortfolioTheory(sample_returns)
        result = mpt.max_sharpe_portfolio()

        weights = np.array(list(result['weights'].values()))
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_max_sharpe_weights_non_negative(self, sample_returns):
        mpt = ModernPortfolioTheory(sample_returns)
        result = mpt.max_sharpe_portfolio()

        for w in result['weights'].values():
            assert w >= -1e-8

    def test_min_vol_lower_than_max_sharpe(self, sample_returns):
        mpt = ModernPortfolioTheory(sample_returns)
        max_sharpe = mpt.max_sharpe_portfolio()
        min_vol = mpt.min_volatility_portfolio()

        assert min_vol['volatility'] <= max_sharpe['volatility'] + 1e-6

    def test_portfolio_performance_returns_three_values(self, sample_returns):
        mpt = ModernPortfolioTheory(sample_returns)
        equal_weights = np.array([1/len(sample_returns.columns)] * len(sample_returns.columns))
        ret, vol, sharpe = mpt.portfolio_performance(equal_weights)

        assert isinstance(ret, float)
        assert vol > 0
        assert isinstance(sharpe, float)

    def test_efficient_frontier_shape(self, sample_returns):
        mpt = ModernPortfolioTheory(sample_returns)
        vols, rets = mpt.efficient_frontier(num_portfolios=20)

        assert len(vols) == 20
        assert len(rets) == 20

    def test_ledoit_wolf_shrinkage(self, sample_returns):
        mpt = ModernPortfolioTheory(sample_returns, use_shrinkage=True)
        assert mpt.shrinkage_coefficient > 0


class TestFactorStrategy:
    def test_weights_sum_to_one(self, sample_returns, sample_prices):
        weights = create_factor_strategy(sample_returns, sample_prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_assets_have_weights(self, sample_returns, sample_prices):
        weights = create_factor_strategy(sample_returns, sample_prices)
        assert set(weights.keys()) == set(sample_returns.columns)

    def test_weights_within_bounds(self, sample_returns, sample_prices):
        weights = create_factor_strategy(sample_returns, sample_prices)
        for w in weights.values():
            assert w > 0
            assert w < 1
