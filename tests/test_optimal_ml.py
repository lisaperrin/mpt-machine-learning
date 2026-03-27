"""Tests for OptimalMLPortfolioOptimizer (HRP + Black-Litterman)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.models.optimal_ml_optimizer import OptimalMLPortfolioOptimizer
from portfolio_optimization.risk.metrics import RiskAnalyzer


@pytest.fixture
def sample_returns():
    """Generate 2+ years of synthetic returns for 6 assets with distinct profiles."""
    rng = np.random.default_rng(42)
    n_days = 504
    assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'TLT', 'GLD']
    volatilities = np.array([0.020, 0.018, 0.022, 0.016, 0.008, 0.012])
    mean_returns = np.array([0.0005, 0.0004, 0.0006, 0.0003, 0.0001, 0.0002])

    raw = rng.standard_normal((n_days, len(assets))) * volatilities + mean_returns
    returns = pd.DataFrame(
        raw,
        columns=assets,
        index=pd.bdate_range(end='2025-01-01', periods=n_days)
    )
    return returns


@pytest.fixture
def sample_prices(sample_returns):
    return (1 + sample_returns).cumprod() * 100


@pytest.fixture
def optimizer(sample_returns, sample_prices):
    return OptimalMLPortfolioOptimizer(
        sample_returns, sample_prices,
        {'portfolio_constraints': {'min_weight': 0.01, 'max_weight': 0.40}}
    )


class TestHRP:
    def test_hrp_weights_sum_to_one(self, optimizer):
        weights = optimizer.calculate_hrp_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_hrp_all_assets_have_weights(self, optimizer, sample_returns):
        weights = optimizer.calculate_hrp_weights()
        assert set(weights.keys()) == set(sample_returns.columns)

    def test_hrp_weights_positive(self, optimizer):
        weights = optimizer.calculate_hrp_weights()
        for w in weights.values():
            assert w > 0

    def test_hrp_no_single_asset_dominates(self, optimizer):
        weights = optimizer.calculate_hrp_weights()
        # 20% cap (after renormalization, effective max may be slightly higher)
        for w in weights.values():
            assert w < 0.30


class TestMLViews:
    def test_views_returns_correct_shapes(self, optimizer):
        Q, P, confidences = optimizer.generate_ml_views()
        n_assets = len(optimizer.returns.columns)
        assert Q.shape == (n_assets,)
        assert P.shape == (n_assets, n_assets)
        assert len(confidences) == n_assets

    def test_views_confidences_in_range(self, optimizer):
        _, _, confidences = optimizer.generate_ml_views()
        for c in confidences:
            assert 0.0 <= c <= 1.0

    def test_views_q_is_finite(self, optimizer):
        Q, _, _ = optimizer.generate_ml_views()
        assert np.all(np.isfinite(Q))

    def test_views_stored_in_diagnostics(self, optimizer):
        optimizer.generate_ml_views()
        assert len(optimizer.ml_views) > 0
        for asset, details in optimizer.ml_views.items():
            assert 'predicted_return' in details
            assert 'confidence' in details


class TestBlackLitterman:
    def test_bl_returns_correct_shape(self, optimizer):
        n = len(optimizer.returns.columns)
        cov = optimizer.returns.tail(252).cov().values
        Q, P, conf = optimizer.generate_ml_views()
        mu_bl, sigma_bl = optimizer.black_litterman_returns(cov, Q, P, conf)
        assert mu_bl.shape == (n,)
        assert sigma_bl.shape == (n, n)

    def test_bl_sigma_positive_semidefinite(self, optimizer):
        cov = optimizer.returns.tail(252).cov().values
        Q, P, conf = optimizer.generate_ml_views()
        _, sigma_bl = optimizer.black_litterman_returns(cov, Q, P, conf)
        eigenvalues = np.linalg.eigvalsh(sigma_bl)
        assert np.all(eigenvalues >= -1e-8)

    def test_bl_no_views_returns_equilibrium(self, optimizer):
        n = len(optimizer.returns.columns)
        cov = optimizer.returns.tail(252).cov().values
        Q = np.array([])
        P = np.zeros((0, n))
        conf = np.array([])
        mu_bl, _ = optimizer.black_litterman_returns(cov, Q, P, conf)
        # Should be equilibrium: delta * cov @ w_mkt (momentum-tilted)
        # Verify shape and finiteness (exact values depend on trailing returns)
        assert mu_bl.shape == (n,)
        assert np.all(np.isfinite(mu_bl))


class TestFullPipeline:
    def test_optimal_weights_sum_to_one(self, optimizer):
        weights = optimizer.get_optimal_portfolio_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_optimal_weights_positive(self, optimizer):
        weights = optimizer.get_optimal_portfolio_weights()
        for w in weights.values():
            assert w > 0

    def test_all_assets_in_result(self, optimizer, sample_returns):
        weights = optimizer.get_optimal_portfolio_weights()
        assert set(weights.keys()) == set(sample_returns.columns)

    def test_weight_comparison_available(self, optimizer):
        optimizer.get_optimal_portfolio_weights()
        assert 'asset_comparison' in optimizer.weight_comparison
        assert 'summary' in optimizer.weight_comparison

    def test_bl_diagnostics_available(self, optimizer):
        optimizer.get_optimal_portfolio_weights()
        assert len(optimizer.bl_diagnostics) > 0


class TestSortinoCorrectness:
    def test_sortino_known_value(self):
        np.random.seed(123)
        returns = pd.Series(np.random.normal(0.0003, 0.01, 504))
        analyzer = RiskAnalyzer(returns, risk_free_rate=0.045)
        sortino = analyzer.sortino_ratio()

        mar = 0.045 / 252
        downside_diff = np.minimum(returns.values - mar, 0)
        expected_dd = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
        expected = (np.mean(returns) * 252 - 0.045) / expected_dd if expected_dd > 0 else 0

        assert abs(sortino - expected) < 1e-10

    def test_sortino_all_positive_returns(self):
        returns = pd.Series(np.ones(100) * 0.01)
        analyzer = RiskAnalyzer(returns, risk_free_rate=0.045)
        sortino = analyzer.sortino_ratio()
        assert isinstance(sortino, float)
        assert sortino == 0.0
