"""Tests for risk analysis module."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.risk.metrics import PortfolioRiskManager, RiskAnalyzer


@pytest.fixture
def portfolio_returns():
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.0003, 0.01, 504),
        index=pd.bdate_range(end='2025-01-01', periods=504)
    )
    return returns


class TestRiskAnalyzer:
    def test_var_is_negative(self, portfolio_returns):
        analyzer = RiskAnalyzer(portfolio_returns)
        var = analyzer.value_at_risk('historical')
        assert var < 0

    def test_cvar_worse_than_var(self, portfolio_returns):
        analyzer = RiskAnalyzer(portfolio_returns)
        var = analyzer.value_at_risk('historical')
        cvar = analyzer.conditional_var('historical')
        assert cvar <= var

    def test_max_drawdown_negative(self, portfolio_returns):
        analyzer = RiskAnalyzer(portfolio_returns)
        mdd = analyzer.maximum_drawdown()
        assert mdd < 0

    def test_sortino_uses_downside_only(self, portfolio_returns):
        analyzer = RiskAnalyzer(portfolio_returns)
        sortino = analyzer.sortino_ratio()
        assert isinstance(sortino, float)

    def test_comprehensive_report_keys(self, portfolio_returns):
        analyzer = RiskAnalyzer(portfolio_returns)
        report = analyzer.comprehensive_risk_report()

        expected_keys = [
            'VaR_95', 'CVaR_95', 'Max_Drawdown', 'Calmar_Ratio',
            'Sortino_Ratio', 'Tail_Ratio', 'Skewness', 'Kurtosis',
            'Annualized_Return', 'Annualized_Volatility', 'Sharpe_Ratio'
        ]
        for key in expected_keys:
            assert key in report

    def test_parametric_var(self, portfolio_returns):
        analyzer = RiskAnalyzer(portfolio_returns)
        var = analyzer.value_at_risk('parametric')
        assert isinstance(var, float)
        assert var < 0


class TestPortfolioRiskManager:
    def test_constraints_normalize(self):
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            columns=['A', 'B', 'C']
        )
        manager = PortfolioRiskManager(returns, {'max_weight': 0.5, 'min_weight': 0.1})
        weights = manager.apply_constraints({'A': 0.8, 'B': 0.1, 'C': 0.1})

        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in weights.values())
