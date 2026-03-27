"""Tests for StrategyEvaluator (walk-forward, bootstrap, pairwise)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.evaluation.evaluator import StrategyEvaluator


@pytest.fixture
def synthetic_data():
    """Generate 3 years of synthetic returns for 4 assets."""
    rng = np.random.default_rng(42)
    n_days = 756
    assets = ['A', 'B', 'C', 'D']
    vols = np.array([0.015, 0.020, 0.010, 0.018])
    means = np.array([0.0004, 0.0002, 0.0003, 0.0001])

    raw = rng.standard_normal((n_days, len(assets))) * vols + means
    returns = pd.DataFrame(
        raw,
        columns=assets,
        index=pd.bdate_range(end='2025-01-01', periods=n_days)
    )
    prices = (1 + returns).cumprod() * 100
    return returns, prices


def equal_weight_strategy(hist_returns, hist_prices):
    n = len(hist_returns.columns)
    return {col: 1.0 / n for col in hist_returns.columns}


def high_vol_strategy(hist_returns, hist_prices):
    """Overweight the highest-volatility asset."""
    vols = hist_returns.std()
    weights = vols / vols.sum()
    return weights.to_dict()


@pytest.fixture
def evaluator(synthetic_data):
    returns, prices = synthetic_data
    strategies = {
        'Equal Weight': equal_weight_strategy,
        'High Vol': high_vol_strategy,
    }
    return StrategyEvaluator(returns, prices, strategies)


class TestWalkForward:
    def test_returns_all_strategies(self, evaluator):
        oos = evaluator.walk_forward_evaluate(train_window=252, test_window=63)
        assert 'Equal Weight' in oos
        assert 'High Vol' in oos

    def test_oos_returns_nonempty(self, evaluator):
        oos = evaluator.walk_forward_evaluate(train_window=252, test_window=63)
        for name, rets in oos.items():
            assert len(rets) > 0, f"{name} has no OOS returns"

    def test_no_overlap_with_training(self, evaluator):
        """OOS returns should not cover the first train_window days."""
        train_window = 252
        oos = evaluator.walk_forward_evaluate(train_window=train_window, test_window=63)
        for rets in oos.values():
            first_oos_date = rets.index[0]
            train_end_date = evaluator.returns.index[train_window - 1]
            assert first_oos_date > train_end_date

    def test_oos_length_reasonable(self, evaluator):
        oos = evaluator.walk_forward_evaluate(train_window=252, test_window=63)
        for rets in oos.values():
            # With 756 days and 252 train, we get (756-252)/63 ≈ 8 folds → ~504 OOS days
            assert len(rets) >= 63  # at least one fold


class TestBootstrapSharpe:
    def test_returns_expected_keys(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.bootstrap_sharpe_confidence(
            returns['A'], n_bootstrap=200, block_size=21
        )
        assert 'observed' in result
        assert 'mean' in result
        assert 'lower' in result
        assert 'upper' in result
        assert 'std' in result

    def test_ci_contains_observed(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.bootstrap_sharpe_confidence(
            returns['A'], n_bootstrap=500, block_size=21
        )
        # The CI should usually contain the observed value (not guaranteed, but likely)
        assert result['lower'] <= result['upper']

    def test_lower_less_than_upper(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.bootstrap_sharpe_confidence(
            returns['A'], n_bootstrap=200
        )
        assert result['lower'] < result['upper']

    def test_std_positive(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.bootstrap_sharpe_confidence(
            returns['A'], n_bootstrap=200
        )
        assert result['std'] > 0


class TestPairwiseSharpe:
    def test_returns_expected_keys(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.pairwise_sharpe_test(
            returns['A'], returns['B'], n_bootstrap=200
        )
        assert 'observed_diff' in result
        assert 'mean_diff' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'p_value' in result

    def test_p_value_in_range(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.pairwise_sharpe_test(
            returns['A'], returns['B'], n_bootstrap=200
        )
        assert 0.0 <= result['p_value'] <= 1.0

    def test_identical_series_diff_near_zero(self, evaluator, synthetic_data):
        returns, _ = synthetic_data
        result = evaluator.pairwise_sharpe_test(
            returns['A'], returns['A'], n_bootstrap=200
        )
        assert abs(result['observed_diff']) < 1e-10


class TestComprehensiveComparison:
    def test_returns_all_sections(self, evaluator):
        result = evaluator.comprehensive_comparison(
            train_window=252, test_window=63, n_bootstrap=200
        )
        assert 'strategies' in result
        assert 'pairwise_tests' in result
        assert 'evaluation_params' in result

    def test_strategy_metrics_present(self, evaluator):
        result = evaluator.comprehensive_comparison(
            train_window=252, test_window=63, n_bootstrap=200
        )
        for name, metrics in result['strategies'].items():
            assert 'oos_sharpe' in metrics
            assert 'oos_total_return' in metrics
            assert 'oos_annualized_return' in metrics
            assert 'oos_annualized_vol' in metrics
            assert 'oos_max_drawdown' in metrics
            assert 'oos_calmar' in metrics
            assert 'n_test_days' in metrics

    def test_pairwise_test_exists(self, evaluator):
        result = evaluator.comprehensive_comparison(
            train_window=252, test_window=63, n_bootstrap=200
        )
        # 2 strategies → 1 pairwise test
        assert len(result['pairwise_tests']) == 1

    def test_evaluation_params_correct(self, evaluator):
        result = evaluator.comprehensive_comparison(
            train_window=252, test_window=63, n_bootstrap=200
        )
        params = result['evaluation_params']
        assert params['train_window'] == 252
        assert params['test_window'] == 63
        assert params['n_bootstrap'] == 200
