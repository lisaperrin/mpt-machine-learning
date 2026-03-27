import logging
from typing import Callable, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Evaluate and compare portfolio strategies with statistical rigor."""

    def __init__(self, returns_data: pd.DataFrame, price_data: pd.DataFrame,
                 strategies: Dict[str, Callable]):
        """
        Args:
            returns_data: Daily returns (dates × assets)
            price_data: Daily prices (dates × assets)
            strategies: {name: func(returns, prices) -> {asset: weight}}
        """
        self.returns = returns_data
        self.prices = price_data
        self.strategies = strategies

    def walk_forward_evaluate(self, train_window: int = 252,
                               test_window: int = 63) -> Dict[str, pd.Series]:
        """
        Walk-forward out-of-sample evaluation with non-overlapping test windows.

        At each step:
        1. Train on [start : start + train_window]
        2. Get portfolio weights from strategy
        3. Evaluate on [start + train_window : start + train_window + test_window]
        4. Advance by test_window (no overlap)

        Returns:
            {strategy_name: pd.Series of OOS daily portfolio returns}
        """
        n_days = len(self.returns)
        oos_returns = {name: [] for name in self.strategies}
        oos_indices = {name: [] for name in self.strategies}
        n_folds = 0

        fold_start = 0
        while fold_start + train_window + test_window <= n_days:
            train_end = fold_start + train_window
            test_end = train_end + test_window

            train_returns = self.returns.iloc[fold_start:train_end]
            train_prices = self.prices.iloc[fold_start:train_end]
            test_returns = self.returns.iloc[train_end:test_end]

            for name, strategy_func in self.strategies.items():
                try:
                    weights = strategy_func(train_returns, train_prices)
                    weights_series = pd.Series(weights).reindex(
                        self.returns.columns, fill_value=0.0
                    )
                    if weights_series.sum() > 0:
                        weights_series /= weights_series.sum()

                    portfolio_rets = (test_returns * weights_series).sum(axis=1)
                    oos_returns[name].extend(portfolio_rets.values.tolist())
                    oos_indices[name].extend(test_returns.index.tolist())
                except Exception as e:
                    logger.warning(f"Strategy {name} failed on fold {n_folds}: {e}")
                    # Use equal weight as fallback for this fold
                    eq_w = 1.0 / len(self.returns.columns)
                    portfolio_rets = (test_returns * eq_w).sum(axis=1)
                    oos_returns[name].extend(portfolio_rets.values.tolist())
                    oos_indices[name].extend(test_returns.index.tolist())

            fold_start += test_window
            n_folds += 1

        logger.info(f"Walk-forward evaluation: {n_folds} folds, "
                    f"~{n_folds * test_window} OOS days")

        result = {}
        for name in self.strategies:
            if oos_indices[name]:
                result[name] = pd.Series(
                    oos_returns[name],
                    index=pd.DatetimeIndex(oos_indices[name])
                )
            else:
                result[name] = pd.Series(dtype=float)

        return result

    def bootstrap_sharpe_confidence(self, returns: pd.Series,
                                     n_bootstrap: int = 1000,
                                     block_size: int = 21,
                                     ci: float = 0.95) -> Dict:
        """
        Block bootstrap confidence interval for annualized Sharpe ratio.

        Uses circular block bootstrap to preserve time-series dependence.

        Args:
            returns: Daily return series
            n_bootstrap: Number of bootstrap resamples
            block_size: Block length (21 ≈ 1 month)
            ci: Confidence level (0.95 = 95% CI)

        Returns:
            {mean, lower, upper, std, observed}
        """
        values = returns.values
        n = len(values)

        if n < block_size:
            block_size = max(1, n // 2)

        # Observed Sharpe
        observed = self._annualized_sharpe(values)

        rng = np.random.default_rng(42)
        n_blocks = int(np.ceil(n / block_size))
        sharpe_samples = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Sample block start indices
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            resampled = np.concatenate([values[s:s + block_size] for s in starts])[:n]
            sharpe_samples[b] = self._annualized_sharpe(resampled)

        alpha = 1.0 - ci
        lower = float(np.percentile(sharpe_samples, 100 * alpha / 2))
        upper = float(np.percentile(sharpe_samples, 100 * (1 - alpha / 2)))

        return {
            'observed': float(observed),
            'mean': float(np.mean(sharpe_samples)),
            'lower': lower,
            'upper': upper,
            'std': float(np.std(sharpe_samples))
        }

    def pairwise_sharpe_test(self, returns_a: pd.Series, returns_b: pd.Series,
                              n_bootstrap: int = 1000,
                              block_size: int = 21) -> Dict:
        """
        Paired block bootstrap test for Sharpe ratio difference.

        Uses the same block indices for both series to preserve correlation.

        Returns:
            {observed_diff, mean_diff, ci_lower, ci_upper, p_value}
        """
        # Align by index
        common = returns_a.index.intersection(returns_b.index)
        a = returns_a.loc[common].values
        b = returns_b.loc[common].values
        n = len(a)

        if n < block_size:
            block_size = max(1, n // 2)

        observed_diff = self._annualized_sharpe(a) - self._annualized_sharpe(b)

        rng = np.random.default_rng(42)
        n_blocks = int(np.ceil(n / block_size))
        diffs = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            a_resampled = np.concatenate([a[s:s + block_size] for s in starts])[:n]
            b_resampled = np.concatenate([b[s:s + block_size] for s in starts])[:n]
            diffs[i] = self._annualized_sharpe(a_resampled) - self._annualized_sharpe(b_resampled)

        # Two-sided p-value: proportion where sign flips
        if observed_diff >= 0:
            p_value = float(np.mean(diffs < 0))
        else:
            p_value = float(np.mean(diffs > 0))
        # Two-sided
        p_value = min(2 * p_value, 1.0)

        return {
            'observed_diff': float(observed_diff),
            'mean_diff': float(np.mean(diffs)),
            'ci_lower': float(np.percentile(diffs, 2.5)),
            'ci_upper': float(np.percentile(diffs, 97.5)),
            'p_value': p_value
        }

    def comprehensive_comparison(self, train_window: int = 252,
                                  test_window: int = 63,
                                  n_bootstrap: int = 1000) -> Dict:
        """
        Full out-of-sample evaluation: walk-forward + bootstrap CIs + pairwise tests.

        Returns structured results for all strategies.
        """
        logger.info("Running comprehensive strategy evaluation...")

        # 1. Walk-forward OOS returns
        oos_returns = self.walk_forward_evaluate(train_window, test_window)

        # 2. Per-strategy metrics with bootstrap CIs
        strategy_results = {}
        for name, rets in oos_returns.items():
            if len(rets) < 21:
                logger.warning(f"Strategy {name} has too few OOS returns ({len(rets)})")
                continue

            sharpe_ci = self.bootstrap_sharpe_confidence(rets, n_bootstrap=n_bootstrap)

            # Additional OOS metrics
            cumulative = (1 + rets).cumprod()
            total_return = float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0.0
            ann_return = float(rets.mean() * 252)
            ann_vol = float(rets.std() * np.sqrt(252))

            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = float(drawdown.min())

            calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

            strategy_results[name] = {
                'oos_sharpe': sharpe_ci,
                'oos_total_return': total_return,
                'oos_annualized_return': ann_return,
                'oos_annualized_vol': ann_vol,
                'oos_max_drawdown': max_dd,
                'oos_calmar': calmar,
                'n_test_days': len(rets)
            }

        # 3. Pairwise Sharpe tests
        names = list(strategy_results.keys())
        pairwise = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, b_name = names[i], names[j]
                test_result = self.pairwise_sharpe_test(
                    oos_returns[a_name], oos_returns[b_name],
                    n_bootstrap=n_bootstrap
                )
                pairwise[f"{a_name} vs {b_name}"] = test_result

        n_total = len(self.returns)
        n_folds = max(1, (n_total - train_window) // test_window)

        return {
            'strategies': strategy_results,
            'pairwise_tests': pairwise,
            'evaluation_params': {
                'train_window': train_window,
                'test_window': test_window,
                'n_folds': n_folds,
                'n_bootstrap': n_bootstrap,
                'total_days': n_total
            }
        }

    @staticmethod
    def _annualized_sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) * 252 / (np.std(returns) * np.sqrt(252)))
