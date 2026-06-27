import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from portfolio_optimization.utils.constraints import project_weights_to_bounds, validate_weight_bounds

logger = logging.getLogger(__name__)


class ModernPortfolioTheory:
    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.045,
                 use_shrinkage: bool = True):
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns_data.mean() * 252
        self.use_shrinkage = use_shrinkage

        if use_shrinkage and len(returns_data) > returns_data.shape[1]:
            try:
                lw = LedoitWolf().fit(returns_data.dropna())
                self.cov_matrix = pd.DataFrame(
                    lw.covariance_ * 252,
                    index=returns_data.columns,
                    columns=returns_data.columns
                )
                self.shrinkage_coefficient = lw.shrinkage_
            except Exception:
                self.cov_matrix = returns_data.cov() * 252
                self.shrinkage_coefficient = 0.0
        else:
            self.cov_matrix = returns_data.cov() * 252
            self.shrinkage_coefficient = 0.0

        self.num_assets = len(returns_data.columns)

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

        return portfolio_return, portfolio_std, sharpe_ratio

    def negative_sharpe(self, weights: np.ndarray) -> float:
        return -self.portfolio_performance(weights)[2]

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return self.portfolio_performance(weights)[1]

    def max_sharpe_portfolio(self, min_weight: float = 0.0, max_weight: float = 1.0) -> Dict:
        min_weight, max_weight = validate_weight_bounds(self.num_assets, min_weight, max_weight)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(self.num_assets))
        initial_guess = project_weights_to_bounds(
            np.array([1/self.num_assets] * self.num_assets),
            min_weight,
            max_weight
        )

        try:
            result = minimize(self.negative_sharpe, initial_guess,
                             method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = result.x
                ret, vol, sharpe = self.portfolio_performance(optimal_weights)

                return {
                    'weights': dict(zip(self.returns.columns, optimal_weights)),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'diagnostics': {'status': 'optimized'}
                }
            else:
                logger.warning("Max Sharpe optimization failed: %s", result.message)
                equal_weights = project_weights_to_bounds(initial_guess, min_weight, max_weight)
                ret, vol, sharpe = self.portfolio_performance(equal_weights)
                return {
                    'weights': dict(zip(self.returns.columns, equal_weights)),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'diagnostics': {
                        'status': 'fallback_equal_weight',
                        'reason': str(result.message)
                    }
                }
        except Exception as e:
            logger.warning("Max Sharpe optimization error: %s", e)
            equal_weights = project_weights_to_bounds(initial_guess, min_weight, max_weight)
            ret, vol, sharpe = self.portfolio_performance(equal_weights)
            return {
                'weights': dict(zip(self.returns.columns, equal_weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'diagnostics': {
                    'status': 'fallback_equal_weight',
                    'reason': str(e)
                }
            }

    def min_volatility_portfolio(self, min_weight: float = 0.0, max_weight: float = 1.0) -> Dict:
        min_weight, max_weight = validate_weight_bounds(self.num_assets, min_weight, max_weight)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(self.num_assets))
        initial_guess = project_weights_to_bounds(
            np.array([1/self.num_assets] * self.num_assets),
            min_weight,
            max_weight
        )

        try:
            result = minimize(self.portfolio_volatility, initial_guess,
                             method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = result.x
                ret, vol, sharpe = self.portfolio_performance(optimal_weights)

                return {
                    'weights': dict(zip(self.returns.columns, optimal_weights)),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'diagnostics': {'status': 'optimized'}
                }
            else:
                logger.warning("Min Vol optimization failed: %s", result.message)
                equal_weights = project_weights_to_bounds(initial_guess, min_weight, max_weight)
                ret, vol, sharpe = self.portfolio_performance(equal_weights)
                return {
                    'weights': dict(zip(self.returns.columns, equal_weights)),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'diagnostics': {
                        'status': 'fallback_equal_weight',
                        'reason': str(result.message)
                    }
                }
        except Exception as e:
            logger.warning("Min Vol optimization error: %s", e)
            equal_weights = project_weights_to_bounds(initial_guess, min_weight, max_weight)
            ret, vol, sharpe = self.portfolio_performance(equal_weights)
            return {
                'weights': dict(zip(self.returns.columns, equal_weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'diagnostics': {
                    'status': 'fallback_equal_weight',
                    'reason': str(e)
                }
            }

    def efficient_frontier(self, num_portfolios: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        min_ret = self.min_volatility_portfolio()['return']
        max_ret = max(self.mean_returns)
        if max_ret <= min_ret:
            max_ret = min_ret + 1e-6

        upper_ret = max(max_ret * 0.95, min_ret + 1e-6)
        target_returns = np.linspace(min_ret, upper_ret, num_portfolios)
        volatilities = []

        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, t=target: self.portfolio_performance(x)[0] - t}
            ]
            bounds = tuple((0, 1) for _ in range(self.num_assets))
            initial_guess = np.array([1/self.num_assets] * self.num_assets)

            result = minimize(self.portfolio_volatility, initial_guess,
                             method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                volatilities.append(result.fun)
            else:
                volatilities.append(np.nan)

        return np.array(volatilities), target_returns
