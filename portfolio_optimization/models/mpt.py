from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


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
        # Only adjust constraints if they're mathematically impossible
        # For portfolio optimization, it's fine if sum of max_weights > 1.0
        # The constraint will ensure weights sum to 1.0 while respecting bounds
        if self.num_assets * min_weight > 1.0:
            min_weight = 0.8 / self.num_assets  # Leave some room

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

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
                    'sharpe_ratio': sharpe
                }
            else:
                print(f"Max Sharpe optimization failed: {result.message}")
                # Try with relaxed constraints
                try:
                    relaxed_bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
                    result2 = minimize(self.negative_sharpe, initial_guess,
                                     method='SLSQP', bounds=relaxed_bounds, constraints=constraints)
                    if result2.success:
                        optimal_weights = result2.x
                        ret, vol, sharpe = self.portfolio_performance(optimal_weights)
                        return {
                            'weights': dict(zip(self.returns.columns, optimal_weights)),
                            'return': ret,
                            'volatility': vol,
                            'sharpe_ratio': sharpe
                        }
                except Exception:
                    pass

                # Fallback to equal weights if optimization fails
                equal_weights = np.array([1/self.num_assets] * self.num_assets)
                ret, vol, sharpe = self.portfolio_performance(equal_weights)
                print("Max Sharpe falling back to equal weights")
                return {
                    'weights': dict(zip(self.returns.columns, equal_weights)),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                }
        except Exception as e:
            print(f"Max Sharpe optimization error: {e}")
            # Fallback to equal weights if any error occurs
            equal_weights = np.array([1/self.num_assets] * self.num_assets)
            ret, vol, sharpe = self.portfolio_performance(equal_weights)
            return {
                'weights': dict(zip(self.returns.columns, equal_weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }

    def min_volatility_portfolio(self, min_weight: float = 0.0, max_weight: float = 1.0) -> Dict:
        if self.num_assets * min_weight > 1.0:
            min_weight = 0.8 / self.num_assets

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(self.num_assets))
        initial_guess = np.array([1/self.num_assets] * self.num_assets)

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
                    'sharpe_ratio': sharpe
                }
            else:
                print(f"Min Vol optimization failed: {result.message}")
                # Try with relaxed constraints
                try:
                    relaxed_bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
                    result2 = minimize(self.portfolio_volatility, initial_guess,
                                     method='SLSQP', bounds=relaxed_bounds, constraints=constraints)
                    if result2.success:
                        optimal_weights = result2.x
                        ret, vol, sharpe = self.portfolio_performance(optimal_weights)
                        return {
                            'weights': dict(zip(self.returns.columns, optimal_weights)),
                            'return': ret,
                            'volatility': vol,
                            'sharpe_ratio': sharpe
                        }
                except Exception:
                    pass

                # Fallback to equal weights if optimization fails
                equal_weights = np.array([1/self.num_assets] * self.num_assets)
                ret, vol, sharpe = self.portfolio_performance(equal_weights)
                print("Min Vol falling back to equal weights")
                return {
                    'weights': dict(zip(self.returns.columns, equal_weights)),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                }
        except Exception as e:
            print(f"Min Vol optimization error: {e}")
            # Fallback to equal weights if any error occurs
            equal_weights = np.array([1/self.num_assets] * self.num_assets)
            ret, vol, sharpe = self.portfolio_performance(equal_weights)
            return {
                'weights': dict(zip(self.returns.columns, equal_weights)),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }

    def efficient_frontier(self, num_portfolios: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        min_vol = self.min_volatility_portfolio()['volatility']
        max_ret = max(self.mean_returns)

        target_returns = np.linspace(min_vol * 1.5, max_ret * 0.9, num_portfolios)
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
