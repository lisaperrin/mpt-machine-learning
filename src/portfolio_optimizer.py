import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# from pypfopt import EfficientFrontier, risk_models, expected_returns

class ModernPortfolioTheory:
    def __init__(self, returns_data):
        self.returns = returns_data
        self.mean_returns = returns_data.mean() * 252
        self.cov_matrix = returns_data.cov() * 252
        self.num_assets = len(returns_data.columns)
        
    def portfolio_performance(self, weights, risk_free_rate=0.045):
        """
        Calculates portfolio performance metrics
        
        Args:
            weights (array): Asset weights
            risk_free_rate (float): Risk-free rate for Sharpe calculation
            
        Returns:
            tuple: (Return, Volatility, Sharpe Ratio)
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def negative_sharpe(self, weights):
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        return self.portfolio_performance(weights)[1]
    
    def max_sharpe_portfolio(self):
        num_assets = self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
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
            raise ValueError("Optimization failed")
    
    def min_volatility_portfolio(self):
        num_assets = self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1/num_assets] * num_assets)
        
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
            raise ValueError("Optimization failed")
    
    def efficient_frontier(self, num_portfolios=100):
        min_vol = self.min_volatility_portfolio()['volatility']
        max_ret = max(self.mean_returns)
        
        target_returns = np.linspace(min_vol * 1.5, max_ret * 0.9, num_portfolios)
        volatilities = []
        
        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target}
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

class PyPfOptimizer:
    
    def __init__(self, price_data):
        print("⚠️ PyPfOptimizer temporarily disabled - installation issues with dependencies")
        print("   Using custom MPT implementation instead")
        
    def max_sharpe_portfolio(self):
        raise NotImplementedError("PyPfOptimizer currently disabled - use ModernPortfolioTheory instead")
    
    def min_volatility_portfolio(self):
        raise NotImplementedError("PyPfOptimizer currently disabled - use ModernPortfolioTheory instead")