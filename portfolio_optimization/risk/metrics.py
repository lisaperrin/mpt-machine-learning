from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


class RiskAnalyzer:
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.045, confidence_level: float = 0.05):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level

    def value_at_risk(self, method: str = 'historical') -> float:
        if method == 'historical':
            return np.percentile(self.returns, self.confidence_level * 100)
        elif method == 'parametric':
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            return mean - std * stats.norm.ppf(1 - self.confidence_level)
        elif method == 'monte_carlo':
            simulated_returns = np.random.normal(
                np.mean(self.returns), np.std(self.returns), 10000
            )
            return np.percentile(simulated_returns, self.confidence_level * 100)
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")

    def conditional_var(self, method: str = 'historical') -> float:
        var = self.value_at_risk(method)
        if method == 'historical':
            return np.mean(self.returns[self.returns <= var])
        elif method == 'parametric':
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            phi = stats.norm.pdf(stats.norm.ppf(self.confidence_level))
            return mean - std * phi / self.confidence_level
        else:
            return self.value_at_risk(method)

    def maximum_drawdown(self) -> float:
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def calmar_ratio(self) -> float:
        annual_return = np.mean(self.returns) * 252
        max_dd = abs(self.maximum_drawdown())
        return (annual_return - self.risk_free_rate) / max_dd if max_dd > 0 else 0

    def sortino_ratio(self) -> float:
        annual_return = np.mean(self.returns) * 252
        mar = self.risk_free_rate / 252  # daily minimum acceptable return
        downside_diff = np.minimum(self.returns - mar, 0)
        downside_deviation = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
        return (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        excess_returns = self.returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        return (np.mean(excess_returns) * 252) / tracking_error if tracking_error > 0 else 0

    def tail_ratio(self) -> float:
        positive_tail = np.mean(self.returns[self.returns > np.percentile(self.returns, 95)])
        negative_tail = abs(np.mean(self.returns[self.returns < np.percentile(self.returns, 5)]))
        return positive_tail / negative_tail if negative_tail > 0 else 0

    def comprehensive_risk_report(self, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        ann_return = np.mean(self.returns) * 252
        ann_vol = np.std(self.returns) * np.sqrt(252)
        metrics = {
            'VaR_95': self.value_at_risk('historical'),
            'CVaR_95': self.conditional_var('historical'),
            'Max_Drawdown': self.maximum_drawdown(),
            'Calmar_Ratio': self.calmar_ratio(),
            'Sortino_Ratio': self.sortino_ratio(),
            'Tail_Ratio': self.tail_ratio(),
            'Skewness': stats.skew(self.returns),
            'Kurtosis': stats.kurtosis(self.returns),
            'Annualized_Return': ann_return,
            'Annualized_Volatility': ann_vol,
            'Sharpe_Ratio': (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0,
            'Risk_Free_Rate': self.risk_free_rate
        }

        if benchmark_returns is not None and len(benchmark_returns) == len(self.returns):
            metrics['Information_Ratio'] = self.information_ratio(benchmark_returns)

        return metrics

class PortfolioRiskManager:
    def __init__(self, returns_data: pd.DataFrame, constraints: Dict):
        self.returns = returns_data
        self.max_position_size = constraints.get('max_weight', 0.25)
        self.min_position_size = constraints.get('min_weight', 0.01)
        self.max_sector_exposure = constraints.get('max_sector_weight', 0.40)

    def apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        weights_array = np.array(list(weights.values()))
        asset_names = list(weights.keys())

        weights_array = np.clip(weights_array, self.min_position_size, self.max_position_size)

        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones(len(weights_array)) / len(weights_array)

        return dict(zip(asset_names, weights_array))

    def calculate_portfolio_risk_metrics(self, weights: Dict[str, float]) -> Dict:
        weights_series = pd.Series(weights)
        portfolio_returns = (self.returns * weights_series).sum(axis=1).dropna()

        risk_analyzer = RiskAnalyzer(portfolio_returns)
        return risk_analyzer.comprehensive_risk_report()

    def stress_test_portfolio(self, weights: Dict[str, float], scenarios: Dict) -> Dict:
        results = {}
        weights_series = pd.Series(weights)
        portfolio_returns = (self.returns * weights_series).sum(axis=1).dropna()

        for scenario_name, shock in scenarios.items():
            if isinstance(shock, dict):
                shocked_returns = portfolio_returns.copy()
                for asset, shock_value in shock.items():
                    if asset in weights:
                        shocked_returns += weights[asset] * shock_value
            else:
                shocked_returns = portfolio_returns + shock

            risk_analyzer = RiskAnalyzer(shocked_returns)
            results[scenario_name] = {
                'return': np.mean(shocked_returns) * 252,
                'volatility': np.std(shocked_returns) * np.sqrt(252),
                'max_drawdown': risk_analyzer.maximum_drawdown(),
                'var_95': risk_analyzer.value_at_risk()
            }

        return results
