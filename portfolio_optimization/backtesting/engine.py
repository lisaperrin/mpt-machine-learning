import logging
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self,
                 price_data: pd.DataFrame,
                 returns_data: pd.DataFrame,
                 transaction_cost: float = 0.001,
                 rebalance_frequency: int = 21,
                 initial_capital: float = 100000):

        self.prices = price_data
        self.returns = returns_data
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital

        self.portfolio_value = pd.Series(dtype=float)
        self.weights_history = pd.DataFrame()
        self.turnover_history: list[float] = []
        self.costs_history: list[float] = []

    def run_backtest(self,
                     strategy_func: Callable,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     min_history: int = 252) -> Dict:

        # Reset state for clean backtest
        self.turnover_history = []
        self.costs_history = []

        if start_date:
            start_idx = self.returns.index.get_loc(start_date)
        else:
            start_idx = min_history

        if end_date:
            end_idx = self.returns.index.get_loc(end_date)
        else:
            end_idx = len(self.returns) - 1

        portfolio_value = self.initial_capital
        current_weights = pd.Series(0.0, index=self.returns.columns)

        value_history = []
        weights_history = []

        for i in range(start_idx, end_idx + 1):
            if i == start_idx or (i - start_idx) % self.rebalance_frequency == 0:
                historical_data = self.returns.iloc[:i]
                historical_prices = self.prices.iloc[:i]

                if len(historical_data) >= min_history:
                    try:
                        new_weights = strategy_func(historical_data, historical_prices)

                        if isinstance(new_weights, dict):
                            new_weights = pd.Series(new_weights).reindex(self.returns.columns, fill_value=0.0)

                        new_weights = new_weights / new_weights.sum()

                        turnover = np.sum(np.abs(new_weights - current_weights))
                        trading_cost = turnover * self.transaction_cost * portfolio_value

                        current_weights = new_weights
                        portfolio_value -= trading_cost

                        self.turnover_history.append(turnover)
                        self.costs_history.append(trading_cost)

                    except Exception as e:
                        logger.warning(f"Strategy failed at {self.returns.index[i]}: {e}")
                        continue

            daily_returns = self.returns.iloc[i]
            portfolio_return = np.sum(current_weights * daily_returns)
            portfolio_value *= (1 + portfolio_return)

            value_history.append(portfolio_value)
            weights_history.append(current_weights.copy())

        self.portfolio_value = pd.Series(value_history,
                                         index=self.returns.index[start_idx:end_idx+1])
        self.weights_history = pd.DataFrame(weights_history,
                                            index=self.returns.index[start_idx:end_idx+1])

        return self._calculate_performance_metrics()

    def _calculate_performance_metrics(self) -> Dict:
        if len(self.portfolio_value) == 0:
            return {}

        portfolio_returns = self.portfolio_value.pct_change().dropna()

        total_return = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.045) / annualized_vol if annualized_vol > 0 else 0

        cummax = self.portfolio_value.expanding().max()
        drawdown = (self.portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min()

        total_costs = sum(self.costs_history) if self.costs_history else 0
        avg_turnover = np.mean(self.turnover_history) if self.turnover_history else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'total_trading_costs': total_costs,
            'avg_turnover': avg_turnover,
            'num_rebalances': len(self.turnover_history)
        }

    def get_portfolio_analytics(self) -> Dict:
        if len(self.weights_history) == 0:
            return {}

        avg_weights = self.weights_history.mean()
        weight_volatility = self.weights_history.std()
        max_weights = self.weights_history.max()

        return {
            'average_weights': avg_weights.to_dict(),
            'weight_volatility': weight_volatility.to_dict(),
            'max_weights': max_weights.to_dict(),
            'portfolio_concentration': (avg_weights ** 2).sum()
        }
