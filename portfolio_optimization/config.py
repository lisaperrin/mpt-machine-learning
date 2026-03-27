import os
from typing import Any, Dict, List

import yaml


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def risk_free_rate(self) -> float:
        return self._config.get('risk_free_rate', 0.045)

    @property
    def data_period(self) -> str:
        return self._config.get('data_period', '3y')

    @property
    def asset_universe(self) -> Dict[str, List[str]]:
        return self._config.get('asset_universe', {})

    @property
    def all_tickers(self) -> List[str]:
        tickers = []
        for sector_tickers in self.asset_universe.values():
            tickers.extend(sector_tickers)
        return list(set(tickers))

    @property
    def portfolio_constraints(self) -> Dict[str, float]:
        return self._config.get('portfolio_constraints', {})

    @property
    def model_params(self) -> Dict[str, Any]:
        return self._config.get('model_params', {})

    @property
    def backtesting_params(self) -> Dict[str, Any]:
        return self._config.get('backtesting', {})
