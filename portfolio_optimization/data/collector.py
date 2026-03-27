import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self):
        self.data = None
        self.failed_tickers = []

    def fetch_stock_data(self, tickers: List[str], period: str = "3y") -> Optional[pd.DataFrame]:
        try:
            logger.info(f"Fetching data for {len(tickers)} tickers over {period}")

            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)

            if data.empty:
                raise ValueError("No data returned from yfinance")

            if len(tickers) == 1:
                if 'Close' in data.columns:
                    self.data = data[['Close']].copy()
                    self.data.columns = tickers
                else:
                    self.data = data.iloc[:, [-1]].copy()
                    self.data.columns = tickers
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        self.data = data['Close'].copy()
                    except KeyError:
                        try:
                            self.data = data['Adj Close'].copy()
                        except KeyError:
                            first_level = data.columns.get_level_values(0)[0]
                            self.data = data[first_level].copy()
                else:
                    close_cols = [col for col in data.columns if 'close' in str(col).lower()]
                    if close_cols:
                        self.data = data[close_cols].copy()
                    else:
                        self.data = data.copy()

            if self.data is None or self.data.empty:
                raise ValueError("No valid data received")

            # Identify tickers that returned all NaN (failed downloads)
            valid_cols = self.data.columns[self.data.notna().any()]
            failed_cols = [t for t in tickers if t not in valid_cols]
            self.data = self.data[valid_cols]

            if failed_cols:
                logger.warning(f"No data returned for: {failed_cols}")
            self.failed_tickers = failed_cols

            self.data = self.data.dropna()
            self.data = self._validate_and_clean_data(self.data)

            logger.info(f"Data loaded: {self.data.shape[0]} days, {self.data.shape[1]} assets")

            return self.data

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return None

    def calculate_returns(self, method: str = 'simple') -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data available. Run fetch_stock_data() first.")

        if method == 'simple':
            returns = self.data.pct_change().dropna()
        elif method == 'log':
            returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")

        return returns

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned_data = data.copy()

        for column in cleaned_data.columns:
            prices = cleaned_data[column]
            returns = prices.pct_change().dropna()

            if len(returns) == 0:
                continue

            return_std = returns.std()
            return_mean = returns.mean()
            outlier_threshold = 5 * return_std

            extreme_outliers = abs(returns - return_mean) > outlier_threshold

            if extreme_outliers.sum() > 0:
                logger.info(f"Cleaning {extreme_outliers.sum()} outliers from {column}")
                outlier_indices = extreme_outliers[extreme_outliers].index
                for idx in outlier_indices:
                    if idx in cleaned_data.index:
                        prev_idx = cleaned_data.index[cleaned_data.index < idx]
                        if len(prev_idx) > 0:
                            cleaned_data.loc[idx, column] = cleaned_data.loc[prev_idx[-1], column]

            invalid_prices = (prices <= 0) | prices.isna()
            if invalid_prices.sum() > 0:
                logger.info(f"Fixing {invalid_prices.sum()} invalid prices in {column}")
                cleaned_data.loc[invalid_prices, column] = np.nan
                cleaned_data[column] = cleaned_data[column].ffill().bfill()

        if len(cleaned_data) < 252:
            logger.warning(f"Only {len(cleaned_data)} data points available")

        return cleaned_data

    def get_data_quality_report(self) -> Dict:
        if self.data is None:
            return {}

        return {
            'total_assets': len(self.data.columns),
            'failed_assets': len(self.failed_tickers),
            'failed_tickers': self.failed_tickers,
            'date_range': (str(self.data.index.min()), str(self.data.index.max())),
            'data_points': len(self.data),
            'missing_data_pct': (self.data.isna().sum() / len(self.data) * 100).to_dict()
        }
