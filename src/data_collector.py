import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        self.data = None
        
    def fetch_stock_data(self, tickers, period="2y"):
        try:
            print(f"Loading data for {tickers} over {period}...")
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            
            print(f"Raw data shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            
            if len(tickers) == 1:
                if 'Close' in data.columns:
                    self.data = data[['Close']].copy()
                    self.data.columns = tickers
                else:
                    close_cols = [col for col in data.columns if 'Close' in str(col)]
                    if close_cols:
                        self.data = data[close_cols].copy()
                        self.data.columns = tickers
                    else:
                        print("Warning: No Close column found, using last column")
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
                            print("Warning: Neither Close nor Adj Close found, using first level")
                            first_level = data.columns.get_level_values(0)[0]
                            self.data = data[first_level].copy()
                else:
                    close_cols = [col for col in data.columns if any(word in str(col).lower() for word in ['close', 'adj'])]
                    if close_cols:
                        self.data = data[close_cols].copy()
                    else:
                        print("Warning: No Close columns found, using all columns")
                        self.data = data.copy()
            
            if self.data is None or self.data.empty:
                raise ValueError("No valid data received")
                
            self.data = self.data.dropna()
            
            self.data = self._validate_and_clean_data(self.data)
            
            print(f"Final data shape: {self.data.shape}")
            print(f"Final columns: {self.data.columns.tolist()}")
            
            return self.data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_returns(self, method='simple'):
        if self.data is None:
            raise ValueError("No data available. Run fetch_stock_data() first.")
        
        if method == 'simple':
            returns = self.data.pct_change().dropna()
        elif method == 'log':
            returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return returns
    
    def _validate_and_clean_data(self, data):
        cleaned_data = data.copy()
        
        for column in cleaned_data.columns:
            prices = cleaned_data[column]
            
            returns = prices.pct_change().dropna()
            
            return_std = returns.std()
            return_mean = returns.mean()
            outlier_threshold = 5 * return_std
            
            extreme_outliers = abs(returns - return_mean) > outlier_threshold
            
            if extreme_outliers.sum() > 0:
                print(f"Removing {extreme_outliers.sum()} extreme outliers from {column}")
                # Forward fill extreme outlier prices
                outlier_indices = extreme_outliers[extreme_outliers].index
                for idx in outlier_indices:
                    if idx in cleaned_data.index:
                        prev_idx = cleaned_data.index[cleaned_data.index < idx]
                        if len(prev_idx) > 0:
                            cleaned_data.loc[idx, column] = cleaned_data.loc[prev_idx[-1], column]
            
            invalid_prices = (prices <= 0) | prices.isna()
            if invalid_prices.sum() > 0:
                print(f"Fixing {invalid_prices.sum()} invalid prices in {column}")
                cleaned_data.loc[invalid_prices, column] = np.nan
                cleaned_data[column] = cleaned_data[column].ffill().bfill()
        
        if len(cleaned_data) < 252:
            print(f"Warning: Only {len(cleaned_data)} data points available (less than 1 year)")
        
        return cleaned_data
    
    def get_sample_portfolio(self):
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG']