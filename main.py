import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_collector import DataCollector
from portfolio_optimizer import ModernPortfolioTheory, PyPfOptimizer
from ml_portfolio_optimizer import MLPortfolioOptimizer

def main():
    print("Modern Portfolio Theory with Machine Learning")
    print("=" * 50)
    
    print("\nLoading market data...")
    collector = DataCollector()

    tickers = [
        # Large Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        # Value & Dividend Stocks  
        'JPM', 'JNJ', 'V', 'PG', 'HD', 'UNH',
        # Growth Stocks
        'DIS', 'ADBE', 'CRM', 'NFLX', 'COST',
        # Treasury Bonds
        'TLT',   # 20+ Year Treasury Bond ETF
        'IEF',   # 7-10 Year Treasury Bond ETF  
        'SHY',   # 1-3 Year Treasury Bond ETF
        # Corporate Bonds
        'LQD',   # Investment Grade Corporate Bond ETF
        'HYG',   # High Yield Corporate Bond ETF
        # Commodities & Alternative Assets
        'GLD',   # Gold ETF
        'SLV',   # Silver ETF
        'USO',   # Oil ETF
        # International Exposure
        'VEA',   # Developed Markets ETF
        'VWO',   # Emerging Markets ETF
        # REITs
        'VNQ'    # Real Estate Investment Trust ETF
    ]
    
    try:
        price_data = collector.fetch_stock_data(tickers, period='3y')
        returns_data = collector.calculate_returns()
        
        print(f"Data loaded: {len(tickers)} assets, {len(price_data)} days")
        print(f"   Period: {price_data.index[0].date()} to {price_data.index[-1].date()}")
        
        # Quick diagnostic: Asset class performance
        stock_assets = [t for t in tickers if t not in ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'GLD', 'SLV', 'USO', 'VEA', 'VWO', 'VNQ']]
        bond_assets = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG']
        
        available_stocks = [t for t in stock_assets if t in returns_data.columns]
        available_bonds = [t for t in bond_assets if t in returns_data.columns]
        
        if available_stocks:
            stock_returns = returns_data[available_stocks].mean(axis=1)
            stock_performance = (stock_returns.mean() * 252, stock_returns.std() * np.sqrt(252))
            print(f"   Stocks avg: {stock_performance[0]:.1%} return, {stock_performance[1]:.1%} vol")
        
        if available_bonds:
            bond_returns = returns_data[available_bonds].mean(axis=1)  
            bond_performance = (bond_returns.mean() * 252, bond_returns.std() * np.sqrt(252))
            print(f"   Bonds avg: {bond_performance[0]:.1%} return, {bond_performance[1]:.1%} vol")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\nClassic Modern Portfolio Theory...")
    mpt = ModernPortfolioTheory(returns_data)
    
    try:
        max_sharpe = mpt.max_sharpe_portfolio()
        min_vol = mpt.min_volatility_portfolio()
        
        print(f"   Max Sharpe: {max_sharpe['return']:.1%} return, {max_sharpe['volatility']:.1%} volatility")
        print(f"   Min Volatility: {min_vol['return']:.1%} return, {min_vol['volatility']:.1%} volatility")
        
    except Exception as e:
        print(f"Error in MPT optimization: {e}")
        return
    
    print("\nMachine Learning Portfolio Optimization...")
    ml_optimizer = MLPortfolioOptimizer(returns_data, price_data)
    
    try:
        features = ml_optimizer.create_features()
        print(f"   {features.shape[1]} features created")
        
        models, predictions = ml_optimizer.predict_returns()
        
        avg_score = np.mean([info['test_score'] for info in models.values()])
        print(f"   Average R² score: {avg_score:.3f}")
        
        ml_weights = ml_optimizer.ml_portfolio_weights()
        
        significant_ml = {k: v for k, v in ml_weights.items() if v > 0.05}
        print(f"   ML portfolio top positions: {significant_ml}")
        
    except Exception as e:
        print(f"Error in ML optimization: {e}")
        ml_weights = None
    
    print("\nRegime-based portfolio allocation...")
    try:
        regime_portfolios, regime_model = ml_optimizer.regime_based_optimization(n_regimes=3)
        print(f"   {len(regime_portfolios)} market regimes identified")
        
        for regime_name, weights in regime_portfolios.items():
            significant_regime = {k: v for k, v in weights.items() if v > 0.1}
            print(f"   {regime_name}: {significant_regime}")
            
    except Exception as e:
        print(f"Error in regime analysis: {e}")
    
    print("\nPortfolio performance comparison...")
    
    def calculate_portfolio_metrics(weights, returns_data):
        weights_series = pd.Series(weights)
        weights_series = weights_series / weights_series.sum()
        
        portfolio_return = (returns_data * weights_series).sum(axis=1)
        annual_return = portfolio_return.mean() * 252
        annual_vol = portfolio_return.std() * np.sqrt(252)
        risk_free_rate = 0.045
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        return annual_return, annual_vol, sharpe
    
    # Add a simple "Stock-Heavy" strategy for comparison
    stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'HD', 'UNH', 'DIS', 'ADBE', 'CRM', 'NFLX', 'COST']
    available_stocks = [t for t in stock_tickers if t in returns_data.columns]
    
    strategies = {
        'Equal Weight (All)': {asset: 1/len(tickers) for asset in tickers},
        'Equal Weight (Stocks)': {asset: 1/len(available_stocks) if asset in available_stocks else 0 for asset in tickers},
        'Max Sharpe (MPT)': max_sharpe['weights'],
        'Min Volatility (MPT)': min_vol['weights']
    }
    
    if ml_weights:
        strategies['ML-Enhanced'] = ml_weights
    
    print("\n" + "="*60)
    print(f"{'Strategy':<20} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
    print("-"*60)
    
    for name, weights in strategies.items():
        try:
            ret, vol, sharpe = calculate_portfolio_metrics(weights, returns_data)
            print(f"{name:<20} {ret:>8.1%} {vol:>10.1%} {sharpe:>10.3f}")
        except Exception as e:
            print(f"{name:<20} Calculation error")
    
    if hasattr(ml_optimizer, 'models'):
        print("\nTop features for ML models:")
        try:
            importance = ml_optimizer.get_feature_importance(top_n=3)
            for asset in list(importance.keys())[:3]:
                print(f"\n   {asset} (R²: {importance[asset]['model_score']:.3f}):")
                for feature, imp in importance[asset]['top_features']:
                    print(f"     - {feature}: {imp:.3f}")
        except Exception as e:
            print(f"   Error in feature importance: {e}")
    
    print("\n" + "="*50)
    print("Analysis completed!")
    print("\nNext steps:")
    print("Open notebook: jupyter notebook notebooks/mpt_example.ipynb")
if __name__ == "__main__":
    main()