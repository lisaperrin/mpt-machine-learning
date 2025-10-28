import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

class MLPortfolioOptimizer:
    def __init__(self, returns_data, price_data):
        """
        ML-Enhanced Portfolio Optimizer
        
        Args:
            returns_data (pd.DataFrame): Tägliche Renditen
            price_data (pd.DataFrame): Preisdaten
        """
        self.returns = returns_data
        self.prices = price_data
        self.features = None
        self.scaler = StandardScaler()
        
    def create_features(self, lookback_window=20):
        if self.features is not None:
            return self.features
            
        features_list = []
        
        for asset in self.prices.columns:
            price_series = self.prices[asset]
            return_series = self.returns[asset]
            
            asset_features = pd.DataFrame(index=price_series.index)
            
            asset_features[f'{asset}_ma_5'] = price_series.rolling(5).mean()
            asset_features[f'{asset}_ma_20'] = price_series.rolling(20).mean()
            asset_features[f'{asset}_ma_50'] = price_series.rolling(50).mean()
            
            asset_features[f'{asset}_vol_5'] = return_series.rolling(5).std()
            asset_features[f'{asset}_vol_20'] = return_series.rolling(20).std()
            
            asset_features[f'{asset}_momentum_5'] = price_series.pct_change(5)
            asset_features[f'{asset}_momentum_20'] = price_series.pct_change(20)
            
            gains = return_series.where(return_series > 0, 0).rolling(14).mean()
            losses = (-return_series.where(return_series < 0, 0)).rolling(14).mean()
            rs = gains / losses.replace(0, np.inf)
            asset_features[f'{asset}_rsi'] = 100 - (100 / (1 + rs))
            
            ma20 = asset_features[f'{asset}_ma_20']
            std20 = price_series.rolling(20).std()
            asset_features[f'{asset}_bb_upper'] = ma20 + (2 * std20)
            asset_features[f'{asset}_bb_lower'] = ma20 - (2 * std20)
            bb_width = asset_features[f'{asset}_bb_upper'] - asset_features[f'{asset}_bb_lower']
            asset_features[f'{asset}_bb_position'] = (price_series - asset_features[f'{asset}_bb_lower']) / bb_width
            
            features_list.append(asset_features)
        
        all_features = pd.concat(features_list, axis=1)
        
        market_return = self.returns.mean(axis=1)
        all_features['market_vol'] = market_return.rolling(20).std()
        all_features['market_momentum'] = market_return.rolling(10).mean()
        all_features['fear_index'] = self.returns.std(axis=1).rolling(20).mean()
        
        self.features = all_features.dropna()
        return self.features
    
    def predict_returns(self, test_size=0.2, random_state=42, prediction_horizon=5):
        if self.features is None:
            self.create_features()
        
        models = {}
        predictions = {}
        scalers = {}
        
        aligned_features = self.features.reindex(self.returns.index).dropna()
        aligned_returns = self.returns.reindex(aligned_features.index)
        for asset in self.returns.columns:
            print(f"Training model for {asset}...")
            
            target = aligned_returns[asset].shift(-prediction_horizon).dropna()
            features_subset = aligned_features.loc[target.index]
            
            split_idx = int(len(features_subset) * (1 - test_size))
            
            X_train = features_subset.iloc[:split_idx]
            X_test = features_subset.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            asset_scaler = StandardScaler()
            X_train_scaled = asset_scaler.fit_transform(X_train)
            X_test_scaled = asset_scaler.transform(X_test)
            scalers[asset] = asset_scaler
            
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            models[asset] = {
                'model': model,
                'scaler': asset_scaler,
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': dict(zip(features_subset.columns, model.feature_importances_)),
                'prediction_horizon': prediction_horizon
            }
            
            predictions[asset] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_actual': y_train.values,
                'test_actual': y_test.values,
                'train_indices': y_train.index,
                'test_indices': y_test.index
            }
        
        self.models = models
        self.predictions = predictions
        self.scalers = scalers
        return models, predictions
    
    def ml_portfolio_weights(self, risk_aversion=0.5, risk_free_rate=None):
        if not hasattr(self, 'models'):
            raise ValueError("Run predict_returns() first")
        
        latest_predictions = {}
        for asset in self.returns.columns:
            if self.features is not None and hasattr(self, 'scalers'):
                asset_scaler = self.scalers[asset]
                
                latest_features = self.features.iloc[-1:].values
                latest_features_scaled = asset_scaler.transform(latest_features)
                pred = self.models[asset]['model'].predict(latest_features_scaled)[0]
                latest_predictions[asset] = pred
        
        expected_returns = pd.Series(latest_predictions) * 252
        
        if risk_free_rate is None:
            risk_free_rate = 0.045
        
        # Apply confidence-based scaling to predictions
        confidence_scaling = {}
        for asset in self.returns.columns:
            test_score = self.models[asset]['test_score']
            # Scale down predictions for low-performing models
            confidence = max(0.1, test_score) if test_score > -0.5 else 0.1
            confidence_scaling[asset] = confidence
        
        # Adjust expected returns by model confidence
        for asset in expected_returns.index:
            expected_returns[asset] *= (1 + confidence_scaling.get(asset, 0.1))
        
        excess_returns = expected_returns - risk_free_rate
        
        cov_matrix = self.returns.cov() * 252
        
        inv_cov = np.linalg.pinv(cov_matrix.values)
        
        weights = inv_cov @ excess_returns.values / risk_aversion
        
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Apply asset class constraints to prevent over-allocation to bonds
        weights_series = pd.Series(weights, index=expected_returns.index)
        
        # Define asset classes
        bond_assets = [asset for asset in weights_series.index if asset in ['TLT', 'IEF', 'SHY', 'LQD', 'HYG']]
        commodity_assets = [asset for asset in weights_series.index if asset in ['GLD', 'SLV', 'USO']]
        
        # Limit bonds to max 40% and commodities to max 20%
        if bond_assets:
            bond_weight = weights_series[bond_assets].sum()
            if bond_weight > 0.4:
                scale_factor = 0.4 / bond_weight
                for asset in bond_assets:
                    weights_series[asset] *= scale_factor
        
        if commodity_assets:
            commodity_weight = weights_series[commodity_assets].sum()
            if commodity_weight > 0.2:
                scale_factor = 0.2 / commodity_weight
                for asset in commodity_assets:
                    weights_series[asset] *= scale_factor
        
        # Renormalize after constraints
        weights_series = weights_series / weights_series.sum()
        weights = weights_series.values
        
        return dict(zip(expected_returns.index, weights))
    
    def regime_based_optimization(self, n_regimes=3, min_regime_size=50):
        market_features = pd.DataFrame(index=self.returns.index)
        
        market_features['volatility'] = self.returns.std(axis=1).rolling(20).mean()
        
        rolling_corr_matrices = []
        for i in range(20, len(self.returns)):
            window_returns = self.returns.iloc[i-20:i]
            corr_matrix = window_returns.corr()
            upper_tri = np.triu(corr_matrix.values, k=1)
            avg_corr = upper_tri[upper_tri != 0].mean()
            rolling_corr_matrices.append(avg_corr)
        
        corr_index = self.returns.index[20:]
        market_features.loc[corr_index, 'avg_correlation'] = rolling_corr_matrices
        
        market_return = self.returns.mean(axis=1)
        market_features['momentum'] = market_return.rolling(20).mean()
        market_features['trend_strength'] = market_return.rolling(60).std()
        
        market_features['fear_index'] = self.returns.std(axis=1).rolling(10).mean()
        
        market_features = market_features.dropna()
        
        if len(market_features) < n_regimes * min_regime_size:
            print(f"Warning: Insufficient data for {n_regimes} regimes")
            n_regimes = max(2, len(market_features) // min_regime_size)
        
        feature_scaler = StandardScaler()
        feature_matrix = feature_scaler.fit_transform(market_features.values)
        
        best_kmeans = None
        best_inertia = float('inf')
        
        for _ in range(10):
            kmeans = KMeans(n_clusters=n_regimes, random_state=np.random.randint(0, 1000), n_init=10)
            kmeans.fit(feature_matrix)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
        
        regimes = best_kmeans.labels_
        
        regime_portfolios = {}
        regime_info = {}
        
        for regime in range(n_regimes):
            regime_mask = regimes == regime
            regime_indices = market_features.index[regime_mask]
            regime_returns = self.returns.loc[regime_indices]
            
            if len(regime_returns) >= min_regime_size:
                mean_ret = regime_returns.mean() * 252
                cov_mat = regime_returns.cov() * 252
                
                asset_vols = np.sqrt(np.diag(cov_mat))
                inv_vol_weights = (1 / asset_vols) / (1 / asset_vols).sum()
                
                regime_portfolios[f'regime_{regime}'] = dict(zip(mean_ret.index, inv_vol_weights))
                
                regime_info[f'regime_{regime}'] = {
                    'avg_volatility': market_features.loc[regime_indices, 'volatility'].mean(),
                    'avg_correlation': market_features.loc[regime_indices, 'avg_correlation'].mean(),
                    'avg_momentum': market_features.loc[regime_indices, 'momentum'].mean(),
                    'sample_size': len(regime_returns)
                }
            else:
                print(f"Regime {regime} has insufficient data ({len(regime_returns)} < {min_regime_size})")
        
        return regime_portfolios, {'model': best_kmeans, 'scaler': feature_scaler, 'regime_info': regime_info}
    
    def get_feature_importance(self, top_n=10):
        if not hasattr(self, 'models'):
            raise ValueError("Run predict_returns() first")
        
        importance_summary = {}
        
        for asset, model_info in self.models.items():
            sorted_features = sorted(
                model_info['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            importance_summary[asset] = {
                'top_features': sorted_features[:top_n],
                'model_score': model_info['test_score']
            }
        
        return importance_summary