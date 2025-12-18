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
        
        market_return = self.returns.mean(axis=1)
        
        market_features = pd.DataFrame(index=self.prices.index)
        market_features['market_mom_5'] = market_return.rolling(5).mean()
        market_features['market_mom_10'] = market_return.rolling(10).mean()
        market_features['market_vol_5'] = market_return.rolling(5).std()
        market_features['market_vol_20'] = market_return.rolling(20).std()
        market_features['vol_regime'] = (market_features['market_vol_5'] / market_features['market_vol_20']).fillna(1)
        
        market_features['dispersion'] = self.returns.std(axis=1)
        market_features['skew_proxy'] = self.returns.skew(axis=1).rolling(10).mean()
        
        pca_returns = self.returns.dropna()
        if len(pca_returns) > 60:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(pca_returns.fillna(0))
            
            for i in range(3):
                market_features.loc[pca_returns.index, f'pca_{i}'] = pca_features[:, i]
        
        sector_groups = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'ADBE', 'CRM', 'NFLX'],
            'financials': ['JPM', 'V'],
            'healthcare': ['JNJ', 'UNH'],
            'consumer': ['PG', 'HD', 'COST', 'DIS'],
            'bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
            'commodities': ['GLD', 'SLV', 'USO'],
            'international': ['VEA', 'VWO', 'VNQ']
        }
        
        for sector_name, tickers in sector_groups.items():
            sector_assets = [t for t in tickers if t in self.returns.columns]
            if len(sector_assets) > 0:
                sector_returns = self.returns[sector_assets].mean(axis=1)
                market_features[f'{sector_name}_mom'] = sector_returns.rolling(10).mean()
                market_features[f'{sector_name}_vol'] = sector_returns.rolling(20).std()
        
        key_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'TLT', 'GLD']
        available_assets = [a for a in key_assets if a in self.prices.columns]
        
        for asset in available_assets[:6]:
            price_series = self.prices[asset]
            return_series = self.returns[asset]
            
            asset_features = pd.DataFrame(index=price_series.index)
            
            asset_features[f'{asset}_mom_3'] = return_series.rolling(3).mean()
            asset_features[f'{asset}_mom_10'] = return_series.rolling(10).mean()
            asset_features[f'{asset}_mom_21'] = return_series.rolling(21).mean()
            
            asset_features[f'{asset}_vol_5'] = return_series.rolling(5).std()
            asset_features[f'{asset}_vol_21'] = return_series.rolling(21).std()
            
            asset_features[f'{asset}_vol_regime'] = (asset_features[f'{asset}_vol_5'] / 
                                                   asset_features[f'{asset}_vol_21']).fillna(1)
            
            sma_20 = price_series.rolling(20).mean()
            sma_50 = price_series.rolling(50).mean()
            asset_features[f'{asset}_trend'] = (sma_20 / sma_50 - 1).fillna(0)
            
            asset_features[f'{asset}_mean_revert'] = ((price_series - sma_20) / sma_20).fillna(0)
            
            high_low_range = price_series.rolling(20).max() - price_series.rolling(20).min()
            asset_features[f'{asset}_range_pos'] = ((price_series - price_series.rolling(20).min()) / 
                                                   high_low_range).fillna(0.5)
            
            returns_rank = return_series.rolling(63).rank(pct=True)
            asset_features[f'{asset}_momentum_rank'] = returns_rank.fillna(0.5)
            
            features_list.append(asset_features)
        
        all_features = pd.concat([market_features] + features_list, axis=1)
        
        feature_cols = []
        for col in all_features.columns:
            if not all_features[col].isna().all():
                feature_cols.append(col)
        
        all_features = all_features[feature_cols].fillna(method='ffill').fillna(0)
        
        self.features = all_features.dropna()
        print(f"Created {self.features.shape[1]} engineered features")
        return self.features
    
    def predict_returns(self, test_size=0.25, random_state=42, prediction_horizon=1):
        if self.features is None:
            self.create_features()
        
        models = {}
        predictions = {}
        
        print(f"Training ensemble models with {prediction_horizon}-day prediction horizon...")
        
        aligned_features = self.features.reindex(self.returns.index).dropna()
        aligned_returns = self.returns.reindex(aligned_features.index)
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.feature_selection import SelectKBest, f_regression
        import warnings
        warnings.filterwarnings('ignore')
        
        for asset in self.returns.columns:
            print(f"Training ensemble for {asset}...")
            
            target = aligned_returns[asset].shift(-prediction_horizon).dropna()
            feature_data = aligned_features.loc[target.index].copy()
            
            if len(target) < 120:
                print(f"Insufficient data for {asset}, skipping...")
                continue
            
            split_idx = int(len(feature_data) * (1 - test_size))
            
            X_train = feature_data.iloc[:split_idx]
            X_test = feature_data.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.fillna(0))
            X_test_scaled = scaler.transform(X_test.fillna(0))
            
            selector = SelectKBest(f_regression, k=min(20, X_train_scaled.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            models_to_try = {
                'ridge': Ridge(alpha=2.0, random_state=random_state),
                'elastic': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=random_state, max_iter=2000),
                'rf': RandomForestRegressor(
                    n_estimators=50,
                    max_depth=6,
                    min_samples_split=30,
                    min_samples_leaf=15,
                    max_features=0.4,
                    random_state=random_state,
                    n_jobs=-1
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=random_state
                )
            }
            
            model_scores = {}
            trained_models = {}
            
            for name, model in models_to_try.items():
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train_selected):
                    X_train_cv, X_val_cv = X_train_selected[train_idx], X_train_selected[val_idx]
                    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_train_cv, y_train_cv)
                    val_score = model.score(X_val_cv, y_val_cv)
                    cv_scores.append(val_score)
                
                model.fit(X_train_selected, y_train)
                trained_models[name] = model
                model_scores[name] = np.mean(cv_scores)
            
            best_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            ensemble_predictions = []
            ensemble_weights = []
            
            for model_name, score in best_models:
                if score > -0.5:
                    weight = max(0.1, score + 0.5)
                    ensemble_weights.append(weight)
                    pred = trained_models[model_name].predict(X_test_selected)
                    ensemble_predictions.append(pred)
            
            if len(ensemble_predictions) > 0:
                ensemble_weights = np.array(ensemble_weights)
                ensemble_weights = ensemble_weights / ensemble_weights.sum()
                
                final_prediction = np.zeros(len(y_test))
                for i, pred in enumerate(ensemble_predictions):
                    final_prediction += ensemble_weights[i] * pred
                
                test_score = 1 - np.sum((y_test - final_prediction) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
                
                train_pred = np.zeros(len(y_train))
                for i, (model_name, _) in enumerate(best_models[:len(ensemble_predictions)]):
                    train_pred += ensemble_weights[i] * trained_models[model_name].predict(X_train_selected)
                
                train_score = 1 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
                
                cv_score = np.mean([score for _, score in best_models[:len(ensemble_predictions)]])
                
                models[asset] = {
                    'ensemble_models': {name: trained_models[name] for name, _ in best_models[:len(ensemble_predictions)]},
                    'ensemble_weights': ensemble_weights,
                    'scaler': scaler,
                    'selector': selector,
                    'model_type': 'Ensemble',
                    'train_score': train_score,
                    'test_score': test_score,
                    'cv_score': cv_score,
                    'prediction_horizon': prediction_horizon,
                    'best_models': [name for name, _ in best_models[:len(ensemble_predictions)]]
                }
                
                predictions[asset] = {
                    'train_pred': train_pred,
                    'test_pred': final_prediction,
                    'train_actual': y_train.values,
                    'test_actual': y_test.values
                }
            else:
                print(f"No valid models for {asset}")
        
        self.models = models
        self.predictions = predictions
        
        valid_scores = [info['test_score'] for info in models.values() if info['test_score'] > -1]
        avg_score = np.mean(valid_scores) if valid_scores else -999
        print(f"Average test R² score: {avg_score:.3f} ({len(valid_scores)} ensemble models)")
        
        positive_scores = [score for score in valid_scores if score > 0]
        if positive_scores:
            print(f"Positive R² models: {len(positive_scores)}/{len(valid_scores)} (avg: {np.mean(positive_scores):.3f})")
        
        return models, predictions
    
    def ml_portfolio_weights(self, risk_aversion=1.0, risk_free_rate=0.045, min_weight=0.01):
        if not hasattr(self, 'models'):
            raise ValueError("Run predict_returns() first")
        
        latest_predictions = {}
        model_confidences = {}
        
        for asset in self.models.keys():
            model_info = self.models[asset]
            scaler = model_info['scaler']
            selector = model_info['selector']
            
            latest_features = self.features.iloc[-1:].fillna(0).values
            latest_features_scaled = scaler.transform(latest_features)
            latest_features_selected = selector.transform(latest_features_scaled)
            
            if 'ensemble_models' in model_info:
                ensemble_pred = 0
                for i, (model_name, model) in enumerate(model_info['ensemble_models'].items()):
                    model_pred = model.predict(latest_features_selected)[0]
                    ensemble_pred += model_info['ensemble_weights'][i] * model_pred
                pred = ensemble_pred
            else:
                pred = model_info['model'].predict(latest_features_selected)[0]
            
            test_score = model_info['test_score']
            cv_score = model_info['cv_score']
            
            confidence = max(0.0, min(test_score, cv_score)) if test_score > -0.5 else 0.0
            confidence = max(0.2, confidence + 0.3)
            
            latest_predictions[asset] = pred
            model_confidences[asset] = confidence
        
        expected_returns = pd.Series(latest_predictions)
        
        adjusted_returns = {}
        for asset in expected_returns.index:
            base_return = expected_returns[asset]
            confidence = model_confidences[asset]
            
            historical_mean = self.returns[asset].mean()
            
            blended_return = confidence * base_return + (1 - confidence) * historical_mean
            adjusted_returns[asset] = blended_return
        
        expected_returns = pd.Series(adjusted_returns) * 252
        
        from scipy.optimize import minimize
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.returns.cov().values * 252, weights))
        
        def portfolio_return(weights):
            return np.sum(expected_returns.values * weights)
        
        def negative_utility(weights):
            ret = portfolio_return(weights)
            var = portfolio_variance(weights)
            return -(ret - 0.5 * risk_aversion * var)
        
        n_assets = len(expected_returns)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((min_weight, 0.25) for _ in range(n_assets))
        
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(negative_utility, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
            else:
                print("Optimization failed, using equal weights")
                weights = np.ones(n_assets) / n_assets
        except:
            print("Optimization error, using equal weights")
            weights = np.ones(n_assets) / n_assets
        
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