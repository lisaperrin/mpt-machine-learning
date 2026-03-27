import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLPortfolioOptimizer:
    def __init__(self, 
                 returns_data: pd.DataFrame, 
                 price_data: pd.DataFrame,
                 config: Dict,
                 model_dir: str = "models"):
        
        self.returns = returns_data
        self.prices = price_data
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.features = None
        self.models = {}
        self.predictions = {}
        
    def create_multi_timeframe_features(self) -> Dict[str, pd.DataFrame]:
        """Create features for different prediction timeframes"""
        if hasattr(self, 'timeframe_features'):
            return self.timeframe_features
            
        logger.info("Creating multi-timeframe features...")

        timeframes = {
            'short': {'horizon': 5, 'focus': 'mean_reversion_momentum'},
            'medium': {'horizon': 63, 'focus': 'trend_sector_rotation'}, 
            'long': {'horizon': 252, 'focus': 'fundamental_cycles'}
        }
        
        self.timeframe_features = {}
        market_return = self.returns.mean(axis=1)
        
        for timeframe, config in timeframes.items():
            features = pd.DataFrame(index=self.prices.index)
            horizon = config['horizon']

            features[f'market_mom_{horizon}'] = market_return.rolling(horizon).mean()
            features[f'market_vol_{horizon}'] = market_return.rolling(horizon).std()
            features[f'market_sharpe_{horizon}'] = features[f'market_mom_{horizon}'] / features[f'market_vol_{horizon}']

            features['vol_regime'] = (market_return.rolling(5).std() / 
                                    market_return.rolling(21).std()).fillna(1)
            features['trend_strength'] = abs(market_return.rolling(horizon//4).mean())
            
            if timeframe == 'short':
                features['rsi_proxy'] = self._calculate_rsi_proxy(market_return, 14)
                features['bollinger_position'] = self._calculate_bollinger_position(market_return, 20)
                features['momentum_5d'] = market_return.rolling(5).mean()
                features['mean_reversion_signal'] = -features['momentum_5d']  # Contrarian
                
            elif timeframe == 'medium':
                features = self._add_sector_rotation_features(features, horizon)
                features['trend_momentum'] = market_return.rolling(horizon//2).mean()
                features['vol_adjusted_return'] = features['trend_momentum'] / features[f'market_vol_{horizon}']
                
            elif timeframe == 'long':
                features['long_trend'] = market_return.rolling(horizon).mean()
                features['cycle_position'] = self._estimate_cycle_position(market_return, horizon)
                features['risk_appetite'] = self._calculate_risk_appetite(horizon)
            
            self.timeframe_features[timeframe] = features.fillna(0)
            
        return self.timeframe_features

    def _calculate_rsi_proxy(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Simple RSI-like momentum indicator"""
        gains = returns.where(returns > 0, 0).rolling(window).mean()
        losses = (-returns.where(returns < 0, 0)).rolling(window).mean()
        rs = gains / (losses + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_position(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Position relative to Bollinger Bands"""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        current_return = returns
        return (current_return - rolling_mean) / (2 * rolling_std + 1e-8)
    
    def _add_sector_rotation_features(self, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Add sector rotation indicators"""
        sector_groups = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'financials': ['JPM', 'V', 'MA', 'BAC'],
            'healthcare': ['JNJ', 'UNH', 'PFE'],
            'defensive': ['TLT', 'GLD', 'VNQ']
        }
        
        for sector_name, tickers in sector_groups.items():
            sector_assets = [t for t in tickers if t in self.returns.columns]
            if len(sector_assets) > 0:
                sector_returns = self.returns[sector_assets].mean(axis=1)
                features[f'{sector_name}_momentum'] = sector_returns.rolling(horizon//2).mean()
                features[f'{sector_name}_relative_strength'] = (
                    sector_returns.rolling(horizon//4).mean() - 
                    self.returns.mean(axis=1).rolling(horizon//4).mean()
                )
        
        return features
    
    def _estimate_cycle_position(self, market_returns: pd.Series, horizon: int) -> pd.Series:
        """Estimate economic cycle position"""
        long_ma = market_returns.rolling(horizon).mean()
        medium_ma = market_returns.rolling(horizon//2).mean()
        short_ma = market_returns.rolling(horizon//4).mean()
        
        # Trend alignment score
        trend_score = ((short_ma > medium_ma).astype(int) + 
                      (medium_ma > long_ma).astype(int) - 1)
        return trend_score
    
    def _calculate_risk_appetite(self, horizon: int) -> pd.Series:
        """Calculate market risk appetite indicator"""
        if 'VIX' in self.returns.columns:
            # Use VIX if available
            return -self.returns['VIX'].rolling(horizon//4).mean()
        else:
            # Use correlation as proxy for risk appetite
            rolling_corr = self.returns.rolling(horizon//4).corr().mean(axis=1).groupby(level=0).mean()
            return -rolling_corr  # High correlation = low risk appetite

    def create_features(self) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        if self.features is not None:
            return self.features
            
        logger.info("Creating features...")
        
        features_list = []
        market_return = self.returns.mean(axis=1)
        
        market_features = pd.DataFrame(index=self.prices.index)
        lookback_periods = self.config.get('features', {}).get('lookback_periods', [5, 10, 20])
        
        for period in lookback_periods:
            market_features[f'market_mom_{period}'] = market_return.rolling(period).mean()
            market_features[f'market_vol_{period}'] = market_return.rolling(period).std()
        
        market_features['vol_regime'] = (market_features['market_vol_5'] / 
                                       market_features['market_vol_20']).fillna(1)
        market_features['dispersion'] = self.returns.std(axis=1)
        market_features['skew_proxy'] = self.returns.skew(axis=1).rolling(10).mean()
        
        pca_returns = self.returns.dropna()
        if len(pca_returns) > 60:
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(pca_returns.fillna(0))
            for i in range(3):
                market_features.loc[pca_returns.index, f'pca_{i}'] = pca_features[:, i]
        
        sector_groups = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'financials': ['JPM', 'V', 'MA', 'BAC'],
            'healthcare': ['JNJ', 'UNH', 'PFE'],
            'bonds': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
            'commodities': ['GLD', 'SLV', 'DBA', 'USO']
        }
        
        for sector_name, tickers in sector_groups.items():
            sector_assets = [t for t in tickers if t in self.returns.columns]
            if len(sector_assets) > 0:
                sector_returns = self.returns[sector_assets].mean(axis=1)
                market_features[f'{sector_name}_mom'] = sector_returns.rolling(10).mean()
                market_features[f'{sector_name}_vol'] = sector_returns.rolling(20).std()
        
        key_assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM']
        available_assets = [a for a in key_assets if a in self.prices.columns][:6]
        
        momentum_windows = self.config.get('features', {}).get('momentum_windows', [21, 63])
        
        for asset in available_assets:
            price_series = self.prices[asset]
            return_series = self.returns[asset]
            
            asset_features = pd.DataFrame(index=price_series.index)
            
            for window in momentum_windows[:2]:
                asset_features[f'{asset}_mom_{window}'] = return_series.rolling(window).mean()
            
            asset_features[f'{asset}_vol_5'] = return_series.rolling(5).std()
            asset_features[f'{asset}_vol_21'] = return_series.rolling(21).std()
            
            asset_features[f'{asset}_vol_regime'] = (asset_features[f'{asset}_vol_5'] / 
                                                   asset_features[f'{asset}_vol_21']).fillna(1)
            
            sma_20 = price_series.rolling(20).mean()
            sma_50 = price_series.rolling(50).mean()
            asset_features[f'{asset}_trend'] = (sma_20 / sma_50 - 1).fillna(0)
            
            asset_features[f'{asset}_mean_revert'] = ((price_series - sma_20) / sma_20).fillna(0)
            
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
        
        logger.info(f"Created {self.features.shape[1]} features")
        return self.features
    
    def train_models(self, save_models: bool = True) -> Tuple[Dict, Dict]:
        if self.features is None:
            self.create_features()
        
        test_size = self.config.get('test_size', 0.25)
        prediction_horizon = self.config.get('prediction_horizon', 1)
        model_params = self.config.get('model_params', {})
        
        logger.info(f"Training models with {prediction_horizon}-day horizon")
        
        aligned_features = self.features.reindex(self.returns.index).dropna()
        aligned_returns = self.returns.reindex(aligned_features.index)
        
        for asset in self.returns.columns:
            logger.info(f"Training ensemble for {asset}")
            
            target = aligned_returns[asset].shift(-prediction_horizon).dropna()
            feature_data = aligned_features.loc[target.index].copy()
            
            if len(target) < 120:
                logger.warning(f"Insufficient data for {asset}")
                continue
            
            split_idx = int(len(feature_data) * (1 - test_size))
            
            X_train = feature_data.iloc[:split_idx]
            X_test = feature_data.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.fillna(0))
            X_test_scaled = scaler.transform(X_test.fillna(0))
            
            k_best = min(20, X_train_scaled.shape[1])
            selector = SelectKBest(f_regression, k=k_best)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            tscv = TimeSeriesSplit(n_splits=model_params.get('cv_folds', 3))
            
            models_to_try = {
                'ridge': Ridge(
                    alpha=model_params.get('regularization_alpha', 2.0)
                ),
                'elastic': ElasticNet(
                    alpha=0.5, 
                    l1_ratio=0.5, 
                    max_iter=2000
                ),
                'rf': RandomForestRegressor(
                    n_estimators=model_params.get('n_estimators', 50),
                    max_depth=model_params.get('max_depth', 6),
                    min_samples_split=model_params.get('min_samples_split', 30),
                    min_samples_leaf=15,
                    max_features=0.4,
                    random_state=42,
                    n_jobs=-1
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42
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
                cv_score = np.mean([score for _, score in best_models[:len(ensemble_predictions)]])
                
                self.models[asset] = {
                    'ensemble_models': {name: trained_models[name] for name, _ in best_models[:len(ensemble_predictions)]},
                    'ensemble_weights': ensemble_weights,
                    'scaler': scaler,
                    'selector': selector,
                    'test_score': test_score,
                    'cv_score': cv_score,
                    'best_models': [name for name, _ in best_models[:len(ensemble_predictions)]]
                }
                
                if save_models:
                    model_path = self.model_dir / f"{asset}_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.models[asset], f)
        
        valid_scores = [info['test_score'] for info in self.models.values() if info['test_score'] > -1]
        avg_score = np.mean(valid_scores) if valid_scores else -999
        positive_scores = [score for score in valid_scores if score > 0]
        
        return self.models, {'avg_score': avg_score, 'valid_scores': valid_scores}
    
    def train_multi_timeframe_models(self, save_models: bool = True) -> Dict[str, Dict]:
        """Train separate models for each timeframe"""
        logger.info("Training multi-timeframe models...")
        
        timeframe_features = self.create_multi_timeframe_features()
        timeframe_models = {}

        timeframe_targets = {
            'short': {'horizon': 5, 'target_type': 'mean_reversion'},
            'medium': {'horizon': 21, 'target_type': 'risk_adjusted_return'},
            'long': {'horizon': 63, 'target_type': 'relative_strength'}
        }
        
        for timeframe, target_config in timeframe_targets.items():
            logger.info(f"Training {timeframe}-term models...")
            
            features = timeframe_features[timeframe]
            horizon = target_config['horizon']
            target_type = target_config['target_type']
            
            timeframe_models[timeframe] = {}
            
            for asset in self.returns.columns:
                logger.info(f"Training {timeframe}-term model for {asset}")
                
                # Create different targets based on timeframe
                if target_type == 'mean_reversion':
                    # Short-term: predict mean reversion (negative of recent momentum)
                    target = -self.returns[asset].rolling(horizon).mean().shift(-horizon)
                elif target_type == 'risk_adjusted_return':
                    # Medium-term: predict risk-adjusted returns
                    ret = self.returns[asset].rolling(horizon).mean()
                    vol = self.returns[asset].rolling(horizon).std()
                    target = (ret / (vol + 1e-8)).shift(-horizon)
                elif target_type == 'relative_strength':
                    # Long-term: predict relative performance vs market
                    asset_ret = self.returns[asset].rolling(horizon).mean()
                    market_ret = self.returns.mean(axis=1).rolling(horizon).mean()
                    target = (asset_ret - market_ret).shift(-horizon)
                
                target = target.dropna()
                feature_data = features.loc[target.index].copy()
                
                if len(target) < 100:
                    logger.warning(f"Insufficient data for {asset} {timeframe}-term model")
                    continue
                
                # Split data
                split_idx = int(len(feature_data) * 0.75)
                X_train = feature_data.iloc[:split_idx]
                X_test = feature_data.iloc[split_idx:]
                y_train = target.iloc[:split_idx]
                y_test = target.iloc[split_idx:]
                
                # Scale and select features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.fillna(0))
                X_test_scaled = scaler.transform(X_test.fillna(0))
                
                selector = SelectKBest(f_regression, k=min(15, X_train_scaled.shape[1]))
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)
                
                # Train ensemble for this timeframe
                models = {
                    'ridge': Ridge(alpha=1.0),
                    'rf': RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42),
                    'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
                }
                
                trained_models = {}
                scores = []
                
                for name, model in models.items():
                    try:
                        model.fit(X_train_selected, y_train)
                        pred = model.predict(X_test_selected)
                        score = np.corrcoef(pred, y_test)[0, 1] if len(set(y_test)) > 1 else 0
                        
                        trained_models[name] = model
                        scores.append(max(0, score))
                        
                    except Exception as e:
                        logger.warning(f"Model {name} failed for {asset}: {e}")
                        scores.append(0)
                
                # Store model info
                total_score = sum(scores)
                if total_score > 0:
                    weights = [s/total_score for s in scores]
                else:
                    weights = [1/len(scores)] * len(scores)
                
                timeframe_models[timeframe][asset] = {
                    'models': trained_models,
                    'weights': weights,
                    'scaler': scaler,
                    'selector': selector,
                    'test_score': np.mean(scores),
                    'horizon': horizon,
                    'target_type': target_type
                }
                
        self.timeframe_models = timeframe_models
        logger.info(f"Completed multi-timeframe model training")
        
        return timeframe_models

    def predict_returns(self, test_size: float = 0.25, random_state: int = 42, prediction_horizon: int = 1) -> Dict:
        """Enhanced prediction with detailed performance metrics"""
        if self.features is None:
            self.create_features()
        
        logger.info("Running enhanced prediction with performance tracking")
        
        aligned_features = self.features.reindex(self.returns.index).dropna()
        aligned_returns = self.returns.reindex(aligned_features.index)
        
        # Performance tracking variables
        model_performance = {}
        feature_importance_agg = {}
        r2_scores = {}
        cv_scores = []
        
        for asset in aligned_returns.columns[:10]:  # Limit for performance
            target = aligned_returns[asset].shift(-prediction_horizon).dropna()
            feature_data = aligned_features.loc[target.index].copy()
            
            if len(target) < 120:
                continue
            
            split_idx = int(len(feature_data) * (1 - test_size))
            
            X_train = feature_data.iloc[:split_idx]
            X_test = feature_data.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]
            
            # Preprocessing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.fillna(0))
            X_test_scaled = scaler.transform(X_test.fillna(0))
            
            # Feature selection
            k_best = min(15, X_train_scaled.shape[1])
            selector = SelectKBest(f_regression, k=k_best)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Model training and evaluation
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42),
                'Ridge': Ridge(alpha=2.0),
                'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=1000)
            }
            
            asset_scores = {}
            for name, model in models.items():
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores_model = []
                
                for train_idx, val_idx in tscv.split(X_train_selected):
                    X_train_cv, X_val_cv = X_train_selected[train_idx], X_train_selected[val_idx]
                    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_train_cv, y_train_cv)
                    val_score = model.score(X_val_cv, y_val_cv)
                    cv_scores_model.append(val_score)
                
                # Final training and test score
                model.fit(X_train_selected, y_train)
                test_score = model.score(X_test_selected, y_test)
                
                asset_scores[name] = {
                    'test_score': test_score,
                    'cv_score': np.mean(cv_scores_model),
                    'cv_scores': cv_scores_model
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    selected_features = selector.get_support()
                    feature_names = [feature_data.columns[i] for i, selected in enumerate(selected_features) if selected]
                    
                    for i, importance in enumerate(model.feature_importances_):
                        feat_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
                        if feat_name not in feature_importance_agg:
                            feature_importance_agg[feat_name] = []
                        feature_importance_agg[feat_name].append(importance)
            
            model_performance[asset] = asset_scores
            
            # Aggregate R² scores
            for model_name, scores in asset_scores.items():
                if model_name not in r2_scores:
                    r2_scores[model_name] = []
                r2_scores[model_name].append(scores['test_score'])
                cv_scores.extend(scores['cv_scores'])
        
        # Aggregate feature importance
        avg_feature_importance = {}
        for feat, importances in feature_importance_agg.items():
            avg_feature_importance[feat] = np.mean(importances)
        
        # Sort and take top features
        sorted_features = dict(sorted(avg_feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10])
        
        # Aggregate model scores
        avg_r2_scores = {model: np.mean(scores) for model, scores in r2_scores.items()}
        
        # Best model selection
        best_model = max(avg_r2_scores, key=avg_r2_scores.get) if avg_r2_scores else 'Ridge'
        
        return {
            'r2_scores': avg_r2_scores,
            'feature_importance': sorted_features,
            'model_comparison': {
                'best_model': best_model,
                'best_score': avg_r2_scores.get(best_model, -999)
            },
            'cv_scores': cv_scores,
            'detailed_performance': model_performance
        }
    
    def load_models(self) -> bool:
        try:
            for asset in self.returns.columns:
                model_path = self.model_dir / f"{asset}_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[asset] = pickle.load(f)
            return len(self.models) > 0
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_portfolio_weights(self, 
                            risk_aversion: float = 1.0, 
                            constraints: Optional[Dict] = None) -> Dict[str, float]:
        
        if not self.models:
            raise ValueError("No trained models available")
        
        if constraints is None:
            constraints = self.config.get('portfolio_constraints', {})
        
        latest_predictions = {}
        model_confidences = {}
        
        for asset in self.models.keys():
            model_info = self.models[asset]
            scaler = model_info['scaler']
            selector = model_info['selector']
            
            latest_features = self.features.iloc[-1:].fillna(0).values
            latest_features_scaled = scaler.transform(latest_features)
            latest_features_selected = selector.transform(latest_features_scaled)
            
            ensemble_pred = 0
            for i, (model_name, model) in enumerate(model_info['ensemble_models'].items()):
                model_pred = model.predict(latest_features_selected)[0]
                ensemble_pred += model_info['ensemble_weights'][i] * model_pred
            
            test_score = model_info['test_score']
            cv_score = model_info['cv_score']
            
            confidence = max(0.0, min(test_score, cv_score)) if test_score > -0.5 else 0.0
            confidence = max(0.2, confidence + 0.3)
            
            latest_predictions[asset] = ensemble_pred
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
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.returns.cov().values * 252, weights))
        
        def portfolio_return(weights):
            return np.sum(expected_returns.values * weights)
        
        def negative_utility(weights):
            ret = portfolio_return(weights)
            var = portfolio_variance(weights)
            return -(ret - 0.5 * risk_aversion * var)
        
        n_assets = len(expected_returns)
        constraint_funcs = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        min_weight = constraints.get('min_weight', 0.01)
        max_weight = constraints.get('max_weight', 0.25)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        initial_weights = np.ones(n_assets) / n_assets
        
        logger.info(f"Starting ML optimization with bounds: min={min_weight:.3f}, max={max_weight:.3f}")
        logger.info(f"Expected returns range: {expected_returns.min():.4f} to {expected_returns.max():.4f}")
        
        try:
            result = minimize(negative_utility, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraint_funcs)
            
            if result.success:
                weights = result.x
                logger.info(f"ML optimization successful. Max weight: {max(weights):.3f}, Min weight: {min(weights):.3f}")
            else:
                logger.warning(f"ML optimization failed: {result.message}")
                # Try with slightly relaxed but still reasonable constraints
                try:
                    relaxed_min = max(0.0, min_weight - 0.005)  # Slightly lower min
                    relaxed_max = min(0.5, max_weight + 0.1)   # Slightly higher max but still reasonable
                    relaxed_bounds = tuple((relaxed_min, relaxed_max) for _ in range(n_assets))
                    result2 = minimize(negative_utility, initial_weights,
                                     method='SLSQP', bounds=relaxed_bounds, constraints=constraint_funcs)
                    if result2.success:
                        weights = result2.x
                        logger.info(f"ML optimization successful with relaxed constraints. Max weight: {max(weights):.3f}")
                    else:
                        logger.warning("ML optimization failed even with relaxed constraints, using equal weights")
                        weights = np.ones(n_assets) / n_assets
                except Exception:
                    logger.warning("ML optimization retry failed, using equal weights")
                    weights = np.ones(n_assets) / n_assets
        except Exception as e:
            logger.error(f"ML optimization error: {e}")
            weights = np.ones(n_assets) / n_assets
        
        return dict(zip(expected_returns.index, weights))
    
    def get_multi_timeframe_portfolio_weights(self, 
                                            risk_aversion: float = 1.0,
                                            constraints: Optional[Dict] = None) -> Dict[str, float]:
        """Enhanced portfolio optimization using multi-timeframe ML signals"""
        
        if not hasattr(self, 'timeframe_models'):
            logger.info("Training multi-timeframe models...")
            self.train_multi_timeframe_models()
        
        if constraints is None:
            constraints = self.config.get('portfolio_constraints', {})
        
        logger.info("Generating multi-timeframe portfolio weights...")
        
        # Get current market regime to weight timeframes
        market_returns = self.returns.mean(axis=1)
        current_vol = market_returns.rolling(21).std().iloc[-1]
        long_term_vol = market_returns.rolling(252).std().iloc[-1]
        vol_regime = current_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Adaptive timeframe weighting based on market conditions
        if vol_regime > 1.5:  # High volatility - focus on short-term
            timeframe_weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
            logger.info("High volatility regime - emphasizing short-term signals")
        elif vol_regime < 0.7:  # Low volatility - focus on medium/long-term
            timeframe_weights = {'short': 0.2, 'medium': 0.4, 'long': 0.4}
            logger.info("Low volatility regime - emphasizing longer-term signals")
        else:  # Normal regime - balanced
            timeframe_weights = {'short': 0.3, 'medium': 0.4, 'long': 0.3}
            logger.info("Normal volatility regime - balanced timeframe weighting")
        
        # Generate predictions for each timeframe
        combined_signals = {}
        signal_confidence = {}
        
        for timeframe, weight in timeframe_weights.items():
            if timeframe not in self.timeframe_models:
                continue
                
            timeframe_features = self.timeframe_features[timeframe]
            latest_features = timeframe_features.iloc[-1:].fillna(0).values
            
            for asset, model_info in self.timeframe_models[timeframe].items():
                if asset not in combined_signals:
                    combined_signals[asset] = 0
                    signal_confidence[asset] = 0
                
                # Get prediction from ensemble
                scaler = model_info['scaler']
                selector = model_info['selector']
                models = model_info['models']
                model_weights = model_info['weights']
                test_score = model_info['test_score']
                
                try:
                    features_scaled = scaler.transform(latest_features)
                    features_selected = selector.transform(features_scaled)
                    
                    ensemble_prediction = 0
                    for i, (name, model) in enumerate(models.items()):
                        pred = model.predict(features_selected)[0]
                        ensemble_prediction += model_weights[i] * pred
                    
                    # Weight by timeframe importance and model confidence
                    signal_strength = weight * max(0, test_score)
                    combined_signals[asset] += signal_strength * ensemble_prediction
                    signal_confidence[asset] += signal_strength
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {asset} in {timeframe}: {e}")
                    continue
        
        # Normalize signals by confidence
        for asset in combined_signals:
            if signal_confidence[asset] > 0:
                combined_signals[asset] /= signal_confidence[asset]
            else:
                combined_signals[asset] = 0
        
        # Convert to expected returns and blend with historical means
        expected_returns = pd.Series(combined_signals)
        
        # Adaptive blending based on overall signal confidence
        avg_confidence = np.mean(list(signal_confidence.values()))
        confidence_weight = min(0.7, max(0.2, avg_confidence))  # 20-70% ML weight
        
        logger.info(f"ML signal confidence: {avg_confidence:.3f}, using {confidence_weight:.1%} ML weight")
        
        blended_returns = {}
        for asset in expected_returns.index:
            ml_signal = expected_returns[asset]
            historical_mean = self.returns[asset].mean()
            
            # Blend ML signal with historical mean based on confidence
            blended_return = confidence_weight * ml_signal + (1 - confidence_weight) * historical_mean
            blended_returns[asset] = blended_return
        
        expected_returns = pd.Series(blended_returns) * 252  # Annualize
        
        # Portfolio optimization with enhanced constraints
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.returns.cov().values * 252, weights))
        
        def portfolio_return(weights):
            return np.sum(expected_returns.values * weights)
        
        def negative_utility(weights):
            ret = portfolio_return(weights)
            var = portfolio_variance(weights)
            # Add regime-based risk adjustment
            regime_penalty = vol_regime * 0.1 if vol_regime > 1.2 else 0
            return -(ret - 0.5 * (risk_aversion + regime_penalty) * var)
        
        n_assets = len(expected_returns)
        constraint_funcs = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        min_weight = constraints.get('min_weight', 0.01)
        max_weight = constraints.get('max_weight', 0.25)
        
        # Dynamic constraint adjustment based on regime
        if vol_regime > 1.5:  # High volatility - more conservative
            max_weight = min(max_weight, 0.15)
            min_weight = max(min_weight, 0.02)
        elif vol_regime < 0.7:  # Low volatility - allow more concentration  
            max_weight = min(max_weight * 1.2, 0.35)
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        logger.info(f"Optimizing with regime-adjusted constraints: min={min_weight:.3f}, max={max_weight:.3f}")
        logger.info(f"Expected returns range: {expected_returns.min():.4f} to {expected_returns.max():.4f}")
        
        try:
            result = minimize(negative_utility, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraint_funcs)
            
            if result.success:
                weights = result.x
                logger.info(f"Multi-timeframe optimization successful. Max weight: {max(weights):.3f}")
            else:
                logger.warning(f"Optimization failed: {result.message}, using equal weights")
                weights = np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Optimization error: {e}, using equal weights")
            weights = np.ones(n_assets) / n_assets
        
        return dict(zip(expected_returns.index, weights))