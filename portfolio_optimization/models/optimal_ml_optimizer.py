import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimalMLPortfolioOptimizer:
    """
    Optimal ML Portfolio Optimizer combining HRP with Black-Litterman.

    Architecture:
    1. Base allocation from Hierarchical Risk Parity — stable, no return prediction
    2. Return-seeking allocation from Black-Litterman with ML views
       - Ridge regression generates per-asset return views with confidence
       - Momentum-informed equilibrium (not equal-weight) as BL prior
       - Low confidence → BL stays near equilibrium (safe default)
       - High confidence → BL moves toward ML prediction
       - Max-Sharpe optimization on BL posterior
    3. Auto-tuned blend ratio via walk-forward cross-validation
       - Tests HRP/BL blends from 30/70 to 70/30
       - Selects the blend with best OOS Sharpe ratio
    """

    def __init__(self,
                 returns_data: pd.DataFrame,
                 price_data: pd.DataFrame,
                 config: Dict):

        self.returns = returns_data
        self.prices = price_data
        self.config = config

        # Black-Litterman parameters
        self.tau = 0.15          # Uncertainty scaling for prior covariance
        self.delta = 2.5         # Risk aversion coefficient
        self.ridge_alpha = 1.0   # Ridge regularization strength
        self.risk_free_rate = config.get('risk_free_rate', 0.045)

        # Diagnostics
        self.ml_views = {}
        self.bl_diagnostics = {}
        self.weight_comparison = {}

    def calculate_hrp_weights(self) -> Dict[str, float]:
        """
        Calculate Hierarchical Risk Parity weights using correlation clustering.
        Uses Lopez de Prado's recursive bisection on the linkage tree.
        """
        logger.info("Calculating HRP base allocation...")

        recent_returns = self.returns.tail(252)
        corr_matrix = recent_returns.corr().fillna(0)

        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        condensed_distances = squareform(distance_matrix.values, checks=False)
        linkage_matrix = linkage(condensed_distances, method='ward')

        root_node = to_tree(linkage_matrix)

        n_assets = len(corr_matrix.columns)
        weights = np.zeros(n_assets)
        self._tree_bisection(root_node, recent_returns, weights, 1.0)

        max_hrp_weight = 0.20
        weight_dict = {}
        for i, asset in enumerate(corr_matrix.columns):
            weight_dict[asset] = min(weights[i], max_hrp_weight)

        total = sum(weight_dict.values())
        for asset in weight_dict:
            weight_dict[asset] /= total

        logger.info(f"HRP allocation completed for {len(weight_dict)} assets")
        return weight_dict

    def _tree_bisection(self, node, returns: pd.DataFrame,
                        weights: np.ndarray, allocation: float) -> None:
        if node.is_leaf():
            weights[node.id] = allocation
            return

        left_leaves = self._get_leaves(node.left)
        right_leaves = self._get_leaves(node.right)

        left_var = self._calculate_cluster_variance(returns.iloc[:, left_leaves])
        right_var = self._calculate_cluster_variance(returns.iloc[:, right_leaves])

        total_inv_var = (1.0 / left_var) + (1.0 / right_var)
        left_alloc = (1.0 / left_var) / total_inv_var
        right_alloc = (1.0 / right_var) / total_inv_var

        self._tree_bisection(node.left, returns, weights, allocation * left_alloc)
        self._tree_bisection(node.right, returns, weights, allocation * right_alloc)

    def _get_leaves(self, node) -> List[int]:
        if node.is_leaf():
            return [node.id]
        return self._get_leaves(node.left) + self._get_leaves(node.right)

    def _calculate_cluster_variance(self, cluster_returns: pd.DataFrame) -> float:
        equal_weight = 1.0 / len(cluster_returns.columns)
        portfolio_returns = (cluster_returns * equal_weight).sum(axis=1)
        return portfolio_returns.var()


    def generate_ml_views(self, lookback: int = 252) -> tuple:
        """
        Generate per-asset return views using cross-sectional factor scoring
        refined by Ridge regression.

        Step 1: Score assets cross-sectionally on momentum, quality, and
                mean-reversion factors (these produce reliable relative rankings).
        Step 2: Train Ridge regression on factor features to predict forward
                returns; blend Ridge predictions where they have signal (R² > 0).
        Step 3: Confidence is driven by signal strength (factor score dispersion),
                not by Ridge R² alone (which is near-zero for raw returns).

        Returns:
            Q: (n_assets,) expected daily returns per asset
            P: (n_assets, n_assets) identity pick matrix (absolute views)
            confidences: (n_assets,) confidence per view [0.2, 0.7]
        """
        logger.info("Generating ML views for Black-Litterman...")

        recent = self.returns.tail(lookback)
        assets = list(self.returns.columns)
        n_assets = len(assets)

        mom_6m = recent.tail(126).mean() / (recent.tail(126).std() + 1e-8)
        quality = recent.mean() / (recent.std() + 1e-8)
        mom_3m = recent.tail(63).mean() / (recent.tail(63).std() + 1e-8)
        mom_1m = recent.tail(21).mean() / (recent.tail(21).std() + 1e-8)

        composite = (
            mom_6m.rank(pct=True) * 0.35
            + quality.rank(pct=True) * 0.30
            + mom_3m.rank(pct=True) * 0.20
            + (1 - mom_1m.rank(pct=True)) * 0.15
        )
        z_scores = (composite - composite.mean()) / (composite.std() + 1e-8)

        hist_means = recent.mean()
        cross_sec_spread = hist_means.std()
        factor_views = hist_means.values + z_scores.values * cross_sec_spread

        ridge_views = np.zeros(n_assets)
        ridge_weight = 0.0  # how much to blend Ridge in

        extended = self.returns.tail(lookback + 126)
        forward_window = 21
        forward_target = extended.rolling(forward_window).mean().shift(-forward_window)
        market_return = extended.mean(axis=1)

        ridge_r2s = []
        for i, asset in enumerate(assets):
            features = pd.DataFrame(index=extended.index)
            features['mom_21d'] = extended[asset].rolling(21).mean()
            features['mom_63d'] = extended[asset].rolling(63).mean()
            features['mom_126d'] = extended[asset].rolling(126).mean()
            features['sharpe_63d'] = (
                extended[asset].rolling(63).mean()
                / (extended[asset].rolling(63).std() + 1e-8)
            )
            features['vol_21d'] = extended[asset].rolling(21).std()
            features['rel_strength'] = (
                extended[asset].rolling(63).mean() - market_return.rolling(63).mean()
            )

            target = forward_target[asset]
            valid = features.notna().all(axis=1) & target.notna()
            X = features.loc[valid].values
            y = target.loc[valid].values

            if len(X) < 60:
                ridge_views[i] = factor_views[i]
                ridge_r2s.append(0.0)
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=self.ridge_alpha)
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            mean_r2 = max(0.0, float(np.mean(cv_scores)))
            ridge_r2s.append(mean_r2)

            model.fit(X_scaled, y)
            latest_row = features.dropna().iloc[-1:].values
            latest_scaled = scaler.transform(latest_row)
            ridge_views[i] = float(model.predict(latest_scaled)[0])

        avg_r2 = np.mean(ridge_r2s)
        ridge_weight = min(0.3, max(0.0, avg_r2 * 3))  # R²=0.1 → 30% Ridge
        Q = (1 - ridge_weight) * factor_views + ridge_weight * ridge_views

        abs_z = np.abs(z_scores.values)
        max_z = abs_z.max() + 1e-8
        confidences = np.clip(0.25 + 0.45 * (abs_z / max_z), 0.25, 0.70)

        view_details = {}
        for i, asset in enumerate(assets):
            view_details[asset] = {
                'predicted_return': float(Q[i]),
                'factor_score': float(z_scores.iloc[i]),
                'ridge_r2': float(ridge_r2s[i]),
                'confidence': float(confidences[i]),
            }
        self.ml_views = view_details

        logger.info(f"  Factor scoring: z-scores range [{z_scores.min():.2f}, {z_scores.max():.2f}]")
        logger.info(f"  Ridge blend weight: {ridge_weight:.1%} (avg R²={avg_r2:.3f})")
        for asset in assets:
            d = view_details[asset]
            logger.info(f"  {asset}: Q={d['predicted_return']:.6f}/day, "
                        f"z={d['factor_score']:+.2f}, conf={d['confidence']:.2f}")

        P = np.eye(n_assets)
        return Q, P, confidences

    def black_litterman_returns(self, cov_matrix: np.ndarray,
                                 Q: np.ndarray, P: np.ndarray,
                                 confidences: np.ndarray) -> tuple:
        """
        Compute Black-Litterman posterior expected returns and covariance.

        Args:
            cov_matrix: (n, n) daily covariance matrix
            Q: (k,) ML-predicted expected returns (daily)
            P: (k, n) pick matrix
            confidences: (k,) confidence per view [0, 1]

        Returns:
            mu_bl: (n,) posterior expected returns
            sigma_bl: (n, n) posterior covariance
        """
        n_assets = cov_matrix.shape[0]

        # Equilibrium returns: implied by momentum-tilted market portfolio
        # Uses trailing 6-month returns to tilt away from equal weight
        trailing_returns = self.returns.tail(126).mean()
        w_mkt = self._softmax_weights(trailing_returns.values, temperature=200.0)
        w_mkt = np.clip(w_mkt, 0.02, 0.40)
        w_mkt /= w_mkt.sum()
        pi = self.delta * cov_matrix @ w_mkt

        # If no views, return equilibrium
        if len(Q) == 0:
            logger.info("No views provided, using equilibrium returns")
            self.bl_diagnostics = {'equilibrium_returns': pi.tolist(), 'posterior': 'equilibrium'}
            return pi, self.tau * cov_matrix

        tau_sigma = self.tau * cov_matrix

        # View uncertainty: Omega = diag(tau * P @ Sigma @ P.T) scaled by 1/confidence
        sigma_views = P @ cov_matrix @ P.T
        omega_diag = self.tau * np.diag(sigma_views)

        # Scale uncertainty inversely by confidence (with floor)
        # confidence=1 → omega stays at tau*sigma_view (fully trust ML)
        # confidence=0.1 (floor) → omega = 10x prior (weak but nonzero influence)
        confidence_floor = 0.10
        for i in range(len(confidences)):
            effective_conf = max(confidences[i], confidence_floor)
            omega_diag[i] /= effective_conf

        omega = np.diag(omega_diag)

        try:
            # Add small ridge for numerical stability
            ridge = 1e-8 * np.eye(n_assets)
            tau_sigma_inv = np.linalg.inv(tau_sigma + ridge)
            omega_inv = np.linalg.inv(omega + 1e-10 * np.eye(len(Q)))

            sigma_bl = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            mu_bl = sigma_bl @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        except np.linalg.LinAlgError:
            logger.warning("BL matrix inversion failed, falling back to equilibrium")
            mu_bl = pi
            sigma_bl = tau_sigma

        self.bl_diagnostics = {
            'equilibrium_returns': pi.tolist(),
            'posterior_returns': mu_bl.tolist(),
            'avg_confidence': float(np.mean(confidences)),
            'n_views': len(Q),
            'return_shift': float(np.mean(np.abs(mu_bl - pi)))
        }

        logger.info(f"BL posterior: {len(Q)} views, avg confidence={np.mean(confidences):.3f}, "
                     f"avg return shift={np.mean(np.abs(mu_bl - pi)):.6f}")

        return mu_bl, sigma_bl

    def optimize_bl_portfolio(self, mu_bl: np.ndarray, sigma_bl: np.ndarray,
                               asset_names: list,
                               min_weight: float = 0.005,
                               max_weight: float = 0.30) -> Dict[str, float]:
        """
        Max-Sharpe optimization on Black-Litterman posterior.
        """
        n = len(asset_names)
        rf_daily = self.risk_free_rate / 252

        def neg_sharpe(w):
            ret = w @ mu_bl
            vol = np.sqrt(w @ sigma_bl @ w)
            return -(ret - rf_daily) / vol if vol > 1e-10 else 0.0

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        w0 = np.ones(n) / n

        try:
            result = minimize(neg_sharpe, w0, method='SLSQP',
                              bounds=bounds, constraints=constraints)
            if result.success:
                weights = result.x
            else:
                logger.warning(f"BL optimization failed: {result.message}, using softmax fallback")
                weights = self._softmax_weights(mu_bl)
        except Exception as e:
            logger.warning(f"BL optimization error: {e}, using softmax fallback")
            weights = self._softmax_weights(mu_bl)

        # Ensure bounds
        weights = np.clip(weights, min_weight, max_weight)
        weights /= weights.sum()

        return dict(zip(asset_names, weights.tolist()))

    def _softmax_weights(self, mu: np.ndarray, temperature: float = 500.0) -> np.ndarray:
        """Softmax fallback when optimization fails."""
        scaled = mu * temperature
        scaled -= scaled.max()  # numerical stability
        exp_scores = np.exp(scaled)
        return exp_scores / exp_scores.sum()

    def _find_optimal_blend(self, constraints: Dict) -> float:
        """
        Find optimal HRP/BL blend ratio via walk-forward cross-validation.

        Tests candidate blends [0.3, 0.4, 0.5, 0.6, 0.7] over non-overlapping
        folds. For each fold, HRP and BL are computed once on training data,
        then all blends are evaluated on the test window. Returns the blend
        with the best average OOS Sharpe ratio.
        """
        candidate_blends = [0.3, 0.4, 0.5, 0.6, 0.7]
        n_days = len(self.returns)
        train_window = min(252, n_days - 63)
        test_window = 63
        max_folds = 3

        if n_days < train_window + test_window:
            logger.info("Insufficient data for blend optimization, using default 0.5")
            return 0.5

        min_w = constraints.get('min_weight', 0.005)
        max_w = constraints.get('max_weight', 0.25)
        blend_sharpes = {b: [] for b in candidate_blends}

        fold_start = max(0, n_days - max_folds * test_window - train_window)
        n_folds = 0

        while fold_start + train_window + test_window <= n_days and n_folds < max_folds:
            train_end = fold_start + train_window
            test_end = train_end + test_window

            train_returns = self.returns.iloc[fold_start:train_end]
            train_prices = self.prices.iloc[fold_start:train_end]
            test_returns = self.returns.iloc[train_end:test_end]

            try:
                temp = OptimalMLPortfolioOptimizer(
                    train_returns, train_prices, self.config
                )

                hrp_w = temp.calculate_hrp_weights()

                recent = train_returns.tail(252).dropna()
                try:
                    cov = LedoitWolf().fit(recent).covariance_
                except Exception:
                    cov = recent.cov().values

                Q, P, conf = temp.generate_ml_views()
                mu_bl, sigma_bl = temp.black_litterman_returns(cov, Q, P, conf)
                bl_w = temp.optimize_bl_portfolio(
                    mu_bl, sigma_bl, list(train_returns.columns),
                    min_weight=min_w, max_weight=max_w
                )

                for blend in candidate_blends:
                    blended = {}
                    bl_share = 1.0 - blend
                    for asset in hrp_w:
                        w = blend * hrp_w[asset] + bl_share * bl_w.get(asset, 0.0)
                        blended[asset] = max(0.001, w)
                    total = sum(blended.values())
                    for asset in blended:
                        blended[asset] /= total

                    weights_s = pd.Series(blended).reindex(test_returns.columns, fill_value=0)
                    port_rets = (test_returns * weights_s).sum(axis=1).values
                    blend_sharpes[blend].append(self._annualized_sharpe(port_rets))

            except Exception as e:
                logger.warning(f"Blend CV fold {n_folds} failed: {e}")

            fold_start += test_window
            n_folds += 1

        avg_sharpes = {b: np.mean(s) for b, s in blend_sharpes.items() if len(s) > 0}
        if not avg_sharpes:
            logger.info("Blend optimization produced no results, using default 0.5")
            return 0.5

        best_blend = max(avg_sharpes, key=avg_sharpes.get)

        logger.info(
            f"Blend CV ({n_folds} folds): "
            + ", ".join(f"{b:.0%}HRP={s:.3f}" for b, s in sorted(avg_sharpes.items()))
        )
        logger.info(f"Selected blend: {best_blend:.0%} HRP / {1 - best_blend:.0%} BL "
                     f"(Sharpe={avg_sharpes[best_blend]:.3f})")

        self.bl_diagnostics['blend_cv'] = {
            'candidate_sharpes': {f"{int(b*100)}pct_hrp": round(s, 4) for b, s in avg_sharpes.items()},
            'selected_hrp_weight': best_blend,
            'n_folds': n_folds
        }

        return best_blend

    @staticmethod
    def _annualized_sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) * 252 / (np.std(returns) * np.sqrt(252)))

    def get_optimal_portfolio_weights(self,
                                      constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Generate optimal portfolio weights: HRP + Black-Litterman with
        auto-tuned blend ratio via walk-forward cross-validation.
        """
        if constraints is None:
            constraints = self.config.get('portfolio_constraints', {})

        min_w = constraints.get('min_weight', 0.005)
        max_w = constraints.get('max_weight', 0.25)

        hrp_weight = self._find_optimal_blend(constraints)

        logger.info(f"Generating optimal portfolio (HRP: {hrp_weight:.0%}, BL: {1 - hrp_weight:.0%})...")

        hrp_weights = self.calculate_hrp_weights()

        recent_returns = self.returns.tail(252).dropna()
        try:
            lw = LedoitWolf().fit(recent_returns)
            cov_matrix = lw.covariance_
        except Exception:
            cov_matrix = recent_returns.cov().values

        Q, P, confidences = self.generate_ml_views()

        mu_bl, sigma_bl = self.black_litterman_returns(cov_matrix, Q, P, confidences)

        bl_weights = self.optimize_bl_portfolio(
            mu_bl, sigma_bl, list(self.returns.columns),
            min_weight=min_w, max_weight=max_w
        )

        bl_share = 1.0 - hrp_weight
        final_weights = {}
        for asset in hrp_weights:
            blended = hrp_weight * hrp_weights[asset] + bl_share * bl_weights.get(asset, 0.0)
            final_weights[asset] = max(0.001, blended)

        total = sum(final_weights.values())
        for asset in final_weights:
            final_weights[asset] /= total

        final_weights = self._apply_constraints(final_weights, constraints)

        self.weight_comparison = self.compare_hrp_vs_final(hrp_weights, final_weights)
        self.bl_diagnostics['selected_hrp_weight'] = hrp_weight

        logger.info(f"Optimal portfolio: {len(final_weights)} assets, "
                    f"range {min(final_weights.values()):.3f}-{max(final_weights.values()):.3f}")

        return final_weights

    def compare_hrp_vs_final(self, hrp_weights: Dict[str, float],
                              final_weights: Dict[str, float]) -> Dict:
        comparison = {}
        for asset in hrp_weights:
            hrp_w = hrp_weights[asset]
            final_w = final_weights.get(asset, 0.0)
            diff = final_w - hrp_w
            pct_change = (diff / hrp_w * 100) if hrp_w > 0 else 0.0
            comparison[asset] = {
                'hrp_weight': round(hrp_w, 4),
                'final_weight': round(final_w, 4),
                'bl_tilt': round(diff, 4),
                'pct_change': round(pct_change, 1)
            }

        tilts = [v['bl_tilt'] for v in comparison.values()]
        summary = {
            'max_overweight': max(tilts),
            'max_underweight': min(tilts),
            'avg_abs_tilt': float(np.mean(np.abs(tilts))),
            'tracking_error_contribution': float(np.std(tilts))
        }

        logger.info("HRP vs Final weight comparison:")
        sorted_by_tilt = sorted(comparison.items(), key=lambda x: abs(x[1]['bl_tilt']), reverse=True)
        for asset, data in sorted_by_tilt[:5]:
            logger.info(f"  {asset}: HRP={data['hrp_weight']:.3f} -> Final={data['final_weight']:.3f} "
                        f"(BL tilt: {data['bl_tilt']:+.4f}, {data['pct_change']:+.1f}%)")

        return {'asset_comparison': comparison, 'summary': summary}

    def _apply_constraints(self, weights: Dict[str, float], constraints: Dict) -> Dict[str, float]:
        min_weight = constraints.get('min_weight', 0.005)
        max_weight = constraints.get('max_weight', 0.15)

        constrained = {}
        for asset, weight in weights.items():
            constrained[asset] = np.clip(weight, min_weight, max_weight)

        total = sum(constrained.values())
        for asset in constrained:
            constrained[asset] /= total

        return constrained

    def get_regime_adjusted_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        recent_vol = self.returns.mean(axis=1).rolling(21).std().iloc[-1]
        long_vol = self.returns.mean(axis=1).rolling(252).std().iloc[-1]
        vol_ratio = recent_vol / (long_vol + 1e-8)

        if vol_ratio > 1.5:
            logger.info("High volatility regime detected - applying defensive positioning")
            asset_vols = self.returns.rolling(63).std().iloc[-1]
            vol_ranks = asset_vols.rank(pct=True)

            adjusted = {}
            for asset, weight in base_weights.items():
                adjustment = 1.0 - 0.2 * vol_ranks[asset]
                adjusted[asset] = weight * adjustment

            total = sum(adjusted.values())
            for asset in adjusted:
                adjusted[asset] /= total
            return adjusted

        logger.info("Normal volatility regime - using base weights")
        return base_weights
