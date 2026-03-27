import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def create_factor_strategy(returns_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
    """
    Factor-based portfolio strategy using momentum, quality, and volatility signals.

    Combines multiple cross-sectional signals into portfolio weights:
    1. Multi-timeframe risk-adjusted momentum (3m, 6m, 12m)
    2. Quality signal (rolling Sharpe proxy)
    3. Short-term mean reversion (contrarian)
    4. Volatility preference (moderate vol favored)
    """
    logger.info("Creating factor-based portfolio strategy...")

    # Multi-timeframe momentum signals
    returns_21d = returns_data.rolling(21).mean().iloc[-1]
    returns_63d = returns_data.rolling(63).mean().iloc[-1]
    returns_126d = returns_data.rolling(126).mean().iloc[-1]
    returns_252d = returns_data.rolling(252).mean().iloc[-1]

    volatility_63d = returns_data.rolling(63).std().iloc[-1]

    # Risk-adjusted momentum
    risk_adj_mom_3m = returns_63d / (volatility_63d + 1e-8)
    risk_adj_mom_6m = returns_126d / (volatility_63d + 1e-8)

    # Quality signal (Sharpe-like ratio)
    quality_signal = returns_63d / (volatility_63d + 1e-8)

    # Mean reversion signal (short-term contrarian)
    mean_reversion = -returns_21d

    # Convert to percentile ranks (robust to outliers)
    momentum_3m_rank = risk_adj_mom_3m.rank(pct=True)
    momentum_6m_rank = risk_adj_mom_6m.rank(pct=True)
    momentum_12m_rank = (
        returns_252d.rank(pct=True)
        if not returns_252d.isna().all()
        else pd.Series(0.5, index=returns_data.columns)
    )
    quality_rank = quality_signal.rank(pct=True)
    mean_rev_rank = mean_reversion.rank(pct=True)

    # Volatility score (prefer moderate volatility)
    vol_rank = volatility_63d.rank(pct=True)
    vol_score = 1 - np.abs(vol_rank - 0.5) * 2

    # Combine signals
    ml_scores = (
        0.25 * momentum_3m_rank
        + 0.20 * momentum_6m_rank
        + 0.15 * momentum_12m_rank
        + 0.20 * quality_rank
        + 0.10 * mean_rev_rank
        + 0.10 * vol_score
    )

    # Softmax with dynamic scaling
    signal_strength = ml_scores.std()
    scale_factor = 2.0 + 3.0 * signal_strength
    exp_scores = np.exp(ml_scores * scale_factor)
    weights = exp_scores / exp_scores.sum()

    # Dynamic constraints based on portfolio size
    n_assets = len(returns_data.columns)
    if n_assets <= 5:
        min_weight, max_weight = 0.05, 0.40
    elif n_assets <= 10:
        min_weight, max_weight = 0.03, 0.30
    else:
        min_weight, max_weight = 0.02, 0.20

    weights = np.clip(weights, min_weight, max_weight)

    # Prevent extreme concentration
    while weights.max() / weights.min() > 15 and max_weight > min_weight * 2:
        max_weight *= 0.9
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

    weights = weights / weights.sum()

    weight_dict = {asset: float(w) for asset, w in zip(returns_data.columns, weights)}

    logger.info(f"Factor strategy: {len(weight_dict)} assets, range {min(weights):.3f}-{max(weights):.3f}")
    return weight_dict
