from typing import Dict, List, Optional

from pydantic import BaseModel


class OptimizationRequest(BaseModel):
    assets: List[str]
    constraints: Optional[Dict[str, float]] = None


class PortfolioWeights(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    top_holdings: List[tuple]


class OptimizationResponse(BaseModel):
    success: bool
    results: Optional[Dict[str, PortfolioWeights]] = None
    error: Optional[str] = None
    selected_assets: Optional[List[str]] = None
