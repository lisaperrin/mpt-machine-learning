from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class OptimizationRequest(BaseModel):
    assets: List[str]
    constraints: Optional[Dict[str, float]] = None


class PortfolioWeights(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    top_holdings: List[tuple]
    diagnostics: Optional[Dict[str, Any]] = None


class OptimizationResponse(BaseModel):
    success: bool
    results: Optional[Dict[str, PortfolioWeights]] = None
    error: Optional[str] = None
    selected_assets: Optional[List[str]] = None
    missing_assets: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    data_quality: Optional[Dict[str, Any]] = None
    optimizer_status: Optional[Dict[str, Any]] = None
