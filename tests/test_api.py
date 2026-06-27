"""Tests for FastAPI endpoints."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.cache as cache
from main import app


class DummyCollector:
    def __init__(self, prices):
        self.prices = prices

    def get_data_quality_report(self):
        return {
            "total_assets": len(self.prices.columns),
            "failed_assets": 0,
            "failed_tickers": [],
            "date_range": (str(self.prices.index.min()), str(self.prices.index.max())),
            "data_points": len(self.prices),
            "missing_data_pct": {col: 0.0 for col in self.prices.columns},
        }


@pytest.fixture(autouse=True)
def synthetic_market_cache():
    rng = np.random.default_rng(42)
    assets = ["AAPL", "MSFT", "GOOGL", "JPM", "TLT", "GLD"]
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(504, len(assets))),
        columns=assets,
        index=pd.bdate_range(end="2025-01-01", periods=504),
    )
    prices = (1 + returns).cumprod() * 100

    cache.MASTER_CACHE["data"] = {
        "prices": prices,
        "returns": returns,
        "collector": DummyCollector(prices),
    }
    cache.MASTER_CACHE["timestamp"] = 10**12
    cache.RESULT_CACHE.clear()
    yield
    cache.MASTER_CACHE["data"] = None
    cache.MASTER_CACHE["timestamp"] = 0
    cache.RESULT_CACHE.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAndAssets:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_get_assets(self, client):
        response = client.get("/api/assets")
        assert response.status_code == 200
        data = response.json()
        assert "asset_universe" in data
        assert data["total_assets"] > 0


class TestOptimization:
    def test_optimize_returns_all_strategies(self, client):
        response = client.post("/api/optimize", json={
            "assets": ["AAPL", "MSFT", "GOOGL", "JPM", "TLT", "GLD"],
            "constraints": {"min_weight": 0.05, "max_weight": 0.40}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        expected_strategies = [
            "Equal Weight",
            "Max Sharpe (MPT)",
            "Min Volatility (MPT)",
            "Factor-Based",
            "Optimal ML (HRP+Black-Litterman)"
        ]
        for strategy in expected_strategies:
            assert strategy in data["results"], f"Missing strategy: {strategy}"
        assert "optimizer_status" in data
        assert "data_quality" in data

    def test_optimize_weights_sum_to_one(self, client):
        response = client.post("/api/optimize", json={
            "assets": ["AAPL", "MSFT", "GOOGL", "JPM", "TLT", "GLD"],
        })
        data = response.json()

        for strategy_name, strategy_data in data["results"].items():
            weight_sum = sum(strategy_data["weights"].values())
            assert abs(weight_sum - 1.0) < 0.01, f"{strategy_name} weights sum to {weight_sum}"

    def test_optimize_rejects_too_few_assets(self, client):
        response = client.post("/api/optimize", json={
            "assets": ["AAPL", "MSFT"],
        })
        assert response.status_code == 400

    def test_optimize_handles_missing_assets(self, client):
        response = client.post("/api/optimize", json={
            "assets": ["AAPL", "MSFT", "GOOGL", "FAKEASSET1", "FAKEASSET2", "TLT"],
        })
        data = response.json()
        assert data["success"] is True
        assert "FAKEASSET1" not in data["selected_assets"]
        assert "FAKEASSET1" in data["missing_assets"]

    def test_optimize_rejects_infeasible_constraints(self, client):
        response = client.post("/api/optimize", json={
            "assets": ["AAPL", "MSFT", "GOOGL"],
            "constraints": {"min_weight": 0.40, "max_weight": 0.60}
        })
        assert response.status_code == 400
