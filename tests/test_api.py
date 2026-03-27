"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from main import app


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
