"""
tests/test_api.py
-----------------
Integration tests for the FastAPI prediction endpoints.
Uses FastAPI's TestClient to exercise the API without a live server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ─── Health endpoint ─────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "ok"


# ─── Predict endpoint (no model loaded) ─────────────────────────────────────

class TestPredictNoModel:
    """When no model is loaded the endpoint should return 503."""

    def test_predict_without_model_returns_503(self, client):
        payload = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_elo": 1650.0,
            "away_elo": 1580.0,
        }
        response = client.post("/predict", json=payload)
        # Model not trained yet → 503
        assert response.status_code == 503

    def test_batch_predict_without_model_returns_503(self, client):
        payload = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "home_elo": 1650.0, "away_elo": 1580.0},
        ]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 503


# ─── Input validation ────────────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_required_fields_returns_422(self, client):
        # home_team and away_team are required
        response = client.post("/predict", json={"home_elo": 1500})
        assert response.status_code == 422

    def test_negative_elo_returns_422(self, client):
        payload = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_elo": -100.0,
            "away_elo": 1500.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_form_above_max_returns_422(self, client):
        payload = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_elo": 1500.0,
            "away_elo": 1500.0,
            "home_form": 5.0,  # max is 3
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_batch_returns_200_empty_list(self, client):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 200
        assert response.json() == []
