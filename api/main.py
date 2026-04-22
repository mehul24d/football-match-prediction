"""
api/main.py
-----------
FastAPI application for football match outcome prediction.

Endpoints
---------
GET  /health          – health check + model status
POST /predict         – predict a single match
POST /predict/batch   – predict multiple matches from JSON list

Start the server
----------------
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    # or
    make run-api
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    MatchFeaturesRequest,
    PredictionResponse,
)
from src.utils.helpers import load_config

# ─── Application state ───────────────────────────────────────────────────────

_predictor: Any = None
_config: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    global _predictor, _config
    try:
        _config = load_config("configs/config.yaml")
        from src.models.predict import Predictor
        _predictor = Predictor(config_path="configs/config.yaml")
        logger.success("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            f"Model not found – prediction endpoints will return 503. Details: {exc}"
        )
    yield
    logger.info("Shutting down API.")


# ─── App factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football Match Prediction API",
    description=(
        "Production-grade REST API for predicting football (soccer) match outcomes. "
        "Returns win/draw/loss probabilities for home and away teams."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _require_model() -> None:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Prediction model is not loaded. "
                "Train the model first: python -m src.models.train"
            ),
        )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
async def health() -> HealthResponse:
    """Returns API health status and whether the prediction model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict single match outcome",
    tags=["Prediction"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"description": "Validation error"},
    },
)
async def predict(request: MatchFeaturesRequest) -> PredictionResponse:
    """
    Predict the outcome probabilities (Home Win / Draw / Away Win)
    for a single football match.
    """
    _require_model()
    features = request.to_feature_dict()
    try:
        result = _predictor.predict(features)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )
    return PredictionResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        home_win_prob=result["home_win_prob"],
        draw_prob=result["draw_prob"],
        away_win_prob=result["away_win_prob"],
        predicted_outcome=result["predicted_outcome"],
    )


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    summary="Predict multiple matches",
    tags=["Prediction"],
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_batch(requests: list[MatchFeaturesRequest]) -> list[PredictionResponse]:
    """
    Predict outcome probabilities for a list of matches in a single call.
    """
    if not requests:
        return []

    _require_model()
    records = [r.to_feature_dict() for r in requests]
    df = pd.DataFrame(records)

    try:
        df_out = _predictor.predict_batch(df)
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        )

    responses = []
    for req, (_, row) in zip(requests, df_out.iterrows()):
        responses.append(
            PredictionResponse(
                home_team=req.home_team,
                away_team=req.away_team,
                home_win_prob=float(row["home_win_prob"]),
                draw_prob=float(row["draw_prob"]),
                away_win_prob=float(row["away_win_prob"]),
                predicted_outcome=str(row["predicted_outcome"]),
            )
        )
    return responses


# ─── Dev runner ──────────────────────────────────────────────────────────────

def start() -> None:
    """Entry point for `fmp-api` CLI command."""
    import uvicorn
    cfg = load_config("configs/config.yaml")
    api_cfg = cfg.get("api", {})
    uvicorn.run(
        "api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
    )


if __name__ == "__main__":
    start()
