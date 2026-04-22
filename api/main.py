"""
api/main.py
-----------
FastAPI application for football match outcome prediction.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    MatchFeaturesRequest,
    PredictionResponse,
)
from src.utils.helpers import load_config


# ─── Lifespan (startup/shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    try:
        config = load_config("configs/config.yaml")
        from src.models.predict import Predictor

        app.state.config = config
        app.state.predictor = Predictor(config_path="configs/config.yaml")

        logger.success("Model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        app.state.predictor = None

    yield

    logger.info("Shutting down API.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football Match Prediction API",
    version="1.0.0",
    description="Predict football match outcomes using ML models",
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

def get_predictor(request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train model first.",
        )
    return predictor


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health(request: Request):
    """Check API + model status."""
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.predictor is not None,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={503: {"model": ErrorResponse}},
)
async def predict(request_data: MatchFeaturesRequest, request: Request):
    """Predict a single match outcome."""
    predictor = get_predictor(request)

    try:
        features = request_data.to_feature_dict()
        result = predictor.predict(features)

        return PredictionResponse(
            home_team=request_data.home_team,
            away_team=request_data.away_team,
            home_win_prob=float(result["home_win_prob"]),
            draw_prob=float(result["draw_prob"]),
            away_win_prob=float(result["away_win_prob"]),
            predicted_outcome=str(result["predicted_outcome"]),
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    tags=["Prediction"],
    responses={503: {"model": ErrorResponse}},
)
async def predict_batch(
    request_list: List[MatchFeaturesRequest],
    request: Request,
):
    """Predict multiple matches."""
    if not request_list:
        return []

    predictor = get_predictor(request)

    try:
        records = [r.to_feature_dict() for r in request_list]
        df = pd.DataFrame(records)

        df_out = predictor.predict_batch(df)

        responses = []
        for req, row in zip(request_list, df_out.to_dict(orient="records")):
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

    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(exc)}",
        )


# ─── Dev runner ──────────────────────────────────────────────────────────────

def start() -> None:
    """Run API via CLI."""
    import uvicorn

    cfg = load_config("configs/config.yaml")
    api_cfg = cfg.get("api", {})

    uvicorn.run(
        "api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", True),
    )


if __name__ == "__main__":
    start()