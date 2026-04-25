"""
api/main.py
-----------
FastAPI application for football match outcome prediction (AUTO FEATURES).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

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


# ─────────────────────────────────────────────────────────────
# Lifespan (startup)
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + feature builder at startup."""
    try:
        config = load_config("configs/config.yaml")

        from src.models.predict import Predictor
        from src.features.live_features import LiveFeatureBuilder

        app.state.config = config
        app.state.predictor = Predictor(config_path="configs/config.yaml")
        app.state.feature_builder = LiveFeatureBuilder()

        logger.success("Model + Feature Builder loaded successfully.")

    except Exception as exc:
        logger.error(f"Startup failed: {exc}")
        app.state.predictor = None
        app.state.feature_builder = None

    yield

    logger.info("Shutting down API.")


# ─────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football Match Prediction API",
    version="2.0.0",
    description="Fully automatic football match prediction API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def get_services(request: Request):
    predictor = request.app.state.predictor
    feature_builder = request.app.state.feature_builder

    if predictor is None or feature_builder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or feature builder not loaded.",
        )

    return predictor, feature_builder


# ─────────────────────────────────────────────────────────────
# Request schema (SIMPLE INPUT)
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel

class MatchRequest(BaseModel):
    home_team: str
    away_team: str


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health(request: Request):
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.predictor is not None,
        version="2.0.0",
    )


# 🔥 SINGLE PREDICTION (AUTO)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={503: {"model": ErrorResponse}},
)
async def predict(req: MatchFeaturesRequest, request: Request):
    predictor, feature_builder = get_services(request)

    try:
        # ✅ AUTO FEATURE BUILD
        features = feature_builder.build_match_features(
            req.home_team,
            req.away_team
        )

        result = predictor.predict(features)

        return PredictionResponse(
            home_team=req.home_team,
            away_team=req.away_team,
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


# 🔥 BATCH PREDICTION (AUTO)
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
    if not request_list:
        return []

    predictor, feature_builder = get_services(request)

    try:
        responses = []

        for req in request_list:
            features = feature_builder.build_match_features(
                req.home_team,
                req.away_team
            )

            result = predictor.predict(features)

            responses.append(
                PredictionResponse(
                    home_team=req.home_team,
                    away_team=req.away_team,
                    home_win_prob=float(result["home_win_prob"]),
                    draw_prob=float(result["draw_prob"]),
                    away_win_prob=float(result["away_win_prob"]),
                    predicted_outcome=str(result["predicted_outcome"]),
                )
            )

        return responses

    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(exc)}",
        )


# ─────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────

def start() -> None:
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