"""
api/main.py
-----------
FastAPI application for football match outcome prediction.
Uses ONLY historical data: season + matchday + country.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List
from pathlib import Path

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
from src.models.predict import Predictor


# ─────────────────────────────────────────────────────────────
# Historical data manager
# ─────────────────────────────────────────────────────────────

class HistoricalDataManager:
    """Load and query historical features by season + matchday."""
    
    def __init__(self, data_dir: str | Path = "data/processed"):
        self.data_dir = Path(data_dir)
        self._cache = {}
    
    def get_season_data(self, season: str, country: str) -> pd.DataFrame:
        """Load processed features for a specific season + country."""
        key = f"{country}_{season}"
        
        if key in self._cache:
            return self._cache[key]
        
        # Map country to league code
        league_map = {
            "England": "E0",
            "Spain": "SP1",
            "Germany": "D1",
            "Italy": "I1",
            "France": "F1",
        }
        league_code = league_map.get(country, "E0")
        
        # File pattern: data/processed/matches_E0_2023_24.csv
        filename = self.data_dir / f"matches_{league_code}_{season.replace('-', '_')}.csv"
        
        if not filename.exists():
            raise FileNotFoundError(
                f"No data found for {country} {season}. "
                f"Expected: {filename}"
            )
        
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        
        self._cache[key] = df
        logger.info(f"✅ Loaded {country} {season}: {len(df)} matches")
        
        return df
    
    def get_features_at_matchday(
        self,
        season: str,
        country: str,
        matchday: int,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Extract features for a specific match from historical data.
        
        Returns features computed BEFORE the matchday (to avoid leakage).
        """
        df = self.get_season_data(season, country)
        
        # Add week/matchday if not present
        if "week" not in df.columns and "matchday" not in df.columns:
            df["week"] = range(1, len(df) + 1)
        
        week_col = "week" if "week" in df.columns else "matchday"
        
        # Get data UP TO (but not including) the current matchday
        historical = df[df[week_col] < matchday].copy()
        
        if historical.empty:
            raise ValueError(
                f"❌ No historical data before matchday {matchday} "
                f"for {home_team} vs {away_team}"
            )
        
        # Get latest stats for each team
        home_stats = self._get_team_stats(historical, home_team, "home")
        away_stats = self._get_team_stats(historical, away_team, "away")
        
        if not home_stats or not away_stats:
            raise ValueError(
                f"❌ Incomplete team data for {home_team} vs {away_team} "
                f"at matchday {matchday}"
            )
        
        # Merge into feature dict
        features = {**home_stats, **away_stats}
        features["season"] = int(season.split("-")[0])
        
        return features
    
    @staticmethod
    def _get_team_stats(df: pd.DataFrame, team: str, prefix: str) -> dict:
        """Extract latest stats for a team from historical dataframe."""
        
        # Find all matches where team played
        if prefix == "home":
            team_games = df[df["home_team"] == team]
        else:
            team_games = df[df["away_team"] == team]
        
        if team_games.empty:
            return {}
        
        latest = team_games.iloc[-1]
        
        col_prefix = f"{prefix}_"
        
        return {
            f"{col_prefix}elo": float(latest.get(f"{col_prefix}elo", 1500)),
            f"{col_prefix}form": float(latest.get(f"{col_prefix}form", 1.5)),
            f"{col_prefix}form_decayed": float(latest.get(f"{col_prefix}form_decayed", 1.5)),
            f"{col_prefix}form_ewma": float(latest.get(f"{col_prefix}form_ewma", 1.5)),
            
            f"{col_prefix}goals_scored_avg": float(latest.get(f"{col_prefix}goals_scored_avg", 1.5)),
            f"{col_prefix}goals_conceded_avg": float(latest.get(f"{col_prefix}goals_conceded_avg", 1.2)),
            
            f"{col_prefix}shots_avg": float(latest.get(f"{col_prefix}shots_avg", 12)),
            f"{col_prefix}shots_on_target_avg": float(latest.get(f"{col_prefix}shots_on_target_avg", 4)),
            
            f"{col_prefix}corners_avg": float(latest.get(f"{col_prefix}corners_avg", 5)),
            f"{col_prefix}rest_days": float(latest.get(f"{col_prefix}rest_days", 7)),
            
            f"{col_prefix}attack_adj": float(latest.get(f"{col_prefix}attack_adj", 1.0)),
            f"{col_prefix}defence_adj": float(latest.get(f"{col_prefix}defence_adj", 1.0)),
            f"{col_prefix}conversion_rate": float(latest.get(f"{col_prefix}conversion_rate", 0.12)),
            f"{col_prefix}defensive_compactness": float(latest.get(f"{col_prefix}defensive_compactness", 0.5)),
        }


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + data manager at startup."""
    try:
        app.state.predictor = Predictor()
        app.state.data_manager = HistoricalDataManager()
        logger.success("✅ Model & data manager loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        app.state.predictor = None
        app.state.data_manager = None
    
    yield
    
    logger.info("🛑 Shutting down...")


# ─────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football Match Prediction API",
    version="3.1.0",
    description="Match prediction using historical data (season + matchday + country)",
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
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health():
    """Health check."""
    return HealthResponse(
        status="ok" if app.state.predictor else "error",
        model_loaded=app.state.predictor is not None,
        version="3.1.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={503: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def predict(req: MatchFeaturesRequest):
    """
    Predict match outcome using ONLY historical data from the specified season + matchday.
    """
    
    if not app.state.predictor or not app.state.data_manager:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # ─────────────────────────────────────────
        # 1. Get historical features
        # ─────────────────────────────────────────
        features = app.state.data_manager.get_features_at_matchday(
            season=req.season,
            country=req.country,
            matchday=req.matchday,
            home_team=req.home_team,
            away_team=req.away_team,
        )
        
        # ─────────────────────────────────────────
        # 2. Compute pressure index from standings
        # ─────────────────────────────────────────
        try:
            df = app.state.data_manager.get_season_data(req.season, req.country)
            
            # Get standings at that matchday
            week_col = "week" if "week" in df.columns else "matchday"
            standings_df = df[df[week_col] <= req.matchday].copy()
            
            if not standings_df.empty:
                # Compute pressure indices
                from src.features.match_importance import MatchImportanceCalculator
                
                calc = MatchImportanceCalculator(standings_df)
                pressure_home, pressure_away = calc.calculate(
                    req.home_team,
                    req.away_team,
                    req.matchday
                )
            else:
                pressure_home, pressure_away = 0.5, 0.5
        
        except Exception as e:
            logger.warning(f"Pressure calculation failed: {e}")
            pressure_home, pressure_away = 0.5, 0.5
        
        features["pressure_index_home"] = pressure_home
        features["pressure_index_away"] = pressure_away
        
        # ─────────────────────────────────────────
        # 3. Predict
        # ─────────────────────────────────────────
        result = app.state.predictor.predict(features)
        
        return PredictionResponse(
            home_team=req.home_team,
            away_team=req.away_team,
            season=req.season,
            matchday=req.matchday,
            country=req.country,
            home_win_prob=float(result["home_win_prob"]),
            draw_prob=float(result["draw_prob"]),
            away_win_prob=float(result["away_win_prob"]),
            predicted_outcome=str(result["predicted_outcome"]),
            pressure_index_home=float(pressure_home),
            pressure_index_away=float(pressure_away),
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Data not found: {str(e)}"
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    tags=["Prediction"],
    responses={503: {"model": ErrorResponse}},
)
async def predict_batch(requests_list: List[MatchFeaturesRequest]):
    """Batch predictions using historical data."""
    
    if not app.state.predictor or not app.state.data_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for req in requests_list:
        try:
            result = await predict(req)
            results.append(result)
        except HTTPException as e:
            logger.warning(f"Skipped {req.home_team} vs {req.away_team}: {e.detail}")
            continue
    
    return results