"""
api/schemas.py
--------------
Pydantic v2 request/response schemas for the prediction API.
Constrained to historical data: season + matchday + country.
"""

from __future__ import annotations

from typing import Optional, List, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Request schema (SEASON + MATCHDAY + COUNTRY)
# ─────────────────────────────────────────────

class MatchFeaturesRequest(BaseModel):
    """
    Match prediction request using ONLY historical data.
    
    The API will query the processed dataset for:
    - Specific season (e.g., "2023-24")
    - Specific matchday/week (e.g., week 15)
    - Specific country/league (e.g., "England")
    
    This ensures predictions use only relevant historical context.
    """

    home_team: str = Field(..., description="Home team name (exact match from dataset)")
    away_team: str = Field(..., description="Away team name (exact match from dataset)")
    
    season: str = Field(
        ...,
        description="Season in format YYYY-YY (e.g., '2023-24')",
        pattern=r"^\d{4}-\d{2}$"
    )
    
    matchday: int = Field(
        ...,
        ge=1,
        le=38,
        description="Matchday/week number (1-38)"
    )
    
    country: Literal["England", "Spain", "Germany", "Italy", "France"] = Field(
        ...,
        description="League country (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "season": "2023-24",
                "matchday": 15,
                "country": "England"
            }
        }
    }


# ─────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Predicted outcome probabilities based on historical data."""

    home_team: str
    away_team: str
    season: str
    matchday: int
    country: str

    home_win_prob: float = Field(..., ge=0, le=1)
    draw_prob: float = Field(..., ge=0, le=1)
    away_win_prob: float = Field(..., ge=0, le=1)

    predicted_outcome: str

    # Historical context
    pressure_index_home: Optional[float] = Field(
        None, ge=0, le=1,
        description="Pressure index based on season standings at matchday"
    )
    pressure_index_away: Optional[float] = Field(
        None, ge=0, le=1,
        description="Pressure index based on season standings at matchday"
    )
    
    model_version: str = "3.1.0"

    model_config = {
        "json_schema_extra": {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "season": "2023-24",
                "matchday": 15,
                "country": "England",
                "home_win_prob": 0.55,
                "draw_prob": 0.22,
                "away_win_prob": 0.23,
                "predicted_outcome": "Home Win",
                "pressure_index_home": 0.72,
                "pressure_index_away": 0.48,
                "model_version": "3.1.0"
            }
        }
    }


# ─────────────────────────────────────────────
# Error response
# ─────────────────────────────────────────────

class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str