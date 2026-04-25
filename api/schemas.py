"""
api/schemas.py
--------------
Pydantic v2 request/response schemas for the prediction API (AUTO FEATURES).
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Request schema (SIMPLIFIED + WEEK SUPPORT)
# ─────────────────────────────────────────────

class MatchFeaturesRequest(BaseModel):
    """
    Minimal input for match prediction.

    Features are built automatically inside the API.
    """

    home_team: str = Field(..., description="Name of the home team")
    away_team: str = Field(..., description="Name of the away team")

    # ✅ NEW: optional matchweek for pressure calculation
    week: Optional[int] = Field(
        None,
        ge=1,
        le=38,
        description="Matchweek (optional, auto-detected if not provided)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "week": 34
            }
        }
    }


# ─────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Predicted outcome probabilities + pressure index."""

    home_team: str
    away_team: str

    home_win_prob: float = Field(..., ge=0, le=1)
    draw_prob: float = Field(..., ge=0, le=1)
    away_win_prob: float = Field(..., ge=0, le=1)

    predicted_outcome: str

    # 🔥 UPDATED VERSION
    model_version: str = "3.0.0"

    # 🔥 NEW: Pressure Index
    pressure_index_home: Optional[float] = Field(
        None, ge=0, le=1,
        description="Match importance / pressure index for home team"
    )
    pressure_index_away: Optional[float] = Field(
        None, ge=0, le=1,
        description="Match importance / pressure index for away team"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_win_prob": 0.55,
                "draw_prob": 0.22,
                "away_win_prob": 0.23,
                "predicted_outcome": "Home Win",
                "model_version": "3.0.0",
                "pressure_index_home": 0.82,
                "pressure_index_away": 0.61
            }
        }
    }


# ─────────────────────────────────────────────
# Utility schemas
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str