"""
api/schemas.py
--------------
Pydantic v2 request/response schemas for the prediction API.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ─── Request schemas ─────────────────────────────────────────────────────────

class MatchFeaturesRequest(BaseModel):
    """
    Input features for a single match prediction.

    At minimum, provide Elo ratings for both teams.
    All other features are optional and default to sensible values.
    """

    home_team: str = Field(..., description="Name of the home team")
    away_team: str = Field(..., description="Name of the away team")

    # Elo ratings
    home_elo: float = Field(1500.0, ge=0, description="Home team Elo rating")
    away_elo: float = Field(1500.0, ge=0, description="Away team Elo rating")

    # Form (points per game over last N matches, 0–3)
    home_form: Optional[float] = Field(None, ge=0, le=3, description="Home team form (pts/game)")
    away_form: Optional[float] = Field(None, ge=0, le=3, description="Away team form (pts/game)")

    # Time-decayed form
    home_form_decayed: Optional[float] = Field(None, ge=0, le=3)
    away_form_decayed: Optional[float] = Field(None, ge=0, le=3)

    # Rolling averages
    home_goals_scored_avg: Optional[float] = Field(None, ge=0)
    home_goals_conceded_avg: Optional[float] = Field(None, ge=0)
    away_goals_scored_avg: Optional[float] = Field(None, ge=0)
    away_goals_conceded_avg: Optional[float] = Field(None, ge=0)
    home_shots_avg: Optional[float] = Field(None, ge=0)
    away_shots_avg: Optional[float] = Field(None, ge=0)
    home_shots_on_target_avg: Optional[float] = Field(None, ge=0)
    away_shots_on_target_avg: Optional[float] = Field(None, ge=0)
    home_corners_avg: Optional[float] = Field(None, ge=0)
    away_corners_avg: Optional[float] = Field(None, ge=0)

    @model_validator(mode="after")
    def compute_derived(self) -> "MatchFeaturesRequest":
        """Auto-fill derived diff fields."""
        return self

    def to_feature_dict(self) -> dict[str, float]:
        """Convert to flat dict for the Predictor."""
        d: dict[str, float] = {
            "home_elo": self.home_elo,
            "away_elo": self.away_elo,
            "elo_diff": self.home_elo - self.away_elo,
        }
        optional_fields = [
            "home_form", "away_form",
            "home_form_decayed", "away_form_decayed",
            "home_goals_scored_avg", "home_goals_conceded_avg",
            "away_goals_scored_avg", "away_goals_conceded_avg",
            "home_shots_avg", "away_shots_avg",
            "home_shots_on_target_avg", "away_shots_on_target_avg",
            "home_corners_avg", "away_corners_avg",
        ]
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                d[field] = value

        if "home_form" in d and "away_form" in d:
            d["form_diff"] = d["home_form"] - d["away_form"]
        return d

    model_config = {"json_schema_extra": {
        "example": {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_elo": 1650.0,
            "away_elo": 1580.0,
            "home_form": 2.2,
            "away_form": 1.8,
            "home_goals_scored_avg": 2.1,
            "home_goals_conceded_avg": 0.9,
            "away_goals_scored_avg": 1.7,
            "away_goals_conceded_avg": 1.2,
        }
    }}


# ─── Response schemas ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Predicted outcome probabilities for a match."""

    home_team: str
    away_team: str
    home_win_prob: float = Field(..., ge=0, le=1)
    draw_prob: float = Field(..., ge=0, le=1)
    away_win_prob: float = Field(..., ge=0, le=1)
    predicted_outcome: str
    model_version: str = "2.0.0"
    pressure_index_home: Optional[float] = Field(None, ge=0, le=1, description="Match importance / pressure index for the home team")
    pressure_index_away: Optional[float] = Field(None, ge=0, le=1, description="Match importance / pressure index for the away team")

    model_config = {"json_schema_extra": {
        "example": {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_win_prob": 0.48,
            "draw_prob": 0.27,
            "away_win_prob": 0.25,
            "predicted_outcome": "Home Win",
            "model_version": "1.0.0",
        }
    }}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str
