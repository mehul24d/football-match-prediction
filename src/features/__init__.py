"""Feature engineering sub-package."""

from src.features.engineering import (
    EloRatingSystem,
    build_features,
    _compute_form,
    _rolling_team_stats,
)
from src.features.match_importance import MatchImportanceCalculator, add_pressure_features
from src.features.advanced_features import (
    add_opponent_adjusted_metrics,
    add_tactical_features,
    add_interaction_features,
    compute_weighted_form,
)
from src.features.temporal_features import (
    add_lag_features,
    add_ewma_features,
    add_momentum_features,
    build_team_sequences,
)

__all__ = [
    "EloRatingSystem",
    "build_features",
    "_compute_form",
    "_rolling_team_stats",
    "MatchImportanceCalculator",
    "add_pressure_features",
    "add_opponent_adjusted_metrics",
    "add_tactical_features",
    "add_interaction_features",
    "compute_weighted_form",
    "add_lag_features",
    "add_ewma_features",
    "add_momentum_features",
    "build_team_sequences",
]
