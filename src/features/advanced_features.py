"""
src/features/advanced_features.py
-----------------------------------
Advanced feature engineering:
  - Opponent-adjusted metrics (performance relative to opponent strength)
  - Weighted form with exponential decay
  - Tactical proxy features (possession/defensive intensity indicators)
  - Home/away context splits
  - Interaction features (attack vs defence, pressure × form, elo × importance)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─── Opponent-adjusted metrics ────────────────────────────────────────────────

def add_opponent_adjusted_metrics(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add opponent-adjusted offensive and defensive ratings.

    For each match we compare a team's goals scored against the average
    goals conceded by their recent opponents (and vice versa for defence).
    This normalises raw stats for opponent difficulty.
    """
    df = df.copy()

    # Build rolling average of goals conceded per team (as a proxy for
    # opponent defensive quality)
    def _rolling_avg(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=1).mean()

    home_conc_avg = df.groupby("home_team")["home_goals"].transform(_rolling_avg)
    away_conc_avg = df.groupby("away_team")["away_goals"].transform(_rolling_avg)

    # Opponent-adjusted attack: goals scored vs how many the opponent concedes
    eps = 1e-6
    df["home_attack_adj"] = (
        df.groupby("home_team")["home_goals"].transform(_rolling_avg)
        / (away_conc_avg + eps)
    )
    df["away_attack_adj"] = (
        df.groupby("away_team")["away_goals"].transform(_rolling_avg)
        / (home_conc_avg + eps)
    )

    # Opponent-adjusted defence: goals conceded vs how many opponent scores
    home_scored_avg = df.groupby("home_team")["away_goals"].transform(_rolling_avg)
    away_scored_avg = df.groupby("away_team")["home_goals"].transform(_rolling_avg)

    df["home_defence_adj"] = (
        df.groupby("home_team")["away_goals"].transform(_rolling_avg)
        / (away_scored_avg + eps)
    )
    df["away_defence_adj"] = (
        df.groupby("away_team")["home_goals"].transform(_rolling_avg)
        / (home_scored_avg + eps)
    )

    logger.debug("Opponent-adjusted metrics added.")
    return df


# ─── Tactical proxy features ─────────────────────────────────────────────────

def add_tactical_features(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add tactical proxy features from available statistics.

    These approximate (without detailed tracking data):
      - Shot conversion efficiency
      - Defensive compactness (goals conceded per shot faced)
      - Aerial / set-piece threat via corner volume
    """
    df = df.copy()

    def _rolling(group_col: str, val_col: str) -> pd.Series:
        return df.groupby(group_col)[val_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    eps = 1e-6

    # Shot conversion efficiency (goals / shots on target)
    home_sot = _rolling("home_team", "home_shots_on_target")
    away_sot = _rolling("away_team", "away_shots_on_target")
    home_gls = _rolling("home_team", "home_goals")
    away_gls = _rolling("away_team", "away_goals")

    df["home_conversion_rate"] = home_gls / (home_sot + eps)
    df["away_conversion_rate"] = away_gls / (away_sot + eps)

    # Defensive compactness: goals conceded per shot faced
    home_shots_faced = _rolling("home_team", "away_shots")
    away_shots_faced = _rolling("away_team", "home_shots")
    home_conc = _rolling("home_team", "away_goals")
    away_conc = _rolling("away_team", "home_goals")

    df["home_defensive_compactness"] = home_conc / (home_shots_faced + eps)
    df["away_defensive_compactness"] = away_conc / (away_shots_faced + eps)

    # Set-piece threat proxy: corners per match
    if "home_corners" in df.columns and "away_corners" in df.columns:
        df["home_corner_rate"] = _rolling("home_team", "home_corners")
        df["away_corner_rate"] = _rolling("away_team", "away_corners")

    logger.debug("Tactical proxy features added.")
    return df


# ─── Interaction features ─────────────────────────────────────────────────────

def add_interaction_features(
    df: pd.DataFrame,
    pressure_col_home: str = "pressure_index_home",
    pressure_col_away: str = "pressure_index_away",
) -> pd.DataFrame:
    """
    Add interaction features that combine different signals.

    These capture synergistic effects that individual features miss:
      - Pressure × form (high form under pressure behaves differently)
      - Attack vs defence mismatch
      - Elo difference × pressure importance
    """
    df = df.copy()

    # Attack vs Defence mismatch
    if "home_attack_adj" in df.columns and "away_defence_adj" in df.columns:
        df["attack_vs_defence"] = df["home_attack_adj"] / (df["away_defence_adj"] + 1e-6)
        df["defence_vs_attack"] = df["away_attack_adj"] / (df["home_defence_adj"] + 1e-6)

    # Pressure × form interactions
    if pressure_col_home in df.columns and "home_form" in df.columns:
        df["home_pressure_form"] = df[pressure_col_home] * df["home_form"]
        df["away_pressure_form"] = df[pressure_col_away] * df["away_form"]

    # Elo × pressure (elite teams performing under pressure)
    if pressure_col_home in df.columns and "elo_diff" in df.columns:
        df["elo_pressure_interaction"] = df["elo_diff"] * (
            df[pressure_col_home] - df[pressure_col_away]
        )

    # Home advantage modulated by pressure
    if pressure_col_home in df.columns and "home_form_decayed" in df.columns:
        df["home_form_pressure"] = df["home_form_decayed"] * df[pressure_col_home]
        df["away_form_pressure"] = df["away_form_decayed"] * df[pressure_col_away]

    logger.debug("Interaction features added.")
    return df


# ─── Weighted form with customisable decay ────────────────────────────────────

def compute_weighted_form(
    df: pd.DataFrame,
    window: int = 5,
    alpha: float = 0.5,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add exponentially-weighted form columns using a simple EWMA over points.

    Parameters
    ----------
    df         : DataFrame with home_team, away_team, date, and target column.
    window     : Number of past matches to consider.
    alpha      : EWMA decay factor (higher = more weight on recent matches).
    target_col : Column with integer labels (0=home win, 1=draw, 2=away win).

    Returns
    -------
    DataFrame with ``home_form_ewma`` and ``away_form_ewma`` columns added.
    """
    if target_col is None:
        target_col = "result_label" if "result_label" in df.columns else "target"

    df = df.copy().sort_values("date").reset_index(drop=True)

    history: dict[str, list[float]] = defaultdict(list)
    home_ewma, away_ewma = [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = int(row[target_col])

        # Compute EWMA before updating history
        def _ewma(pts_list: list[float]) -> float:
            recent = pts_list[-window:]
            if not recent:
                return np.nan
            weights = np.array([alpha ** (len(recent) - 1 - i) for i in range(len(recent))])
            return float(np.average(recent, weights=weights))

        home_ewma.append(_ewma(history[home]))
        away_ewma.append(_ewma(history[away]))

        home_pts = 3.0 if result == 0 else (1.0 if result == 1 else 0.0)
        away_pts = 3.0 if result == 2 else (1.0 if result == 1 else 0.0)

        history[home].append(home_pts)
        history[away].append(away_pts)

    df["home_form_ewma"] = home_ewma
    df["away_form_ewma"] = away_ewma

    logger.debug("Exponentially-weighted form columns added.")
    return df
