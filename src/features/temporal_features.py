"""
src/features/temporal_features.py
-----------------------------------
Temporal feature engineering:
  - Lag features (previous N match outcomes/stats for each team)
  - EWMA (exponentially weighted moving averages) of rolling stats
  - Momentum indicators (win streaks, unbeaten runs, volatility)
  - Sequence preparation for LSTM-based models
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─── Lag features ─────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    lag_cols: list[str],
    lags: list[int] = [1, 2, 3],
    team_cols: tuple[str, str] = ("home_team", "away_team"),
) -> pd.DataFrame:
    """
    Add lagged versions of ``lag_cols`` for both home and away teams.

    Each lagged column is shifted forward by ``lag`` rows *within each team
    group*, so lag-1 is the previous match, lag-2 the one before that, etc.

    Parameters
    ----------
    df        : DataFrame sorted by date.
    lag_cols  : Feature columns to lag (must already be in df).
    lags      : List of lag steps.
    team_cols : (home_team_col, away_team_col).

    Returns
    -------
    DataFrame with new columns ``{col}_lag{n}`` added for each combo.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    home_col, away_col = team_cols

    for col in lag_cols:
        if col not in df.columns:
            logger.warning(f"Lag column '{col}' not in DataFrame – skipping.")
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(home_col)[col].shift(lag)
            # Note: we use home_team grouping for home stats and away_team for away.
            # For a unified lag we'd need a per-team view, but this is a pragmatic
            # approximation that preserves index alignment.

    logger.debug(f"Lag features added: {lag_cols} × lags {lags}")
    return df


# ─── EWMA of stats ─────────────────────────────────────────────────────────────

def add_ewma_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    span: int = 5,
) -> pd.DataFrame:
    """
    Add exponentially-weighted moving average (EWMA) columns for given stats.

    Uses pandas EWMA with the specified ``span`` (half-life = span / 2 roughly).
    Shift(1) is applied to prevent lookahead.

    Parameters
    ----------
    df        : DataFrame with stat columns.
    stat_cols : Columns to compute EWMA for.
    span      : EWMA span parameter.

    Returns
    -------
    DataFrame with new ``{col}_ewma`` columns.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    for col in stat_cols:
        if col not in df.columns:
            logger.warning(f"EWMA column '{col}' not in DataFrame – skipping.")
            continue

        # Determine the team grouping from the column name
        if col.startswith("home_"):
            team_col = "home_team"
        elif col.startswith("away_"):
            team_col = "away_team"
        else:
            team_col = "home_team"  # fallback

        df[f"{col}_ewma"] = df.groupby(team_col)[col].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )

    logger.debug(f"EWMA features added for: {stat_cols} (span={span})")
    return df


# ─── Momentum indicators ──────────────────────────────────────────────────────

def add_momentum_features(
    df: pd.DataFrame,
    window: int = 5,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add momentum indicators:
      - ``home_win_streak``  / ``away_win_streak``   : current win streak
      - ``home_unbeaten_run`` / ``away_unbeaten_run`` : matches without a loss
      - ``home_form_volatility`` / ``away_form_volatility`` : std of recent points

    Parameters
    ----------
    df         : DataFrame with match data, sorted by date.
    window     : Rolling window for volatility.
    target_col : Column with integer result labels (0=home win, 1=draw, 2=away).
    """
    if target_col is None:
        target_col = "result_label" if "result_label" in df.columns else "target"

    df = df.copy().sort_values("date").reset_index(drop=True)

    home_streak, away_streak = [], []
    home_unbeaten, away_unbeaten = [], []
    home_vol, away_vol = [], []

    history: dict[str, list[float]] = defaultdict(list)  # team → list of points

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = int(row[target_col])

        def _streak(pts_list: list[float], win_only: bool) -> int:
            streak = 0
            for p in reversed(pts_list):
                if win_only and p == 3:
                    streak += 1
                elif not win_only and p > 0:  # unbeaten = win or draw
                    streak += 1
                else:
                    break
            return streak

        def _volatility(pts_list: list[float]) -> float:
            recent = pts_list[-window:]
            return float(np.std(recent)) if len(recent) > 1 else 0.0

        home_streak.append(_streak(history[home], win_only=True))
        away_streak.append(_streak(history[away], win_only=True))
        home_unbeaten.append(_streak(history[home], win_only=False))
        away_unbeaten.append(_streak(history[away], win_only=False))
        home_vol.append(_volatility(history[home]))
        away_vol.append(_volatility(history[away]))

        home_pts = 3.0 if result == 0 else (1.0 if result == 1 else 0.0)
        away_pts = 3.0 if result == 2 else (1.0 if result == 1 else 0.0)
        history[home].append(home_pts)
        history[away].append(away_pts)

    df["home_win_streak"] = home_streak
    df["away_win_streak"] = away_streak
    df["home_unbeaten_run"] = home_unbeaten
    df["away_unbeaten_run"] = away_unbeaten
    df["home_form_volatility"] = home_vol
    df["away_form_volatility"] = away_vol

    logger.debug("Momentum features added.")
    return df


# ─── Sequence preparation for LSTM ───────────────────────────────────────────

def build_team_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 5,
    target_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build fixed-length match sequences for each team to train sequential models.

    For each match we look back ``sequence_length`` previous matches for the
    home team and construct a 3-D array suitable for LSTM input.

    Parameters
    ----------
    df              : Feature DataFrame sorted by date.
    feature_cols    : Columns to include in each time-step.
    sequence_length : Number of past matches per sequence.
    target_col      : Label column (0/1/2).

    Returns
    -------
    X : np.ndarray of shape (n_samples, sequence_length, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    if target_col is None:
        target_col = "result_label" if "result_label" in df.columns else "target"

    df = df.sort_values("date").reset_index(drop=True)

    # Build per-team match history
    team_history: dict[str, list[dict]] = defaultdict(list)
    X_list, y_list = [], []

    for _, row in df.iterrows():
        home = row["home_team"]
        feat_vals = [row.get(c, 0.0) for c in feature_cols]
        label = int(row[target_col])

        # Build sequence from home team's history
        hist = team_history[home]
        if len(hist) >= sequence_length:
            seq = [h["features"] for h in hist[-sequence_length:]]
            X_list.append(seq)
            y_list.append(label)

        # Update history
        team_history[home].append({"features": feat_vals, "result": label})

    if not X_list:
        raise ValueError(
            f"No sequences generated. Need at least {sequence_length} "
            "historical matches per team."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    logger.info(f"Built {len(X)} sequences of length {sequence_length}.")
    return X, y
