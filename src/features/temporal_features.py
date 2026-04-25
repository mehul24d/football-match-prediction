"""
src/features/temporal_features.py
-----------------------------------
Temporal feature engineering:
  - Lag features (generic + form lags)
  - EWMA stats
  - Momentum indicators (streaks, volatility)
  - Rolling momentum
  - Sequence prep for LSTM
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
# GENERIC LAG FEATURES (SAFE)
# ─────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    lag_cols: list[str],
    lags: list[int] = [1, 2, 3],
):
    df = df.copy().sort_values("date").reset_index(drop=True)

    for col in lag_cols:
        if col not in df.columns:
            continue

        for lag in lags:
            if col.startswith("home_"):
                df[f"{col}_lag_{lag}"] = df.groupby("home_team")[col].shift(lag)
            elif col.startswith("away_"):
                df[f"{col}_lag_{lag}"] = df.groupby("away_team")[col].shift(lag)

    logger.debug("Generic lag features added.")
    return df


# ─────────────────────────────────────────────
# SIMPLE LAG + MOMENTUM (FIXED NAMES)
# ─────────────────────────────────────────────

def add_simple_temporal_features(df: pd.DataFrame):
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Ensure required columns exist
    if "home_form" not in df.columns or "away_form" not in df.columns:
        logger.warning("Form columns missing → skipping lag features")
        return df

    # ── Lag features (MATCH PIPELINE EXPECTATION ✅)
    for lag in [1, 2, 3]:
        df[f"home_form_lag_{lag}"] = (
            df.groupby("home_team")["home_form"].shift(lag)
        )
        df[f"away_form_lag_{lag}"] = (
            df.groupby("away_team")["away_form"].shift(lag)
        )

    # ── Rolling momentum (SAFE: shift to avoid leakage)
    df["home_momentum"] = (
        df.groupby("home_team")["home_form"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df["away_momentum"] = (
        df.groupby("away_team")["away_form"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    logger.debug("Simple temporal features added.")
    return df


# ─────────────────────────────────────────────
# EWMA FEATURES (SAFE)
# ─────────────────────────────────────────────

def add_ewma_features(
    df: pd.DataFrame,
    stat_cols: list[str],
    span: int = 5,
):
    df = df.copy().sort_values("date").reset_index(drop=True)

    for col in stat_cols:
        if col not in df.columns:
            continue

        if col.startswith("home_"):
            team_col = "home_team"
        elif col.startswith("away_"):
            team_col = "away_team"
        else:
            continue

        df[f"{col}_ewma"] = df.groupby(team_col)[col].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )

    logger.debug("EWMA features added.")
    return df


# ─────────────────────────────────────────────
# MOMENTUM (ADVANCED, NO LEAKAGE)
# ─────────────────────────────────────────────

def add_momentum_features(
    df: pd.DataFrame,
    window: int = 5,
    target_col: Optional[str] = None,
):
    if target_col is None:
        if "result_label" in df.columns:
            target_col = "result_label"
        elif "target" in df.columns:
            target_col = "target"
        else:
            logger.warning("No target column → skipping momentum features")
            return df

    df = df.copy().sort_values("date").reset_index(drop=True)

    history = defaultdict(list)

    home_streak, away_streak = [], []
    home_vol, away_vol = [], []

    for _, row in df.iterrows():
        h, a, r = row["home_team"], row["away_team"], int(row[target_col])

        def streak(vals):
            s = 0
            for v in reversed(vals):
                if v == 3:
                    s += 1
                else:
                    break
            return s

        def volatility(vals):
            return np.std(vals[-window:]) if len(vals) > 1 else 0

        home_streak.append(streak(history[h]))
        away_streak.append(streak(history[a]))

        home_vol.append(volatility(history[h]))
        away_vol.append(volatility(history[a]))

        hp = 3 if r == 0 else (1 if r == 1 else 0)
        ap = 3 if r == 2 else (1 if r == 1 else 0)

        history[h].append(hp)
        history[a].append(ap)

    df["home_win_streak"] = home_streak
    df["away_win_streak"] = away_streak
    df["home_volatility"] = home_vol
    df["away_volatility"] = away_vol

    logger.debug("Advanced momentum features added.")
    return df


# ─────────────────────────────────────────────
# LSTM SEQUENCES (UNCHANGED)
# ─────────────────────────────────────────────

def build_team_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 5,
    target_col: Optional[str] = None,
):
    if target_col is None:
        target_col = "target"

    df = df.sort_values("date").reset_index(drop=True)

    team_history = defaultdict(list)
    X_list, y_list = [], []

    for _, row in df.iterrows():
        team = row["home_team"]
        features = [row.get(c, 0.0) for c in feature_cols]
        label = int(row[target_col])

        hist = team_history[team]

        if len(hist) >= sequence_length:
            seq = hist[-sequence_length:]
            X_list.append(seq)
            y_list.append(label)

        team_history[team].append(features)

    if not X_list:
        raise ValueError("Not enough sequence data")

    return np.array(X_list), np.array(y_list)


# ─────────────────────────────────────────────
# MASTER FUNCTION (FINAL)
# ─────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_simple_temporal_features(df)
    df = add_momentum_features(df)

    # Optional: EWMA (only if needed later)
    # df = add_ewma_features(df, stat_cols=[...])

    df = df.fillna(0)

    logger.success(f"Temporal features added: {df.shape[1]} columns")
    return df