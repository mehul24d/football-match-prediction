"""
src/features/advanced_features.py
-----------------------------------
Advanced feature engineering:
  - Opponent-adjusted metrics
  - Tactical proxy features
  - Interaction features
  - Weighted form (EWMA)
  - High-signal matchup + pressure features
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
# Opponent-adjusted metrics
# ─────────────────────────────────────────────

def add_opponent_adjusted_metrics(df: pd.DataFrame, window: int = 5):
    df = df.copy()

    def _rolling(series):
        return series.shift(1).rolling(window, min_periods=1).mean()

    eps = 1e-6

    home_conc = df.groupby("home_team")["away_goals"].transform(_rolling)
    away_conc = df.groupby("away_team")["home_goals"].transform(_rolling)

    df["home_attack_adj"] = (
        df.groupby("home_team")["home_goals"].transform(_rolling)
        / (away_conc + eps)
    )

    df["away_attack_adj"] = (
        df.groupby("away_team")["away_goals"].transform(_rolling)
        / (home_conc + eps)
    )

    logger.debug("Opponent-adjusted metrics added.")
    return df


# ─────────────────────────────────────────────
# Tactical features
# ─────────────────────────────────────────────

def add_tactical_features(df: pd.DataFrame, window: int = 5):
    df = df.copy()

    def _rolling(group, col):
        return df.groupby(group)[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    eps = 1e-6

    home_sot = _rolling("home_team", "home_shots_on_target")
    away_sot = _rolling("away_team", "away_shots_on_target")

    home_gls = _rolling("home_team", "home_goals")
    away_gls = _rolling("away_team", "away_goals")

    df["home_conversion_rate"] = home_gls / (home_sot + eps)
    df["away_conversion_rate"] = away_gls / (away_sot + eps)

    logger.debug("Tactical features added.")
    return df


# ─────────────────────────────────────────────
# Interaction features (SAFE)
# ─────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame):
    df = df.copy()

    # Advanced matchup
    if {"home_attack_adj", "away_attack_adj"}.issubset(df.columns):
        df["attack_vs_defence"] = df["home_attack_adj"] / (
            df["away_attack_adj"] + 1e-6
        )

    # Pressure interactions
    if {"pressure_index_home", "pressure_index_away"}.issubset(df.columns):
        if {"home_form", "away_form"}.issubset(df.columns):
            df["home_pressure_form"] = df["pressure_index_home"] * df["home_form"]
            df["away_pressure_form"] = df["pressure_index_away"] * df["away_form"]

        if "elo_diff" in df.columns:
            df["elo_pressure_interaction"] = df["elo_diff"] * (
                df["pressure_index_home"] - df["pressure_index_away"]
            )

    logger.debug("Interaction features added.")
    return df


# ─────────────────────────────────────────────
# Weighted form (EWMA) — FIXED
# ─────────────────────────────────────────────

def compute_weighted_form(
    df: pd.DataFrame,
    window: int = 5,
    alpha: float = 0.5,
    target_col: Optional[str] = None,
):
    # Auto-detect target column
    if target_col is None:
        if "result_label" in df.columns:
            target_col = "result_label"
        elif "target" in df.columns:
            target_col = "target"
        else:
            raise ValueError("Missing target column (result_label/target)")

    df = df.copy().sort_values("date").reset_index(drop=True)

    history = defaultdict(list)
    home_ewma, away_ewma = [], []

    for _, row in df.iterrows():
        h, a, r = row["home_team"], row["away_team"], int(row[target_col])

        def ewma(vals):
            recent = vals[-window:]
            if not recent:
                return np.nan
            weights = np.array([
                alpha ** (len(recent) - 1 - i)
                for i in range(len(recent))
            ])
            return np.average(recent, weights=weights)

        home_ewma.append(ewma(history[h]))
        away_ewma.append(ewma(history[a]))

        hp = 3 if r == 0 else (1 if r == 1 else 0)
        ap = 3 if r == 2 else (1 if r == 1 else 0)

        history[h].append(hp)
        history[a].append(ap)

    df["home_form_ewma"] = home_ewma
    df["away_form_ewma"] = away_ewma

    logger.debug("EWMA form added.")
    return df


# ─────────────────────────────────────────────
# Simple matchup + REQUIRED pipeline features
# ─────────────────────────────────────────────

def add_simple_matchup_features(df: pd.DataFrame):
    df = df.copy()

    # ── Attack vs defence (REQUIRED)
    if {"home_goals_scored_avg", "away_goals_conceded_avg"}.issubset(df.columns):
        df["home_attack_vs_def"] = (
            df["home_goals_scored_avg"] /
            (df["away_goals_conceded_avg"] + 1)
        )
    else:
        df["home_attack_vs_def"] = 0

    if {"away_goals_scored_avg", "home_goals_conceded_avg"}.issubset(df.columns):
        df["away_attack_vs_def"] = (
            df["away_goals_scored_avg"] /
            (df["home_goals_conceded_avg"] + 1)
        )
    else:
        df["away_attack_vs_def"] = 0

    # ── Pressure × form
    if {"pressure_index_home", "home_form"}.issubset(df.columns):
        df["pressure_form_home"] = df["pressure_index_home"] * df["home_form"]
        df["pressure_form_away"] = df["pressure_index_away"] * df["away_form"]
    else:
        df["pressure_form_home"] = 0
        df["pressure_form_away"] = 0

    # ── Tempo (REQUIRED)
    if {"home_shots_avg", "away_shots_avg"}.issubset(df.columns):
        df["tempo_diff"] = df["home_shots_avg"] - df["away_shots_avg"]
    else:
        df["tempo_diff"] = 0

    # ── Control (REQUIRED)
    if {"home_shots_on_target_avg", "away_shots_on_target_avg"}.issubset(df.columns):
        df["control_diff"] = (
            df["home_shots_on_target_avg"]
            - df["away_shots_on_target_avg"]
        )
    else:
        df["control_diff"] = 0

    logger.debug("Simple matchup features added.")
    return df


# ─────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = compute_weighted_form(df)
    df = add_opponent_adjusted_metrics(df)
    df = add_tactical_features(df)
    df = add_interaction_features(df)
    df = add_simple_matchup_features(df)

    logger.success(f"All advanced features added: {df.shape[1]} columns")
    return df