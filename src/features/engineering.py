"""
src/features/engineering.py
----------------------------
Feature engineering for football match prediction.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger


# ─── Elo Rating System ───────────────────────────────────────────────────────

class EloRatingSystem:
    def __init__(
        self,
        k_factor: float = 32,
        initial_rating: float = 1500,
        home_advantage: float = 100,
    ) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.ratings = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, home_team: str, away_team: str, result: int):
        r_home = self.ratings[home_team]
        r_away = self.ratings[away_team]

        e_home = self.expected_score(r_home + self.home_advantage, r_away)
        e_away = 1.0 - e_home

        if result == 0:
            s_home, s_away = 1.0, 0.0
        elif result == 1:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        # Store ratings BEFORE update (important)
        r_home_before, r_away_before = r_home, r_away

        self.ratings[home_team] += self.k_factor * (s_home - e_home)
        self.ratings[away_team] += self.k_factor * (s_away - e_away)

        return r_home_before, r_away_before


# ─── Form Calculation (Unified History) ──────────────────────────────────────

def _compute_form(df: pd.DataFrame, window: int):
    team_history = defaultdict(list)

    home_form = pd.Series(np.nan, index=df.index)
    away_form = pd.Series(np.nan, index=df.index)

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        result = row["target"]

        def get_form(team):
            history = team_history[team][-window:]
            if not history:
                return np.nan
            return np.mean(history)

        home_form[idx] = get_form(home_team)
        away_form[idx] = get_form(away_team)

        # Update after computing form
        home_pts = 3 if result == 0 else (1 if result == 1 else 0)
        away_pts = 3 if result == 2 else (1 if result == 1 else 0)

        team_history[home_team].append(home_pts)
        team_history[away_team].append(away_pts)

    return home_form, away_form


# ─── Time Decay Form ─────────────────────────────────────────────────────────

def _decay(days, half_life):
    return 2 ** (-days / half_life)


def _compute_decayed_form(df, window=5, half_life=30):
    history = defaultdict(list)

    home_form = pd.Series(np.nan, index=df.index)
    away_form = pd.Series(np.nan, index=df.index)

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = row["target"]
        date = row["date"]

        def calc(team):
            records = history[team][-window:]
            if not records:
                return np.nan

            weights = [_decay((date - d).days, half_life) for d, _ in records]
            pts = [p for _, p in records]

            return np.average(pts, weights=weights)

        home_form[idx] = calc(home)
        away_form[idx] = calc(away)

        home_pts = 3 if result == 0 else (1 if result == 1 else 0)
        away_pts = 3 if result == 2 else (1 if result == 1 else 0)

        history[home].append((date, home_pts))
        history[away].append((date, away_pts))

    return home_form, away_form


# ─── Rolling Stats (Unified) ─────────────────────────────────────────────────

def _rolling_stats(df, stat_col, window):
    result = pd.Series(np.nan, index=df.index)
    history = defaultdict(list)

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        def avg(team):
            values = history[team][-window:]
            return np.mean(values) if values else np.nan

        result[idx] = avg(home)

        # Update BOTH teams properly
        if stat_col in df.columns:
            history[home].append(row[stat_col])
            history[away].append(row[stat_col])

    return result


# ─── Main Feature Builder ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, form_window=5):
    df = df.copy().sort_values("date").reset_index(drop=True)

    logger.info("Building features...")

    # ── Elo ────────────────────────────────────────────────────────────────
    elo = EloRatingSystem()

    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        r_home, r_away = elo.update(
            row["home_team"],
            row["away_team"],
            int(row["target"]),
        )
        home_elos.append(r_home)
        away_elos.append(r_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # ── Form ───────────────────────────────────────────────────────────────
    home_form, away_form = _compute_form(df, form_window)
    df["home_form"] = home_form
    df["away_form"] = away_form
    df["form_diff"] = df["home_form"] - df["away_form"]

    # ── Decayed Form ───────────────────────────────────────────────────────
    home_d, away_d = _compute_decayed_form(df, form_window)
    df["home_form_decayed"] = home_d
    df["away_form_decayed"] = away_d

    # ── Rest Days (NEW) ────────────────────────────────────────────────────
    df["home_rest_days"] = df.groupby("home_team")["date"].diff().dt.days
    df["away_rest_days"] = df.groupby("away_team")["date"].diff().dt.days

    # ── Interaction Features ───────────────────────────────────────────────
    df["elo_form_interaction"] = df["elo_diff"] * df["form_diff"]

    # ── Fill missing safely ────────────────────────────────────────────────
    df = df.fillna(method="ffill").fillna(0)

    logger.success(f"Features built: {df.shape[1]} columns")

    return df