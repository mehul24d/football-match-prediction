"""
src/features/engineering.py
----------------------------
Feature engineering for football match prediction.

Features computed
-----------------
* Elo ratings (home & away, with home-advantage offset)
* Rolling team form: points per game over last N matches
* Rolling averages: goals scored/conceded, shots, shots on target, corners
* Home/away split averages
* Goal difference rolling average
* Time-decay weighted form (exponential decay)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─── Elo Rating System ───────────────────────────────────────────────────────

class EloRatingSystem:
    """
    Simple Elo rating system for football teams.

    Parameters
    ----------
    k_factor          : Controls how much ratings change after each match.
    initial_rating    : Starting Elo for new teams.
    home_advantage    : Elo points added to home team's expected score.
    """

    def __init__(
        self,
        k_factor: float = 32,
        initial_rating: float = 1500,
        home_advantage: float = 100,
    ) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for player/team A against B (logistic)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        result: int,  # 0 = home win, 1 = draw, 2 = away win
    ) -> tuple[float, float]:
        """
        Update ratings after a match.

        Returns the (home_rating_before, away_rating_before) snapshot used
        for prediction features.
        """
        r_home = self.ratings[home_team]
        r_away = self.ratings[away_team]

        # Apply home-advantage offset to expected score calculation
        e_home = self.expected_score(r_home + self.home_advantage, r_away)
        e_away = 1.0 - e_home

        # Actual scores: win=1, draw=0.5, loss=0
        if result == 0:
            s_home, s_away = 1.0, 0.0
        elif result == 1:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        self.ratings[home_team] = r_home + self.k_factor * (s_home - e_home)
        self.ratings[away_team] = r_away + self.k_factor * (s_away - e_away)

        return r_home, r_away

    def get_rating(self, team: str) -> float:
        return self.ratings[team]


# ─── Rolling statistics helper ───────────────────────────────────────────────

def _rolling_team_stats(
    df: pd.DataFrame,
    team_col: str,
    stat_cols: list[str],
    window: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Compute per-team rolling averages of *stat_cols* over the last *window*
    matches played by that team (home or away), preserving match order.

    Returns a DataFrame with the same index as *df* and new columns named
    ``{prefix}_{stat}``.
    """
    result = pd.DataFrame(index=df.index)
    teams = df[team_col].unique()

    for stat in stat_cols:
        if stat not in df.columns:
            continue
        col_name = f"{prefix}_{stat}_avg"
        values = pd.Series(np.nan, index=df.index)

        for team in teams:
            mask = df[team_col] == team
            team_idx = df.index[mask]
            team_vals = df.loc[mask, stat]
            rolled = team_vals.shift(1).rolling(window=window, min_periods=1).mean()
            values.loc[team_idx] = rolled.values

        result[col_name] = values

    return result


# ─── Form calculator ─────────────────────────────────────────────────────────

def _compute_form(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling points-per-game form for home and away teams
    using each team's last *window* matches across all appearances.

    Returns (home_form_series, away_form_series) aligned to df.index.
    """
    # Build a match-by-match record for each team
    team_results: dict[str, list[tuple[int, float]]] = defaultdict(list)
    # (match_index, points)

    home_form = pd.Series(np.nan, index=df.index)
    away_form = pd.Series(np.nan, index=df.index)

    for idx, row in df.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        result = row.get("result_label")

        if pd.isna(home_team) or pd.isna(away_team) or pd.isna(result):
            continue

        result = int(result)

        # Compute form BEFORE updating with current match
        def _form(team: str) -> float:
            history = team_results[team]
            if not history:
                return np.nan
            recent = [pts for _, pts in history[-window:]]
            return np.mean(recent)

        home_form.loc[idx] = _form(home_team)
        away_form.loc[idx] = _form(away_team)

        # Update history
        home_pts = 3.0 if result == 0 else (1.0 if result == 1 else 0.0)
        away_pts = 3.0 if result == 2 else (1.0 if result == 1 else 0.0)
        team_results[home_team].append((idx, home_pts))
        team_results[away_team].append((idx, away_pts))

    return home_form, away_form


# ─── Time-decay weighted form ─────────────────────────────────────────────────

def _decay_weight(days_ago: float, half_life: float = 30.0) -> float:
    """Exponential decay weight; more recent = higher weight."""
    return 2 ** (-days_ago / half_life)


def _compute_decayed_form(
    df: pd.DataFrame,
    window: int = 5,
    half_life: float = 30.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Like _compute_form but weights each match by recency via exponential decay.
    """
    team_history: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)

    home_form_decayed = pd.Series(np.nan, index=df.index)
    away_form_decayed = pd.Series(np.nan, index=df.index)

    for idx, row in df.iterrows():
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        result = row.get("result_label")
        match_date = row.get("date")

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [home_team, away_team, result]):
            continue

        result = int(result)

        def _weighted_form(team: str) -> float:
            history = team_history[team][-window:]
            if not history:
                return np.nan
            if not isinstance(match_date, pd.Timestamp):
                return np.nan
            weights = [
                _decay_weight((match_date - d).days, half_life)
                for d, _ in history
            ]
            pts = [p for _, p in history]
            total_w = sum(weights)
            if total_w == 0:
                return np.nan
            return sum(w * p for w, p in zip(weights, pts)) / total_w

        home_form_decayed.loc[idx] = _weighted_form(home_team)
        away_form_decayed.loc[idx] = _weighted_form(away_team)

        home_pts = 3.0 if result == 0 else (1.0 if result == 1 else 0.0)
        away_pts = 3.0 if result == 2 else (1.0 if result == 1 else 0.0)

        if isinstance(match_date, pd.Timestamp):
            team_history[home_team].append((match_date, home_pts))
            team_history[away_team].append((match_date, away_pts))

    return home_form_decayed, away_form_decayed


# ─── Main feature builder ────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    form_window: int = 5,
    elo_k_factor: float = 32,
    elo_initial_rating: float = 1500,
    elo_home_advantage: float = 100,
    time_decay_half_life: float = 30.0,
) -> pd.DataFrame:
    """
    Build the full feature set from a preprocessed matches DataFrame.

    Parameters
    ----------
    df                    : output of ``clean_matches()``
    form_window           : rolling window for form/averages
    elo_k_factor          : Elo K-factor
    elo_initial_rating    : Elo starting value for new teams
    elo_home_advantage    : Elo home advantage offset (points)
    time_decay_half_life  : half-life (days) for decay weighting

    Returns
    -------
    DataFrame with all original columns plus engineered features.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    logger.info("Building features…")

    # ── Elo ratings ──────────────────────────────────────────────────────────
    elo = EloRatingSystem(
        k_factor=elo_k_factor,
        initial_rating=elo_initial_rating,
        home_advantage=elo_home_advantage,
    )

    home_elos: list[float] = []
    away_elos: list[float] = []

    for _, row in df.iterrows():
        if pd.isna(row.get("result_label")):
            home_elos.append(np.nan)
            away_elos.append(np.nan)
            continue
        r_home, r_away = elo.update(
            home_team=row["home_team"],
            away_team=row["away_team"],
            result=int(row["result_label"]),
        )
        home_elos.append(r_home)
        away_elos.append(r_away)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # ── Team form (points per game) ───────────────────────────────────────────
    home_form, away_form = _compute_form(df, window=form_window)
    df["home_form"] = home_form
    df["away_form"] = away_form
    df["form_diff"] = df["home_form"] - df["away_form"]

    # ── Time-decayed form ─────────────────────────────────────────────────────
    home_form_d, away_form_d = _compute_decayed_form(
        df, window=form_window, half_life=time_decay_half_life
    )
    df["home_form_decayed"] = home_form_d
    df["away_form_decayed"] = away_form_d

    # ── Rolling averages for home team ───────────────────────────────────────
    home_stats = _rolling_team_stats(
        df,
        team_col="home_team",
        stat_cols=[
            "home_goals", "away_goals",
            "home_shots", "home_shots_on_target", "home_corners",
        ],
        window=form_window,
        prefix="home",
    )
    df = pd.concat([df, home_stats], axis=1)

    # ── Rolling averages for away team ───────────────────────────────────────
    away_stats = _rolling_team_stats(
        df,
        team_col="away_team",
        stat_cols=[
            "away_goals", "home_goals",
            "away_shots", "away_shots_on_target", "away_corners",
        ],
        window=form_window,
        prefix="away",
    )
    df = pd.concat([df, away_stats], axis=1)

    # Rename for clarity
    rename_map = {
        "home_home_goals_avg": "home_goals_scored_avg",
        "home_away_goals_avg": "home_goals_conceded_avg",
        "away_away_goals_avg": "away_goals_scored_avg",
        "away_home_goals_avg": "away_goals_conceded_avg",
        "home_home_shots_avg": "home_shots_avg",
        "home_home_shots_on_target_avg": "home_shots_on_target_avg",
        "home_home_corners_avg": "home_corners_avg",
        "away_away_shots_avg": "away_shots_avg",
        "away_away_shots_on_target_avg": "away_shots_on_target_avg",
        "away_away_corners_avg": "away_corners_avg",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Derived difference features ───────────────────────────────────────────
    for stat in ["goals_scored", "goals_conceded", "shots", "shots_on_target", "corners"]:
        h_col = f"home_{stat}_avg"
        a_col = f"away_{stat}_avg"
        if h_col in df.columns and a_col in df.columns:
            df[f"{stat}_diff_avg"] = df[h_col] - df[a_col]

    logger.success(
        f"Feature engineering complete – {len(df)} rows, {len(df.columns)} columns"
    )
    return df
