"""
src/features/engineering.py
----------------------------
Advanced feature engineering for football match prediction.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
# Elo Rating System
# ─────────────────────────────────────────────

class EloRatingSystem:
    def __init__(self, k_factor=32, initial_rating=1500, home_advantage=100):
        self.k = k_factor
        self.initial = initial_rating
        self.home_adv = home_advantage
        self.ratings = defaultdict(lambda: self.initial)

    def expected(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def expected_score(self, ra: float, rb: float) -> float:
        """Alias for expected() for API clarity."""
        return self.expected(ra, rb)

    def get_rating(self, team: str) -> float:
        """Return current Elo rating for a team."""
        return self.ratings[team]

    def update(self, home, away, result):
        ra = self.ratings[home]
        rb = self.ratings[away]

        exp_home = self.expected(ra + self.home_adv, rb)

        if result == 0:
            s_home = 1
        elif result == 1:
            s_home = 0.5
        else:
            s_home = 0

        r_home_before, r_away_before = ra, rb

        self.ratings[home] += self.k * (s_home - exp_home)
        self.ratings[away] += self.k * ((1 - s_home) - (1 - exp_home))

        return r_home_before, r_away_before


# ─────────────────────────────────────────────
# Rolling helper (NO LEAKAGE)
# ─────────────────────────────────────────────

def _rolling(df, group, col, window):
    return (
        df.groupby(group)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )


def _rolling_team_stats(
    df: pd.DataFrame,
    team_col: str,
    stat_cols: list,
    window: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Compute rolling averages for each team with no lookahead leakage.

    Parameters
    ----------
    df       : DataFrame sorted by date
    team_col : column identifying the team (e.g. 'home_team')
    stat_cols: list of numeric columns to average
    window   : rolling window size
    prefix   : prefix for output column names

    Returns
    -------
    DataFrame with columns ``{prefix}_{col}_avg`` for each col in stat_cols.
    """
    result = pd.DataFrame(index=df.index)
    for col in stat_cols:
        series = df.groupby(team_col)[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        result[f"{prefix}_{col}_avg"] = series
    return result


# ─────────────────────────────────────────────
# Normal form
# ─────────────────────────────────────────────

def _compute_form(df, window, target_col=None):
    if target_col is None:
        if "result_label" in df.columns:
            target_col = "result_label"
        elif "target" in df.columns:
            target_col = "target"
        else:
            raise ValueError("DataFrame must contain 'result_label' or 'target' column.")
    hist = defaultdict(list)
    home_form, away_form = [], []

    for _, row in df.iterrows():
        h, a, r = row["home_team"], row["away_team"], row[target_col]

        hf = np.mean(hist[h][-window:]) if hist[h] else np.nan
        af = np.mean(hist[a][-window:]) if hist[a] else np.nan

        home_form.append(hf)
        away_form.append(af)

        hp = 3 if r == 0 else (1 if r == 1 else 0)
        ap = 3 if r == 2 else (1 if r == 1 else 0)

        hist[h].append(hp)
        hist[a].append(ap)

    return pd.Series(home_form), pd.Series(away_form)


# ─────────────────────────────────────────────
# 🔥 Decayed form (IMPORTANT)
# ─────────────────────────────────────────────

def _decay(days, half_life=30):
    return 2 ** (-days / half_life)


def _compute_decayed_form(df, window=5, half_life=30, target_col=None):
    if target_col is None:
        if "result_label" in df.columns:
            target_col = "result_label"
        elif "target" in df.columns:
            target_col = "target"
        else:
            raise ValueError("DataFrame must contain 'result_label' or 'target' column.")
    history = defaultdict(list)

    home_form = []
    away_form = []

    for _, row in df.iterrows():
        h, a, r, date = (
            row["home_team"],
            row["away_team"],
            row[target_col],
            row["date"],
        )

        def calc(team):
            recs = history[team][-window:]
            if not recs:
                return np.nan

            weights = [_decay((date - d).days, half_life) for d, _ in recs]
            pts = [p for _, p in recs]

            return np.average(pts, weights=weights)

        home_form.append(calc(h))
        away_form.append(calc(a))

        hp = 3 if r == 0 else (1 if r == 1 else 0)
        ap = 3 if r == 2 else (1 if r == 1 else 0)

        history[h].append((date, hp))
        history[a].append((date, ap))

    return pd.Series(home_form), pd.Series(away_form)


# ─────────────────────────────────────────────
# MAIN FEATURE BUILDER
# ─────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    form_window: int = 5,
    elo_k_factor: float = 32,
    elo_initial_rating: float = 1500,
    elo_home_advantage: float = 100,
):
    df = df.copy().sort_values("date").reset_index(drop=True)

    logger.info("Building ADVANCED features...")

    # Auto-detect target column
    if "result_label" in df.columns:
        target_col = "result_label"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise ValueError("DataFrame must contain a 'result_label' or 'target' column")

    # ── Elo ───────────────────────────────────
    elo = EloRatingSystem(
        k_factor=elo_k_factor,
        initial_rating=elo_initial_rating,
        home_advantage=elo_home_advantage,
    )

    home_elo, away_elo = [], []

    for _, row in df.iterrows():
        h, a = elo.update(row["home_team"], row["away_team"], int(row[target_col]))
        home_elo.append(h)
        away_elo.append(a)

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # ── Form ─────────────────────────────────
    hf, af = _compute_form(df, form_window, target_col=target_col)
    df["home_form"] = hf
    df["away_form"] = af
    df["form_diff"] = hf - af

    # ── 🔥 Decayed form (FIXED) ──────────────
    hfd, afd = _compute_decayed_form(df, form_window, target_col=target_col)
    df["home_form_decayed"] = hfd
    df["away_form_decayed"] = afd

    # ── Rolling stats ────────────────────────
    df["home_goals_scored_avg"] = _rolling(df, "home_team", "home_goals", form_window)
    df["home_goals_conceded_avg"] = _rolling(df, "home_team", "away_goals", form_window)

    df["away_goals_scored_avg"] = _rolling(df, "away_team", "away_goals", form_window)
    df["away_goals_conceded_avg"] = _rolling(df, "away_team", "home_goals", form_window)

    df["home_shots_avg"] = _rolling(df, "home_team", "home_shots", form_window)
    df["away_shots_avg"] = _rolling(df, "away_team", "away_shots", form_window)

    df["home_shots_on_target_avg"] = _rolling(df, "home_team", "home_shots_on_target", form_window)
    df["away_shots_on_target_avg"] = _rolling(df, "away_team", "away_shots_on_target", form_window)

    df["home_corners_avg"] = _rolling(df, "home_team", "home_corners", form_window)
    df["away_corners_avg"] = _rolling(df, "away_team", "away_corners", form_window)

    # ── Rest days ───────────────────────────
    df["home_rest_days"] = df.groupby("home_team")["date"].diff().dt.days
    df["away_rest_days"] = df.groupby("away_team")["date"].diff().dt.days

    # ── Advanced features ───────────────────
    df["home_goal_diff"] = df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]
    df["away_goal_diff"] = df["away_goals_scored_avg"] - df["away_goals_conceded_avg"]
    df["goal_diff_diff"] = df["home_goal_diff"] - df["away_goal_diff"]

    df["home_shot_eff"] = df["home_goals_scored_avg"] / (df["home_shots_on_target_avg"] + 1)
    df["away_shot_eff"] = df["away_goals_scored_avg"] / (df["away_shots_on_target_avg"] + 1)

    df["home_pressure"] = df["home_shots_avg"] + df["home_corners_avg"]
    df["away_pressure"] = df["away_shots_avg"] + df["away_corners_avg"]

    df["elo_form_interaction"] = df["elo_diff"] * df["form_diff"]

    # ── Cleanup ─────────────────────────────
    df = df.ffill().fillna(0)

    logger.success(f"Advanced features built: {df.shape[1]} columns")

    return df