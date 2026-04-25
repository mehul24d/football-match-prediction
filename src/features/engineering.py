"""
src/features/engineering.py
----------------------------
Complete feature engineering pipeline combining:
- Elo Rating System
- Form (recent & decayed)
- Rolling statistics
- League position features
- Advanced features
- Temporal features
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

# 🔥 EXISTING IMPORTS
from src.features.advanced_features import add_advanced_features
from src.features.temporal_features import add_temporal_features


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: ELO RATING SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

class EloRatingSystem:
    """Elo rating system for teams with home advantage."""
    
    def __init__(self, k_factor=32, initial_rating=1500, home_advantage=100):
        self.k = k_factor
        self.initial = initial_rating
        self.home_adv = home_advantage
        self.ratings = defaultdict(lambda: self.initial)

    def expected(self, ra, rb):
        """Calculate expected win probability."""
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, home, away, result):
        """
        Update ratings after a match.
        
        result: 0=home win, 1=draw, 2=away win
        """
        ra = self.ratings[home]
        rb = self.ratings[away]

        exp_home = self.expected(ra + self.home_adv, rb)
        s_home = 1 if result == 0 else (0.5 if result == 1 else 0)

        r_home_before, r_away_before = ra, rb

        self.ratings[home] += self.k * (s_home - exp_home)
        self.ratings[away] += self.k * ((1 - s_home) - (1 - exp_home))

        return r_home_before, r_away_before


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: LEAGUE POSITION FEATURE EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

class LeaguePositionFeatureExtractor:
    """Extract features from league standings at each matchday."""
    
    def __init__(self, standings_df: pd.DataFrame):
        """
        Initialize with standings dataframe.
        
        Parameters:
        -----------
        standings_df : pd.DataFrame
            Combined standings data with columns:
            country, season, matchday, position, team, played, won, drawn, lost,
            goals_for, goals_against, goal_diff, points
        """
        self.standings_df = standings_df.copy()
    
    def get_team_standing_at_matchday(
        self,
        team: str,
        country: str,
        season: str,
        matchday: int,
    ) -> Dict | None:
        """Get team's standing at a specific matchday."""
        query = (
            (self.standings_df['team'] == team) &
            (self.standings_df['country'] == country) &
            (self.standings_df['season'] == season) &
            (self.standings_df['matchday'] == matchday)
        )
        
        result = self.standings_df[query]
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        return {
            'position': int(row['position']),
            'points': int(row['points']),
            'played': int(row['played']),
            'won': int(row['won']),
            'drawn': int(row['drawn']),
            'lost': int(row['lost']),
            'goals_for': int(row['goals_for']),
            'goals_against': int(row['goals_against']),
            'goal_diff': int(row['goal_diff']),
        }
    
    def get_position_differential(self, home_team: str, away_team: str, 
                                 country: str, season: str, matchday: int) -> float:
        """Position differential (away - home). Positive = away team better."""
        home_standing = self.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_standing = self.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        if not home_standing or not away_standing:
            return np.nan
        
        return away_standing['position'] - home_standing['position']
    
    def get_points_differential(self, home_team: str, away_team: str,
                               country: str, season: str, matchday: int) -> float:
        """Points differential (home - away). Positive = home team better."""
        home_standing = self.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_standing = self.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        if not home_standing or not away_standing:
            return np.nan
        
        return home_standing['points'] - away_standing['points']
    
    def get_goal_diff_differential(self, home_team: str, away_team: str,
                                  country: str, season: str, matchday: int) -> float:
        """Goal difference differential (home - away)."""
        home_standing = self.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_standing = self.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        if not home_standing or not away_standing:
            return np.nan
        
        return home_standing['goal_diff'] - away_standing['goal_diff']
    
    def get_win_rate(self, team: str, country: str, season: str, matchday: int) -> float:
        """Win rate (wins / games played)."""
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        if not standing or standing['played'] == 0:
            return 0.0
        return standing['won'] / standing['played']
    
    def get_draw_rate(self, team: str, country: str, season: str, matchday: int) -> float:
        """Draw rate (draws / games played)."""
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        if not standing or standing['played'] == 0:
            return 0.0
        return standing['drawn'] / standing['played']
    
    def get_loss_rate(self, team: str, country: str, season: str, matchday: int) -> float:
        """Loss rate (losses / games played)."""
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        if not standing or standing['played'] == 0:
            return 0.0
        return standing['lost'] / standing['played']
    
    def get_goals_per_game(self, team: str, country: str, season: str, matchday: int) -> float:
        """Average goals scored per game."""
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        if not standing or standing['played'] == 0:
            return 0.0
        return standing['goals_for'] / standing['played']
    
    def get_goals_conceded_per_game(self, team: str, country: str, season: str, matchday: int) -> float:
        """Average goals conceded per game."""
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        if not standing or standing['played'] == 0:
            return 0.0
        return standing['goals_against'] / standing['played']
    
    def get_points_per_game(self, team: str, country: str, season: str, matchday: int) -> float:
        """Average points per game."""
        standing = self.get_team_standing_at_matchday(team, country, season, matchday)
        if not standing or standing['played'] == 0:
            return 0.0
        return standing['points'] / standing['played']


# ═════════════════════════════════════════════════════════════════════════════
# PART 3: HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _rolling(df, group, col, window):
    """Compute rolling average grouped by team."""
    return (
        df.groupby(group)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )


def _compute_form(df, window, target_col):
    """Compute recent form (points average over window)."""
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


def _decay(days, half_life=30):
    """Exponential decay function for recent form weighting."""
    return 2 ** (-days / half_life)


def _compute_decayed_form(df, window, target_col):
    """Compute form with exponential decay (recent games weighted more)."""
    history = defaultdict(list)
    home_form, away_form = [], []

    for _, row in df.iterrows():
        h, a, r = row["home_team"], row["away_team"], row[target_col]
        date = row["date"]
        
        # Ensure date is datetime
        if isinstance(date, str):
            date = pd.to_datetime(date)

        def calc(team):
            recs = history[team][-window:]
            if not recs:
                return np.nan

            weights = [_decay((date - d).days) for d, _ in recs]
            pts = [p for _, p in recs]

            return np.average(pts, weights=weights)

        home_form.append(calc(h))
        away_form.append(calc(a))

        hp = 3 if r == 0 else (1 if r == 1 else 0)
        ap = 3 if r == 2 else (1 if r == 1 else 0)

        history[h].append((date, hp))
        history[a].append((date, ap))

    return pd.Series(home_form), pd.Series(away_form)


# ═════════════════════════════════════════════════════════════════════════════
# PART 4: LEAGUE POSITION FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def add_league_position_features(
    df: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add league position features to matches dataframe.
    
    Features added:
    - home_position, away_position (1-20)
    - home_points, away_points
    - home_goal_diff, away_goal_diff
    - position_differential (away - home)
    - points_differential (home - away)
    - goal_diff_differential (home - away)
    - home_win_rate, away_win_rate
    - home_draw_rate, away_draw_rate
    - home_loss_rate, away_loss_rate
    - home_goals_per_game, away_goals_per_game
    - home_goals_conceded_per_game, away_goals_conceded_per_game
    - home_ppg, away_ppg (points per game)
    """
    df = df.copy()
    
    # Initialize feature extractor
    extractor = LeaguePositionFeatureExtractor(standings_df)
    
    logger.info(f"Adding league position features to {len(df)} matches...")
    
    # Initialize feature columns
    features = {
        'home_position': [],
        'away_position': [],
        'home_points': [],
        'away_points': [],
        'home_goal_diff': [],
        'away_goal_diff': [],
        'position_differential': [],
        'points_differential': [],
        'goal_diff_differential': [],
        'home_win_rate': [],
        'away_win_rate': [],
        'home_draw_rate': [],
        'away_draw_rate': [],
        'home_loss_rate': [],
        'away_loss_rate': [],
        'home_goals_per_game': [],
        'away_goals_per_game': [],
        'home_goals_conceded_per_game': [],
        'away_goals_conceded_per_game': [],
        'home_ppg': [],
        'away_ppg': [],
    }
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        country = row.get('country', '')
        season = row.get('season', '')
        
        # Use latest available matchday for this country/season
        available_matchdays = standings_df[
            (standings_df['country'] == country) &
            (standings_df['season'] == season)
        ]['matchday'].unique()
        
        if len(available_matchdays) > 0:
            matchday = int(np.max(available_matchdays))
        else:
            matchday = 1
        
        # Extract standings
        home_pos = extractor.get_team_standing_at_matchday(home_team, country, season, matchday)
        away_pos = extractor.get_team_standing_at_matchday(away_team, country, season, matchday)
        
        # Handle missing standings
        if home_pos is None or away_pos is None:
            for key in features:
                features[key].append(np.nan)
            continue
        
        # Add features
        features['home_position'].append(home_pos['position'])
        features['away_position'].append(away_pos['position'])
        features['home_points'].append(home_pos['points'])
        features['away_points'].append(away_pos['points'])
        features['home_goal_diff'].append(home_pos['goal_diff'])
        features['away_goal_diff'].append(away_pos['goal_diff'])
        
        features['position_differential'].append(away_pos['position'] - home_pos['position'])
        features['points_differential'].append(home_pos['points'] - away_pos['points'])
        features['goal_diff_differential'].append(home_pos['goal_diff'] - away_pos['goal_diff'])
        
        features['home_win_rate'].append(extractor.get_win_rate(home_team, country, season, matchday))
        features['away_win_rate'].append(extractor.get_win_rate(away_team, country, season, matchday))
        features['home_draw_rate'].append(extractor.get_draw_rate(home_team, country, season, matchday))
        features['away_draw_rate'].append(extractor.get_draw_rate(away_team, country, season, matchday))
        features['home_loss_rate'].append(extractor.get_loss_rate(home_team, country, season, matchday))
        features['away_loss_rate'].append(extractor.get_loss_rate(away_team, country, season, matchday))
        
        features['home_goals_per_game'].append(extractor.get_goals_per_game(home_team, country, season, matchday))
        features['away_goals_per_game'].append(extractor.get_goals_per_game(away_team, country, season, matchday))
        features['home_goals_conceded_per_game'].append(extractor.get_goals_conceded_per_game(home_team, country, season, matchday))
        features['away_goals_conceded_per_game'].append(extractor.get_goals_conceded_per_game(away_team, country, season, matchday))
        
        features['home_ppg'].append(extractor.get_points_per_game(home_team, country, season, matchday))
        features['away_ppg'].append(extractor.get_points_per_game(away_team, country, season, matchday))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} matches")
    
    # Add features to dataframe
    for key, values in features.items():
        df[key] = values
    
    logger.success(f"✅ Added {len(features)} league position features")
    
    return df


# ═════════════════════════════════════════════════════════════════════════════
# PART 5: MAIN FEATURE BUILDER 🚀
# ═════════════════════════════════════════════════════════════════════════════

def build_features(
    df: pd.DataFrame,
    standings_df: pd.DataFrame | None = None,
    form_window: int = 5,
    elo_k_factor: float = 32,
    elo_initial_rating: float = 1500,
    elo_home_advantage: float = 100,
) -> pd.DataFrame:
    """
    Build complete feature set:
    - Elo ratings
    - Form (recent & decayed)
    - Rolling statistics
    - League position features (if standings provided)
    - Advanced features
    - Temporal features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Matches dataframe with columns: date, home_team, away_team, home_goals, 
        away_goals, home_shots, away_shots, home_shots_on_target, away_shots_on_target,
        home_corners, away_corners, result_label or target, country, season
    standings_df : pd.DataFrame, optional
        Combined standings dataframe for league position features
    form_window : int
        Rolling window for form calculation (default: 5)
    elo_k_factor : float
        K-factor for Elo rating (default: 32)
    elo_initial_rating : float
        Initial Elo rating (default: 1500)
    elo_home_advantage : float
        Home advantage bonus in Elo (default: 100)
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with all engineered features
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    
    # ✅ ENSURE DATE IS DATETIME
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Remove any rows with invalid dates
    if df['date'].isna().any():
        logger.warning(f"⚠️  Removing {df['date'].isna().sum()} rows with invalid dates")
        df = df.dropna(subset=['date'])

    logger.info("="*100)
    logger.info("BUILDING COMPLETE FEATURE PIPELINE")
    logger.info("="*100 + "\n")

    # Detect target
    if "result_label" in df.columns:
        target_col = "result_label"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise ValueError("Missing target column (result_label or target)")

    # ─── ELO RATINGS ──────────────────────────────────────────────────────────
    logger.info("1️⃣  Computing Elo ratings...")
    
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
    
    logger.success(f"   ✅ Added Elo features (home_elo, away_elo, elo_diff)\n")

    # ─── FORM ─────────────────────────────────────────────────────────────────
    logger.info("2️⃣  Computing form (recent & decayed)...")
    
    hf, af = _compute_form(df, form_window, target_col)
    df["home_form"] = hf
    df["away_form"] = af
    df["form_diff"] = hf - af

    hfd, afd = _compute_decayed_form(df, form_window, target_col)
    df["home_form_decayed"] = hfd
    df["away_form_decayed"] = afd
    
    logger.success(f"   ✅ Added form features (home_form, away_form, form_diff, decayed)\n")

    # ─── ROLLING STATISTICS ──────────────────────────────────────────────────
    logger.info("3️⃣  Computing rolling statistics...")
    
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
    
    logger.success(f"   ✅ Added rolling statistics (10 features)\n")

    # ─── REST DAYS ────────────────────────────────────────────────────────────
    logger.info("4️⃣  Computing rest days...")
    
    df["home_rest_days"] = df.groupby("home_team")["date"].diff().dt.days
    df["away_rest_days"] = df.groupby("away_team")["date"].diff().dt.days
    
    logger.success(f"   ✅ Added rest days features (home_rest_days, away_rest_days)\n")

    # ─── INTERACTIONS ─────────────────────────────────────────────────────────
    logger.info("5️⃣  Computing interactions...")
    
    df["elo_form_interaction"] = df["elo_diff"] * df["form_diff"]
    
    logger.success(f"   ✅ Added interaction features (elo_form_interaction)\n")

    # ─── LEAGUE POSITION FEATURES ─────────────────────────────────────────────
    if standings_df is not None:
        logger.info("6️⃣  Adding league position features...")
        df = add_league_position_features(df, standings_df)
        logger.success(f"   ✅ Added 20 league position features\n")
    else:
        logger.warning("⚠️  Standings not provided - skipping league position features\n")

    # ─── ADVANCED FEATURES ────────────────────────────────────────────────────
    logger.info("7️⃣  Adding advanced features...")
    df = add_advanced_features(df)
    logger.success(f"   ✅ Advanced features added\n")

    # ─── TEMPORAL FEATURES ────────────────────────────────────────────────────
    logger.info("8️⃣  Adding temporal features...")
    df = add_temporal_features(df)
    logger.success(f"   ✅ Temporal features added\n")

    # ─── CLEANUP ──────────────────────────────────────────────────────────────
    logger.info("9️⃣  Cleaning up...")
    df = df.ffill().fillna(0)
    logger.success(f"   ✅ Forward fill & fillna(0) complete\n")

    # ─── Summary ──────────────────────────────────────────────────────────────
    logger.info("="*100)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*100 + "\n")
    logger.success(f"✅ Final feature count: {df.shape[1]} columns")
    logger.info(f"✅ Final row count: {len(df)} matches\n")

    return df